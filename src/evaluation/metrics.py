from __future__ import annotations

from pathlib import Path
from typing import Any

import museval
import numpy as np
import torchaudio

from src.evaluation.sisdr import si_sdr
from src.evaluation.summarize import summarize_results
from src.utils.audio import find_audio_file


def _load_stacked_sources(
    source_paths: list[str | Path],
    track_name: str = "",
    label: str = "",
) -> tuple[np.ndarray, int]:
    """
    Load multiple source waveforms and stack them into shape:
        (time, sources, channels)

    All sources must have the same sample rate and channel count.
    Length mismatches are cropped to the minimum length.
    """
    waveforms = []
    sample_rate = None
    num_channels = None

    for path in source_paths:
        waveform, sr = torchaudio.load(str(path))  # (channels, time)

        # print(
        #     f"    [{track_name}] loaded {label} file {path} "
        #     f"with raw shape={tuple(waveform.shape)}, sr={sr}"
        # )

        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(
                f"Sample rate mismatch: expected {sample_rate}, got {sr} for {path}"
            )

        if num_channels is None:
            num_channels = waveform.shape[0]
        elif waveform.shape[0] != num_channels:
            raise ValueError(
                f"Channel count mismatch: expected {num_channels}, "
                f"got {waveform.shape[0]} for {path}"
            )

        waveform_np = waveform.T.numpy()  # (time, channels)
        # print(
        #     f"    [{track_name}] converted {label} file {path} "
        #     f"to shape={waveform_np.shape}"
        # )
        waveforms.append(waveform_np)

    min_length = min(w.shape[0] for w in waveforms)
    if min_length <= 0:
        raise ValueError("Encountered an empty waveform during evaluation.")

    waveforms = [w[:min_length, :] for w in waveforms]

    # for idx, w in enumerate(waveforms):
    #     print(f"    [{track_name}] cropped {label}[{idx}] shape={w.shape}")

    # Final desired shape: (time, sources, channels)
    stacked = np.stack(waveforms, axis=1)

    # print(
    #     f"    [{track_name}] final stacked {label} shape={stacked.shape}"
    # )

    return stacked, int(sample_rate)


def _nan_list(length: int) -> list[float]:
    return [float("nan")] * length


def _source_is_silent(source_audio: np.ndarray, eps: float = 1e-12) -> bool:
    """
    source_audio shape: (time, channels)
    """
    return float(np.sum(np.abs(source_audio))) <= eps


def evaluate_predictions(
    predictions_dir: str | Path,
    manifest_entries: list[dict],
    model_sources: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_names = model_sources or ["guitar1", "guitar2"]
    predictions_root = Path(predictions_dir)

    results: dict[str, Any] = {}
    missing_tracks: list[str] = []
    failed_tracks: dict[str, str] = {}
    skipped_bss_eval_tracks: dict[str, str] = {}

    total_tracks = len(manifest_entries)

    for track_idx, entry in enumerate(manifest_entries, start=1):
        track_name = entry["track_name"]
        prediction_track_dir = predictions_root / track_name

        print(f"\n[{track_idx}/{total_tracks}] Evaluating track: {track_name}")

        if not prediction_track_dir.exists():
            print(f"  -> missing prediction directory: {prediction_track_dir}")
            missing_tracks.append(track_name)
            continue

        try:
            # print("  -> resolving file paths")
            reference_paths = [entry["sources"][source] for source in source_names]
            estimate_paths = [
                find_audio_file(prediction_track_dir, source) for source in source_names
            ]

            # print(f"  -> reference paths: {reference_paths}")
            # print(f"  -> estimate paths:  {estimate_paths}")

            print("  -> loading references")
            references, sr_ref = _load_stacked_sources(
                reference_paths, track_name=track_name, label="reference"
            )

            print("  -> loading estimates")
            estimates, sr_est = _load_stacked_sources(
                estimate_paths, track_name=track_name, label="estimate"
            )

            # print(
            #     f"  -> loaded references shape={references.shape}, sr={sr_ref}"
            # )
            # print(
            #     f"  -> loaded estimates shape={estimates.shape}, sr={sr_est}"
            # )

            if sr_ref != sr_est:
                raise ValueError(
                    f"Reference/prediction sample-rate mismatch for track {track_name}: "
                    f"{sr_ref} vs {sr_est}"
                )

            if references.shape[2] != estimates.shape[2]:
                raise ValueError(
                    f"Reference/prediction channel mismatch for track {track_name}: "
                    f"{references.shape[2]} vs {estimates.shape[2]}"
                )

            # Align lengths.
            min_length = min(references.shape[0], estimates.shape[0])
            references = references[:min_length, :, :]
            estimates = estimates[:min_length, :, :]

            # print(f"  -> after length align: references={references.shape}, estimates={estimates.shape}")

            # museval expects (sources, time, channels)
            references_museval = np.transpose(references, (1, 0, 2))
            estimates_museval = np.transpose(estimates, (1, 0, 2))

            # print(f"  -> museval reference shape={references_museval.shape}")
            # print(f"  -> museval estimate  shape={estimates_museval.shape}")

            silent_reference_sources = [
                source_names[idx]
                for idx in range(references.shape[1])
                if _source_is_silent(references[:, idx, :])
            ]

            if silent_reference_sources:
                msg = (
                    "Skipped BSS-eval because silent reference sources were present: "
                    + ", ".join(silent_reference_sources)
                )
                print(f"  -> {msg}")
                skipped_bss_eval_tracks[track_name] = msg
            else:
                print("  -> computing museval BSS metrics")
                sdr, sir, isr, sar, permutation = museval.metrics.bss_eval(
                    references_museval,
                    estimates_museval,
                    compute_permutation=True,
                    window=int(1.0 * sr_ref),
                    hop=int(1.0 * sr_ref),
                    framewise_filters=False,
                    bsseval_sources_version=False,
                )

                print(f"  -> museval permutation={permutation}")

                permutation = np.asarray(permutation).reshape(-1).astype(int)
                inv_perm = np.argsort(permutation)
                estimates_aligned = estimates[:, inv_perm, :]

                # print(f"  -> aligned estimate shape for SI-SDR={estimates_aligned.shape}")
                print("  -> computing SI-SDR")

                sisdr_scores_aligned = si_sdr(
                    estimated_sources=estimates_aligned,
                    reference_sources=references,
                    window=int(1.0 * sr_ref),
                    hop=int(1.0 * sr_ref),
                )
                
                num_windows = sdr.shape[1]
                
                track_metrics = {}

                for idx, source in enumerate(source_names):
                    track_metrics[source] = {
                        "SDR": np.asarray(sdr[idx], dtype=np.float64).tolist(),
                        "SIR": np.asarray(sir[idx], dtype=np.float64).tolist(),
                        "ISR": np.asarray(isr[idx], dtype=np.float64).tolist(),
                        "SAR": np.asarray(sar[idx], dtype=np.float64).tolist(),
                        "SI-SDR": np.asarray(sisdr_scores_aligned[idx], dtype=np.float64).tolist(),
                    }

            for source in source_names:
                results.setdefault(source, {})
                results[source][track_name] = track_metrics[source]

            print("  -> done")

        except Exception as exc:
            print(f"  -> FAILED: {type(exc).__name__}: {exc}")
            failed_tracks[track_name] = f"{type(exc).__name__}: {exc}"

    summary = summarize_results(results)
    summary["_meta"] = {
        "tracks_requested": len(manifest_entries),
        "tracks_evaluated": len(
            {track_name for source_tracks in results.values() for track_name in source_tracks}
        ),
        "missing_tracks": missing_tracks,
        "failed_tracks": failed_tracks,
        "skipped_bss_eval_tracks": skipped_bss_eval_tracks,
    }

    return results, summary