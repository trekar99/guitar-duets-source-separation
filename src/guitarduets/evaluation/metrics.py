from __future__ import annotations

from pathlib import Path

import museval
import torchaudio

from guitarduets.evaluation.sisdr import si_sdr
from guitarduets.evaluation.summarize import summarize_results
from guitarduets.utils.audio import find_audio_file


def evaluate_predictions(predictions_dir: str | Path, manifest_entries: list[dict], model_sources: list[str] | None = None) -> tuple[dict, dict]:
    source_names = model_sources or ["guitar1", "guitar2"]
    predictions_root = Path(predictions_dir)
    results: dict = {}

    for entry in manifest_entries:
        track_name = entry["track_name"]
        prediction_track_dir = predictions_root / track_name
        if not prediction_track_dir.exists():
            continue

        references = [
            torchaudio.load(entry["sources"]["guitar1"])[0].T.numpy(),
            torchaudio.load(entry["sources"]["guitar2"])[0].T.numpy(),
        ]
        estimates = [
            torchaudio.load(find_audio_file(prediction_track_dir, "guitar1"))[0].T.numpy(),
            torchaudio.load(find_audio_file(prediction_track_dir, "guitar2"))[0].T.numpy(),
        ]

        scores = museval.metrics.bss_eval(
            references,
            estimates,
            compute_permutation=True,
            window=int(1.0 * 44100),
            hop=int(1.0 * 44100),
            framewise_filters=False,
            bsseval_sources_version=False,
        )[:-1]
        sisdr_scores = si_sdr(estimates, references, compute_permutation=True, window=44100, hop=44100)

        for idx, source in enumerate(source_names):
            results.setdefault(source, {})
            results[source][track_name] = {
                "SDR": scores[0][idx].tolist(),
                "SIR": scores[1][idx].tolist(),
                "ISR": scores[2][idx].tolist(),
                "SAR": scores[3][idx].tolist(),
                "SI-SDR": sisdr_scores[idx].tolist(),
            }

    return results, summarize_results(results)

