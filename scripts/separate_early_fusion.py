"""separate_early_fusion.py
Run full-track inference with the early-fusion HTDemucs model.

For each test track the script:
1. Loads mix.wav (stereo, 2 channels).
2. Synthesises synth_guitar1 and synth_guitar2 from notes.csv using the same
   sine-wave synthesis as synthesize_score.py.
3. Normalises the stereo mix (mean/std of the mono reference channel).
4. Concatenates:  model_input = [normalised_mix (2ch) | synth_g1 (1ch) | synth_g2 (1ch)]
5. Calls apply_model() — because note_conditioning=False the model receives the
   4-channel tensor directly (no special handling in apply_model needed).
6. Saves guitar1.wav, guitar2.wav, and mix.wav to the output directory.

Usage
-----
    python separate_early_fusion.py \
        --config configs/experiments/eval_early_fusion.yaml \
        --checkpoint artifacts/checkpoints/early_fusion/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap

repo_root = bootstrap()

from data.manifests import load_manifest
from models.apply import apply_model
from models.factory import build_model
from utils.audio import load_audio, save_audio
from utils.io import load_config, save_json
from utils.paths import ensure_dir

# Reuse the synthesis helpers from synthesize_score.py (same scripts/ directory)
from synthesize_score import synthesize_instrument_from_notes_csv, SAMPLE_RATE


# ---------------------------------------------------------------------------
# Per-track inference
# ---------------------------------------------------------------------------

def separate_early_fusion_tracks(
    model,
    manifest_entries: list[dict],
    output_dir: str | Path,
    device,
) -> list[dict]:
    """Separate every entry in *manifest_entries* and write results to *output_dir*.

    Returns a list of dicts suitable for a predictions manifest.
    """
    output_root = ensure_dir(output_dir)
    written: list[dict] = []
    total = len(manifest_entries)

    for track_idx, entry in enumerate(manifest_entries, start=1):
        track_name = entry["track_name"]
        print(f"\n[{track_idx}/{total}] Separating {track_name}")

        # -- 1. Load stereo mix --------------------------------------------------
        mix, sample_rate = load_audio(entry["mix"])   # (2, T)

        # -- 2. Synthesise guide tracks from notes.csv ---------------------------
        notes_csv = entry.get("notes_csv")
        if not notes_csv or not Path(notes_csv).exists():
            raise FileNotFoundError(
                f"Missing notes.csv for early-fusion inference: {track_name}"
            )
        notes_csv_path = Path(notes_csv)
        synth_g1 = synthesize_instrument_from_notes_csv(notes_csv_path, instrument_id=1, sample_rate=sample_rate)  # (1, T)
        synth_g2 = synthesize_instrument_from_notes_csv(notes_csv_path, instrument_id=2, sample_rate=sample_rate)  # (1, T)

        # Align lengths: trim synth to mix length or pad with zeros
        mix_len = mix.shape[-1]
        if synth_g1.shape[-1] > mix_len:
            synth_g1 = synth_g1[..., :mix_len]
        else:
            pad = mix_len - synth_g1.shape[-1]
            synth_g1 = torch.nn.functional.pad(synth_g1, (0, pad))

        if synth_g2.shape[-1] > mix_len:
            synth_g2 = synth_g2[..., :mix_len]
        else:
            pad = mix_len - synth_g2.shape[-1]
            synth_g2 = torch.nn.functional.pad(synth_g2, (0, pad))

        # -- 3. Normalise the mix (same as standard separate.py) -----------------
        ref = mix.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()
        if torch.isclose(ref_std, torch.tensor(0.0)):
            raise ValueError(f"Reference std is zero for track {track_name}")
        normalised_mix = (mix - ref_mean) / ref_std

        # -- 4. Build 4-channel model input -------------------------------------
        model_input = torch.cat([normalised_mix, synth_g1, synth_g2], dim=0)  # (4, T)

        # -- 5. Run apply_model -------------------------------------------------
        # note_conditioning=False → apply_model calls model(padded_mix) directly,
        # passing the full 4-channel tensor through to the network.
        print("  -> early-fusion inference (4-channel input)")
        with torch.no_grad():
            sources = apply_model(
                model,
                model_input[None],   # add batch dim → (1, 4, T)
                progress=False,
                device=device,
            )[0]                     # → (num_sources, 2, T)

        # -- 6. Denormalise and save --------------------------------------------
        sources = sources * ref_std + ref_mean

        track_dir = ensure_dir(output_root / track_name)
        save_audio(track_dir / "mix.wav", mix, sample_rate)
        for source, name in zip(sources, model.sources):
            save_audio(track_dir / f"{name}.wav", source.cpu(), sample_rate)

        written.append({
            "track_name": track_name,
            "prediction_dir": str(track_dir.resolve()),
        })
        print("  -> done")

    return written


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def load_checkpoint_into_model(model, checkpoint_path: str | Path):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
    else:
        model.load_state_dict(payload)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run full-track inference with the early-fusion HTDemucs model."
    )
    parser.add_argument("--config", required=True, help="Path to eval_early_fusion.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_entries = load_manifest(repo_root / config["dataset"]["manifest"])
    test_entries = [
        e for e in manifest_entries if e["split"] == config["dataset"]["test_split"]
    ]
    print(f"Test tracks: {len(test_entries)}")

    # Build and load the early-fusion model
    model = build_model(config["model"]["name"], config["model"].get("kwargs", {}))
    model = load_checkpoint_into_model(model, args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint_stem = Path(args.checkpoint).stem
    output_dir = repo_root / "artifacts" / "predictions" / config["run"]["name"] / checkpoint_stem

    written = separate_early_fusion_tracks(model, test_entries, output_dir, device)
    save_json(output_dir / "predictions_manifest.json", written)
    print(f"\nWrote predictions to {output_dir}")


if __name__ == "__main__":
    main()
