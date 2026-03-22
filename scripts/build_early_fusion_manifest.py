"""build_early_fusion_manifest.py
Build a manifest JSON for the early-fusion experiment.

The manifest extends the standard format with two extra fields per entry:
    "synth_guitar1"  -- path to synth_guitar1.wav
    "synth_guitar2"  -- path to synth_guitar2.wav

Track splits (based on track name):
    test  : Track1  – Track5
    val   : Track6  – Track10
    train : Track11 – Track36

Usage
-----
    python build_early_fusion_manifest.py \
        --dataset-root GuitarDuets/Synth/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio

from _bootstrap import bootstrap

repo_root = bootstrap()


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def assign_split(track_name: str) -> str:
    """Return 'test', 'val', or 'train' based on the track number."""
    # Extract numeric suffix, e.g. "Track12" -> 12
    digits = "".join(ch for ch in track_name if ch.isdigit())
    if not digits:
        raise ValueError(f"Cannot determine track number from name: {track_name!r}")
    num = int(digits)
    if 1 <= num <= 5:
        return "test"
    if 6 <= num <= 10:
        return "val"
    return "train"


# ---------------------------------------------------------------------------
# Manifest entry builder
# ---------------------------------------------------------------------------

def build_entry(track_dir: Path) -> dict:
    """Build a single manifest entry for *track_dir*."""
    mix_path = track_dir / "mix.wav"
    guitar1_path = track_dir / "guitar1.wav"
    guitar2_path = track_dir / "guitar2.wav"
    notes_csv_path = track_dir / "notes.csv"
    synth_guitar1_path = track_dir / "synth_guitar1.wav"
    synth_guitar2_path = track_dir / "synth_guitar2.wav"

    # Validate required files
    for p in (mix_path, guitar1_path, guitar2_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    for p in (synth_guitar1_path, synth_guitar2_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Synthesised audio missing: {p}. "
                "Run synthesize_score.py first."
            )

    # Load mix to extract stats and length
    waveform, sample_rate = torchaudio.load(str(mix_path))
    mean = torch.mean(waveform).item()
    std = torch.std(waveform).item()
    length = int(waveform.shape[-1])

    track_name = track_dir.name
    split = assign_split(track_name)

    return {
        "track_name": track_name,
        "split": split,
        "root": str(track_dir.resolve()),
        "mix": str(mix_path.resolve()),
        "sources": {
            "guitar1": str(guitar1_path.resolve()),
            "guitar2": str(guitar2_path.resolve()),
        },
        "notes_csv": str(notes_csv_path.resolve()) if notes_csv_path.exists() else None,
        "synth_guitar1": str(synth_guitar1_path.resolve()),
        "synth_guitar2": str(synth_guitar2_path.resolve()),
        "samplerate": sample_rate,
        "length": length,
        "mean": mean,
        "std": std,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the early-fusion manifest from a GuitarDuets/Synth/ root."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the dataset root (e.g. GuitarDuets/Synth/).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Collect track directories (any subdirectory containing mix.wav)
    track_dirs = sorted(
        p.parent for p in dataset_root.rglob("mix.wav")
    )

    if not track_dirs:
        print(f"No track directories found under {dataset_root}")
        return

    print(f"Found {len(track_dirs)} track(s) under {dataset_root}")

    entries: list[dict] = []
    for idx, track_dir in enumerate(track_dirs, start=1):
        print(f"  [{idx}/{len(track_dirs)}] {track_dir.name} ...", end=" ", flush=True)
        entry = build_entry(track_dir)
        entries.append(entry)
        print(f"split={entry['split']}")

    # Save manifest
    output_path = repo_root / "data" / "manifests" / "guitarrecordings_early_fusion.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2)

    # Print split summary
    splits: dict[str, int] = {}
    for entry in entries:
        splits[entry["split"]] = splits.get(entry["split"], 0) + 1
    print(f"\nManifest written to {output_path}")
    for split_name, count in sorted(splits.items()):
        print(f"  {split_name}: {count} track(s)")


if __name__ == "__main__":
    main()
