from __future__ import annotations

from pathlib import Path

import torch
import torchaudio

from guitarduets.data.manifests import save_manifest
from guitarduets.utils.audio import find_audio_file


def build_track_entry(track_dir: str | Path, split: str) -> dict:
    track_path = Path(track_dir)
    mix_path = find_audio_file(track_path, "mix")
    guitar1_path = find_audio_file(track_path, "guitar1")
    guitar2_path = find_audio_file(track_path, "guitar2")
    notes_csv_path = track_path / "notes.csv"

    waveform, sample_rate = torchaudio.load(str(mix_path))
    mean = torch.mean(waveform).item()
    std = torch.std(waveform).item()

    return {
        "track_name": track_path.name,
        "split": split,
        "root": str(track_path.resolve()),
        "mix": str(mix_path.resolve()),
        "sources": {
            "guitar1": str(guitar1_path.resolve()),
            "guitar2": str(guitar2_path.resolve()),
        },
        "notes_csv": str(notes_csv_path.resolve()) if notes_csv_path.exists() else None,
        "samplerate": sample_rate,
        "length": int(waveform.shape[-1]),
        "mean": mean,
        "std": std,
    }


def build_manifest_from_split_roots(split_roots: dict[str, str | Path], output_path: str | Path) -> list[dict]:
    entries: list[dict] = []
    for split, root in split_roots.items():
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Split root does not exist: {root_path}")
        for track_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            entries.append(build_track_entry(track_dir, split))
    save_manifest(entries, output_path)
    return entries
