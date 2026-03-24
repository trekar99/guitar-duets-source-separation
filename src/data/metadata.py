from __future__ import annotations

from pathlib import Path

import torch

from src.data.manifests import save_manifest
from src.utils.audio import find_audio_file, load_audio


def build_track_entry(track_dir: str | Path, split: str, repo_root: Path | None = None) -> dict:
    track_path = Path(track_dir)
    mix_path = find_audio_file(track_path, "mix")
    guitar1_path = find_audio_file(track_path, "guitar1")
    guitar2_path = find_audio_file(track_path, "guitar2")
    notes_csv_path = track_path / "notes.csv"

    waveform, sample_rate = load_audio(mix_path)
    mean = torch.mean(waveform).item()
    std = torch.std(waveform).item()

    def _portable(p: Path) -> str:
        """Store relative to repo_root for portability, fallback to absolute."""
        if repo_root is not None:
            try:
                return str(p.resolve().relative_to(repo_root.resolve()))
            except ValueError:
                pass
        return str(p.resolve())

    return {
        "track_name": track_path.name,
        "split": split,
        "root": _portable(track_path),
        "mix": _portable(mix_path),
        "sources": {
            "guitar1": _portable(guitar1_path),
            "guitar2": _portable(guitar2_path),
        },
        "notes_csv": _portable(notes_csv_path) if notes_csv_path.exists() else None,
        "samplerate": sample_rate,
        "length": int(waveform.shape[-1]),
        "mean": mean,
        "std": std,
    }


def build_manifest_from_split_roots(
    split_roots: dict[str, str | Path],
    output_path: str | Path,
    repo_root: Path | None = None,
) -> list[dict]:
    entries: list[dict] = []
    for split, root in split_roots.items():
        root_path = Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Split root does not exist: {root_path}")
        for track_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
            entries.append(build_track_entry(track_dir, split, repo_root=repo_root))
    save_manifest(entries, output_path)
    return entries
