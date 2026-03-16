from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torchaudio


def candidate_audio_names(stem: str) -> Iterable[str]:
    return (
        f"{stem}.wav",
        f"{stem} stereo.wav",
        f"{stem}.flac",
        f"{stem}.mp3",
    )


def find_audio_file(track_dir: str | Path, stem: str) -> Path:
    track_path = Path(track_dir)
    for candidate in candidate_audio_names(stem):
        path = track_path / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find audio file for '{stem}' under {track_path}")


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    return torchaudio.load(str(path))


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), wav, sample_rate=sample_rate)

