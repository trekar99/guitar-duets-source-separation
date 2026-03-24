from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
import torch


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
    """Load an audio file and return (waveform, sample_rate).

    Uses soundfile directly to avoid the torchaudio/torchcodec dependency
    introduced in torchaudio 2.5+.  Returns a float32 tensor of shape
    (channels, samples), matching the torchaudio.load() convention.
    """
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile → (samples, channels); we need (channels, samples)
    return torch.from_numpy(np.ascontiguousarray(data.T)), sample_rate


def save_audio(path: str | Path, wav: torch.Tensor, sample_rate: int) -> None:
    """Save a (channels, samples) float32 tensor to a WAV file via soundfile."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (samples, channels)
    data = wav.detach().cpu().numpy().T
    sf.write(str(output_path), data, sample_rate)

