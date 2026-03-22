"""synthesize_score.py
Synthesize sine-wave guitar audio from notes.csv for every track in a dataset.

For each track directory that contains a notes.csv file this script generates:
  - synth_guitar1.wav  (instrument 1 notes)
  - synth_guitar2.wav  (instrument 2 notes)

Both files are mono, 44 100 Hz.  Each note uses a pure sine wave at the MIDI
frequency, amplitude 0.3, with 5 ms linear fade-in and fade-out to avoid clicks.

Usage
-----
    python synthesize_score.py --dataset-root GuitarDuets/Synth/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio

from _bootstrap import bootstrap

repo_root = bootstrap()


# ---------------------------------------------------------------------------
# Core synthesis helpers (also imported by separate_early_fusion.py)
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 44100
AMPLITUDE: float = 0.3
FADE_SAMPLES: int = int(0.005 * SAMPLE_RATE)  # 5 ms


def midi_to_hz(midi_note: float) -> float:
    """Convert a MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def synthesize_note(
    freq: float,
    duration_samples: int,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    fade_samples: int = FADE_SAMPLES,
) -> np.ndarray:
    """Return a mono numpy array with a sine-wave note, faded in and out."""
    t = np.arange(duration_samples) / sample_rate
    wave = amplitude * np.sin(2.0 * np.pi * freq * t)

    # Apply linear fade-in / fade-out (guard against notes shorter than 2x fade)
    actual_fade = min(fade_samples, duration_samples // 2)
    if actual_fade > 0:
        ramp = np.linspace(0.0, 1.0, actual_fade)
        wave[:actual_fade] *= ramp
        wave[-actual_fade:] *= ramp[::-1]

    return wave.astype(np.float32)


def synthesize_instrument_from_notes_csv(
    notes_csv_path: Path,
    instrument_id: int,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    """Synthesise all notes for a single instrument from a notes.csv file.

    Parameters
    ----------
    notes_csv_path:
        Path to notes.csv containing columns: instrument, note, start_time,
        end_time (all in *samples* at `sample_rate`).
    instrument_id:
        1-based instrument identifier to filter rows.
    sample_rate:
        Output audio sample rate.

    Returns
    -------
    torch.Tensor of shape (1, total_samples) — mono audio.
    """
    df = pd.read_csv(notes_csv_path)
    rows = df[df["instrument"] == instrument_id]

    if rows.empty:
        # Return 1 second of silence when there are no notes for this instrument
        return torch.zeros((1, sample_rate), dtype=torch.float32)

    total_samples = int(rows["end_time"].max())
    audio = np.zeros(total_samples, dtype=np.float32)

    for _, row in rows.iterrows():
        start = int(row["start_time"])
        end = int(row["end_time"])
        duration = end - start
        if duration <= 0:
            continue
        freq = midi_to_hz(float(row["note"]))
        note_wave = synthesize_note(freq, duration, sample_rate=sample_rate)
        # Mix (additive) – clip is applied at the very end
        audio[start:end] += note_wave

    # Soft clip to [-1, 1] to handle overlapping notes
    audio = np.clip(audio, -1.0, 1.0)
    return torch.from_numpy(audio).unsqueeze(0)  # (1, T)


def synthesize_track(track_dir: Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Generate synth_guitar1.wav and synth_guitar2.wav inside *track_dir*."""
    notes_csv = track_dir / "notes.csv"
    if not notes_csv.exists():
        return

    for instrument_id, stem in [(1, "synth_guitar1"), (2, "synth_guitar2")]:
        out_path = track_dir / f"{stem}.wav"
        audio = synthesize_instrument_from_notes_csv(notes_csv, instrument_id, sample_rate)
        torchaudio.save(str(out_path), audio, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthesise sine-wave guitar audio from notes.csv files."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Root directory to scan recursively for notes.csv files.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=SAMPLE_RATE,
        help=f"Output sample rate (default: {SAMPLE_RATE}).",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    # Collect all directories that contain a notes.csv
    track_dirs = sorted(
        p.parent for p in dataset_root.rglob("notes.csv")
    )

    if not track_dirs:
        print(f"No notes.csv files found under {dataset_root}")
        return

    print(f"Found {len(track_dirs)} track(s) with notes.csv under {dataset_root}")

    for idx, track_dir in enumerate(track_dirs, start=1):
        print(f"[{idx}/{len(track_dirs)}] Synthesising {track_dir.name} ...", end=" ", flush=True)
        synthesize_track(track_dir, sample_rate=args.sample_rate)
        print("done")

    print("Synthesis complete.")


if __name__ == "__main__":
    main()
