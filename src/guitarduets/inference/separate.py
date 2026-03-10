from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from guitarduets.models.apply import apply_model
from guitarduets.utils.audio import load_audio, save_audio
from guitarduets.utils.paths import ensure_dir


def create_tensor_for_segment(csv_path: str, segment_start: int, segment_end: int) -> torch.Tensor:
    dataframe = pd.read_csv(csv_path)
    relevant_rows = dataframe[(dataframe["start_time"] < segment_end) & (dataframe["end_time"] > segment_start)]
    table = torch.zeros((2, 128, segment_end - segment_start), dtype=torch.uint8)

    for _, row in relevant_rows.iterrows():
        instrument_id = int(row["instrument"]) - 1
        note_id = int(row["note"])
        note_start = max(segment_start, int(row["start_time"])) - segment_start
        note_end = min(segment_end, int(row["end_time"])) - segment_start
        table[instrument_id, note_id, note_start:note_end] = 1

    return torch.cat((table[0], table[1]), dim=0)


def separate_tracks(model, manifest_entries: list[dict], output_dir: str | Path, device) -> list[dict]:
    output_root = ensure_dir(output_dir)
    written: list[dict] = []

    for entry in manifest_entries:
        mix, sample_rate = load_audio(entry["mix"])
        notes = create_tensor_for_segment(entry["notes_csv"], segment_start=0, segment_end=mix.shape[-1])
        ref = mix.mean(0)
        normalized = mix - ref.mean()
        normalized = normalized / ref.std()
        model_input = torch.cat((normalized, notes), dim=0)

        with torch.no_grad():
            sources = apply_model(model, model_input[None], progress=False, device=device)[0]

        sources = sources * ref.std()
        sources = sources + ref.mean()

        track_dir = ensure_dir(output_root / entry["track_name"])
        save_audio(track_dir / "mix.wav", mix, sample_rate)
        for source, name in zip(sources, model.sources):
            save_audio(track_dir / f"{name}.wav", source.cpu(), sample_rate)

        written.append(
            {
                "track_name": entry["track_name"],
                "prediction_dir": str(track_dir.resolve()),
            }
        )

    return written
