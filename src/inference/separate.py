from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.models.apply import apply_model
from src.utils.audio import load_audio, save_audio
from src.utils.paths import ensure_dir


def create_tensor_for_segment(csv_path: str, segment_start: int, segment_end: int) -> torch.Tensor:
    dataframe = pd.read_csv(csv_path)
    relevant_rows = dataframe[
        (dataframe["start_time"] < segment_end) & (dataframe["end_time"] > segment_start)
    ]
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

    total_tracks = len(manifest_entries)

    for track_idx, entry in enumerate(manifest_entries, start=1):
        track_name = entry["track_name"]
        print(f"\n[{track_idx}/{total_tracks}] Separating track: {track_name}")

        mix, sample_rate = load_audio(entry["mix"])

        ref = mix.mean(0)
        ref_mean = ref.mean()
        ref_std = ref.std()

        if torch.isclose(ref_std, torch.tensor(0.0, device=ref_std.device, dtype=ref_std.dtype)):
            raise ValueError(f"Reference standard deviation is zero for track {track_name}")

        normalized = mix - ref_mean
        normalized = normalized / ref_std


        with torch.no_grad():
            if getattr(model, "note_conditioning", False):
                print("  -> note conditioning enabled")

                if not entry.get("notes_csv"):
                    raise FileNotFoundError(
                        f"Missing notes.csv for conditioned inference: {track_name}"
                    )

                notes = create_tensor_for_segment(
                    entry["notes_csv"],
                    segment_start=0,
                    segment_end=mix.shape[-1],
                )

                model_input = torch.cat((normalized, notes), dim=0)

                sources = apply_model(
                    model,
                    model_input[None],
                    progress=False,
                    device=device,
                )[0]
            else:
                print("  -> unconditioned inference")

                sources = apply_model(
                    model,
                    normalized[None],
                    progress=False,
                    device=device,
                )[0]


        sources = sources * ref_std
        sources = sources + ref_mean

        track_dir = ensure_dir(output_root / track_name)
        save_audio(track_dir / "mix.wav", mix, sample_rate)

        for source, name in zip(sources, model.sources):
            save_audio(track_dir / f"{name}.wav", source.cpu(), sample_rate)

        written.append(
            {
                "track_name": track_name,
                "prediction_dir": str(track_dir.resolve()),
            }
        )

        print("  -> done")

    return written