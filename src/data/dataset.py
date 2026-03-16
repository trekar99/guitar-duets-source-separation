from __future__ import annotations

import math

import pandas as pd
import torch
import torchaudio
from torch.nn import functional as F
from torch.utils.data import Dataset


def create_tensor_for_segment(csv_path: str, segment_start: int, segment_end: int) -> torch.Tensor:
    dataframe = pd.read_csv(csv_path)
    relevant_rows = dataframe[(dataframe["start_time"] < segment_end) & (dataframe["end_time"] > segment_start)]

    num_notes = 128
    num_guitars = 2
    segment_length = segment_end - segment_start
    table = torch.zeros((num_guitars, num_notes, segment_length), dtype=torch.uint8)

    for _, row in relevant_rows.iterrows():
        instrument_id = int(row["instrument"]) - 1
        note_id = int(row["note"])
        note_start = max(segment_start, int(row["start_time"])) - segment_start
        note_end = min(segment_end, int(row["end_time"])) - segment_start
        table[instrument_id, note_id, note_start:note_end] = 1

    return torch.cat((table[0], table[1]), dim=0)


class GuitarDataset(Dataset):
    def __init__(
        self,
        manifest_entries: list[dict],
        sample_length: int,
        sample_rate: int = 44100,
        stride: int = 44100,
        normalize: bool = False,
        use_notes: bool = False,
    ):
        self.entries = manifest_entries
        self.sample_rate = sample_rate
        self.sample_length = sample_length * sample_rate
        self.stride = stride
        self.normalize = normalize
        self.use_notes = use_notes
        self.num_examples: list[int] = []

        for entry in self.entries:
            track_length = int(entry["length"])
            if self.sample_length is None or track_length < self.sample_length:
                examples = 1
            else:
                examples = int(math.ceil((track_length - self.sample_length) / self.stride) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index: int):
        for entry, examples in zip(self.entries, self.num_examples):
            if index >= examples:
                index -= examples
                continue

            offset = int(self.stride * index)
            num_frames = int(self.sample_length)
            mixture, _ = torchaudio.load(entry["mix"], frame_offset=offset, num_frames=num_frames)
            guitar1, _ = torchaudio.load(entry["sources"]["guitar1"], frame_offset=offset, num_frames=num_frames)
            guitar2, _ = torchaudio.load(entry["sources"]["guitar2"], frame_offset=offset, num_frames=num_frames)
            if self.use_notes:
                if not entry.get("notes_csv"):
                    raise FileNotFoundError(f"Missing notes.csv for conditioned training: {entry['track_name']}")
                notes = create_tensor_for_segment(entry["notes_csv"], segment_start=offset, segment_end=offset + num_frames)
            else:
                notes = torch.zeros((256, num_frames), dtype=torch.uint8)
            example = torch.stack([mixture, guitar1, guitar2])

            if self.normalize:
                example = (example - entry["mean"]) / max(entry["std"], 1e-8)

            example = example[..., : self.sample_length]
            notes = notes[..., : self.sample_length]
            example = F.pad(example, (0, self.sample_length - example.shape[-1]))
            notes = F.pad(notes, (0, self.sample_length - notes.shape[-1]))
            return example, notes

        raise IndexError(index)
        
        
