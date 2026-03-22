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


# ---------------------------------------------------------------------------
# Early-fusion dataset
# ---------------------------------------------------------------------------

class EarlyFusionDataset(Dataset):
    """Dataset for score-informed early-fusion source separation.

    Each item returns:
        input_4ch  -- (4, T) tensor: [stereo_mix (2ch) | synth_g1 (1ch) | synth_g2 (1ch)]
        sources    -- (2, 2, T) tensor: [guitar1 (2ch), guitar2 (2ch)]

    The synth guide tracks (synth_guitar1.wav, synth_guitar2.wav) are mono.
    They are concatenated onto the stereo mix to form a 4-channel model input.
    No separate notes tensor is required because the score information is
    already embedded in the extra audio channels.

    Manifest entries must contain:
        "synth_guitar1"  -- path to synth_guitar1.wav (mono)
        "synth_guitar2"  -- path to synth_guitar2.wav (mono)
    """

    def __init__(
        self,
        manifest_entries: list[dict],
        sample_length: int,
        sample_rate: int = 44100,
        stride: int = 44100,
        normalize: bool = False,
    ):
        self.entries = manifest_entries
        self.sample_rate = sample_rate
        self.sample_length = sample_length * sample_rate
        self.stride = stride
        self.normalize = normalize
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

            # Load stereo mix (2, T)
            mixture, _ = torchaudio.load(entry["mix"], frame_offset=offset, num_frames=num_frames)

            # Load individual guitar sources (2, T each)
            guitar1, _ = torchaudio.load(entry["sources"]["guitar1"], frame_offset=offset, num_frames=num_frames)
            guitar2, _ = torchaudio.load(entry["sources"]["guitar2"], frame_offset=offset, num_frames=num_frames)

            # Load mono synthesised guide tracks (1, T each).
            # Synth files may be shorter than the mix (they end at the last note),
            # so we load the full file and slice/pad rather than using frame_offset.
            synth_g1_full, _ = torchaudio.load(entry["synth_guitar1"])
            synth_g2_full, _ = torchaudio.load(entry["synth_guitar2"])
            synth_g1 = synth_g1_full[..., offset:offset + num_frames]
            synth_g2 = synth_g2_full[..., offset:offset + num_frames]
            # Pad ALL tensors to num_frames before concatenating.
            # Any audio file (mix, sources, or synth) may be shorter than
            # num_frames on the last segment of a track.
            mixture = F.pad(mixture, (0, num_frames - mixture.shape[-1]))
            guitar1 = F.pad(guitar1, (0, num_frames - guitar1.shape[-1]))
            guitar2 = F.pad(guitar2, (0, num_frames - guitar2.shape[-1]))
            synth_g1 = F.pad(synth_g1, (0, num_frames - synth_g1.shape[-1]))
            synth_g2 = F.pad(synth_g2, (0, num_frames - synth_g2.shape[-1]))

            # Optionally normalise using pre-computed mix statistics
            if self.normalize:
                mixture = (mixture - entry["mean"]) / max(entry["std"], 1e-8)
                guitar1 = (guitar1 - entry["mean"]) / max(entry["std"], 1e-8)
                guitar2 = (guitar2 - entry["mean"]) / max(entry["std"], 1e-8)
                # Guide tracks are synthesised — normalise independently
                synth_mean = synth_g1.mean().item()
                synth_std = max(synth_g1.std().item(), 1e-8)
                synth_g1 = (synth_g1 - synth_mean) / synth_std
                synth_mean2 = synth_g2.mean().item()
                synth_std2 = max(synth_g2.std().item(), 1e-8)
                synth_g2 = (synth_g2 - synth_mean2) / synth_std2

            # Build 4-channel input: [mix_L, mix_R, synth_g1, synth_g2]
            input_4ch = torch.cat([mixture, synth_g1, synth_g2], dim=0)  # (4, T)

            # Build (2, 2, T) target: stack sources
            sources = torch.stack([guitar1, guitar2], dim=0)  # (2, 2, T)

            # Truncate / pad to exact segment length
            input_4ch = input_4ch[..., : self.sample_length]
            sources = sources[..., : self.sample_length]
            input_4ch = F.pad(input_4ch, (0, self.sample_length - input_4ch.shape[-1]))
            sources = F.pad(sources, (0, self.sample_length - sources.shape[-1]))

            return input_4ch, sources

        raise IndexError(index)
