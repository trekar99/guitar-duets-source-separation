from __future__ import annotations
import math
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from src.utils.audio import load_audio

class GuitarDataset(Dataset):
    def __init__(
        self,
        manifest_entries: list[dict],
        sample_length: int,
        sample_rate: int = 44100,
        stride_seconds: float = 1.0,
        normalize: bool = False,
        use_notes: bool = False,
        preload_to_memory: bool = True
    ):
        self.sample_rate = sample_rate
        self.sample_length = int(sample_length * sample_rate)
        self.stride = int(stride_seconds * sample_rate)
        self.normalize = normalize
        self.use_notes = use_notes
        
        self.audio_data = []
        self.notes_data = []
        self.num_examples = []
        self.track_offsets = [0]

        for entry in manifest_entries:
            # 1. Pre-load Audio Stems
            if preload_to_memory:
                # Load: [Channels, Time]
                mix, _ = load_audio(entry["mix"])
                g1, _ = load_audio(entry["sources"]["guitar1"])
                g2, _ = load_audio(entry["sources"]["guitar2"])
                
                # Stack to: [3, Channels, Time]
                full_audio = torch.stack([mix, g1, g2])
                
                if self.normalize:
                    full_audio = (full_audio - entry["mean"]) / max(entry["std"], 1e-8)
                self.audio_data.append(full_audio)
            else:
                self.audio_data.append(entry)

            # 2. Pre-load and Pre-process CSVs
            if self.use_notes:
                df = pd.read_csv(entry["notes_csv"])
                # Columns: instrument, note, start_time, end_time
                notes_tensor = torch.tensor(df[["instrument", "note", "start_time", "end_time"]].values)
                self.notes_data.append(notes_tensor)
            else:
                self.notes_data.append(None)

            # 3. Calculate Indexing
            track_length = int(entry["length"])
            if track_length <= self.sample_length:
                examples = 1
            else:
                # Standard windowing formula
                examples = int(math.ceil((track_length - self.sample_length) / self.stride) + 1)
            
            self.num_examples.append(examples)
            self.track_offsets.append(self.track_offsets[-1] + examples)

    def __len__(self):
        return self.track_offsets[-1]

    def _get_notes_tensor(self, notes_df_tensor, start, end):
        """Vectorized note-to-grid conversion"""
        num_notes = 128
        num_guitars = 2
        # Use target sample_length to ensure consistent output size
        grid = torch.zeros((num_guitars, num_notes, self.sample_length), dtype=torch.uint8)

        if notes_df_tensor is None:
            return grid.view(-1, self.sample_length)

        # Filter notes within segment
        mask = (notes_df_tensor[:, 2] < end) & (notes_df_tensor[:, 3] > start)
        relevant_notes = notes_df_tensor[mask]

        for row in relevant_notes:
            instr = int(row[0]) - 1
            note = int(row[1])
            # Calculate local boundaries relative to the window 'start'
            n_start = max(start, int(row[2])) - start
            n_end = min(end, int(row[3])) - start
            
            # Bound checking to prevent index errors
            n_start = max(0, n_start)
            n_end = min(self.sample_length, n_end)
            
            if n_start < n_end:
                grid[instr, note, n_start:n_end] = 1

        return grid.view(-1, self.sample_length)

    def __getitem__(self, index: int):
        # Locate which track the index belongs to
        track_idx = 0
        for i in range(len(self.track_offsets) - 1):
            if index < self.track_offsets[i+1]:
                track_idx = i
                break
        
        inner_index = index - self.track_offsets[track_idx]
        start_idx = int(inner_index * self.stride)
        end_idx = start_idx + self.sample_length

        # 1. Handle Audio Slicing
        full_audio = self.audio_data[track_idx]
        
        # Ensure we don't slice past the end of the actual audio tensor
        real_end = min(end_idx, full_audio.shape[-1])
        example = full_audio[:, :, start_idx : real_end]
        
        # 2. Handle Notes Slicing/Generation
        notes = self._get_notes_tensor(self.notes_data[track_idx], start_idx, end_idx)

        # 3. Force Uniform Size (Padding)
        # This solves the RuntimeError: stack expects each tensor to be equal size
        if example.shape[-1] < self.sample_length:
            pad_amt = self.sample_length - example.shape[-1]
            # Pad the time dimension (last dimension)
            example = F.pad(example, (0, pad_amt))
            # Notes are already self.sample_length wide via _get_notes_tensor logic

        return example, notes