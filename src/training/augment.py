# Adapted from Demucs (https://github.com/facebookresearch/demucs)
# Copyright (c) Facebook, Inc. and its affiliates.
# Licensed under the MIT License.

"""Data augmentations."""

import random
import torch as th
from torch import nn


class Shift(nn.Module):
    """
    Randomly shift audio in time by up to `shift` samples.

    WARNING: when use_notes=True this augmentation breaks note-audio alignment
    because the notes tensor is not shifted accordingly. Use ShiftWithNotes
    instead, or disable this augmentation when training with notes.
    """
    def __init__(self, shift=8192, same=False):
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        length = time - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                srcs = 1 if self.same else sources
                offsets = th.randint(self.shift, [batch, srcs, 1, 1], device=wav.device)
                offsets = offsets.expand(-1, sources, channels, -1)
                indexes = th.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav


class ShiftWithNotes(nn.Module):
    """
    Randomly shift audio AND notes in time by up to `shift` samples.

    Replaces Shift when use_notes=True. Applies the same random offset to
    both the audio waveform and the notes piano roll so alignment is preserved.

    Args:
        shift:      max shift in samples (default 8192 ~= 186ms at 44.1kHz)
        same:       if True, use the same offset for all sources in a batch item
        notes_sr:   sample rate used when the notes piano roll was built.
                    Used to convert the sample-domain shift to piano-roll frames.
                    If notes are already at audio sample resolution, set to None.
        audio_sr:   audio sample rate (default 44100)
    """
    def __init__(self, shift=8192, same=False, audio_sr=44100):
        super().__init__()
        self.shift    = shift
        self.same     = same
        self.audio_sr = audio_sr

    def forward(self, wav: th.Tensor, notes: th.Tensor):
        """
        Args:
            wav:   (batch, sources, channels, time)
            notes: any tensor whose last dimension is the time axis, e.g.
                   (batch, 256, time) or (batch, 2, 128, time) etc.
        Returns:
            wav_shifted, notes_shifted — both cropped by the same random offset
        """
        batch, sources, channels, time = wav.size()
        length = time - self.shift

        if self.shift <= 0 or not self.training:
            wav_out   = wav[..., :length]
            note_out  = notes[..., :length] if notes is not None else notes
            return wav_out, note_out

        # One scalar offset per batch item
        offsets = th.randint(self.shift, [batch], device=wav.device)  # (batch,)

        # Shift audio: expand offset to (batch, sources, channels, 1)
        audio_off = offsets.view(batch, 1, 1, 1).expand(-1, sources, channels, -1)
        indexes   = th.arange(length, device=wav.device)
        wav_shifted = wav.gather(3, indexes + audio_off)

        # Shift notes along the last (time) axis.
        # We avoid gather() on GPU because the index tensor would be
        # (B, D, note_length) — ~700 MB for typical shapes — causing OOM.
        # Instead we loop over the batch and use plain slicing, which is
        # memory-free and fast enough since batch sizes are small (<=4).
        if notes is not None:
            note_time   = notes.shape[-1]
            note_length = note_time - self.shift
            if note_length > 0:
                pieces = []
                for b in range(batch):
                    off = offsets[b].item()
                    pieces.append(notes[b, ..., off:off + note_length])
                notes_shifted = th.stack(pieces, dim=0)
            else:
                notes_shifted = notes
        else:
            notes_shifted = notes

        return wav_shifted, notes_shifted


class NoteJitter(nn.Module):
    """
    Per-note timing perturbation that simulates human timing imprecision.

    In real recordings, note durations deviate from their notated values.
    A quarter note (value=1.0) might be played as 0.88 or 1.12 beats.
    This augmentation replicates that by perturbing individual note
    onset/offset positions in the piano roll rather than shifting the
    whole roll by a constant offset.

    For each active note region in the piano roll:
      - onset is shifted by ±onset_jitter_frames (small, ~10-30ms)
      - duration is scaled by a random factor in [1-dur_range, 1+dur_range]
        proportional to the note's original duration

    Notes tensor layout assumed: (batch, 256, T) where
      [:, 0:128, :]  = guitar 1 notes
      [:, 128:256, :] = guitar 2 notes

    Args:
        onset_jitter_ms:  max onset shift in ms, default 30ms
        duration_range:   fractional duration perturbation, default 0.15
                          (±15% of original note duration)
        audio_sr:         sample rate, used to convert ms to frames
        notes_fps:        frame rate of the piano roll (frames per second)
        p:                probability of applying jitter per batch item
    """
    def __init__(
        self,
        onset_jitter_ms: float = 30.0,
        duration_range:  float = 0.15,
        audio_sr:        int   = 44100,
        notes_fps:       float = 100.0,
        p:               float = 0.5,
    ):
        super().__init__()
        self.onset_jitter_frames = int(onset_jitter_ms * notes_fps / 1000)
        self.duration_range      = duration_range
        self.p                   = p

    def _perturb_roll(self, roll: th.Tensor) -> th.Tensor:
        """Perturb one (128, T) single-instrument piano roll.

        Finds contiguous active regions per MIDI pitch and shifts/stretches
        each one independently.
        """
        n_notes, T = roll.shape
        out = th.zeros_like(roll)

        for pitch in range(n_notes):
            row = roll[pitch]  # (T,)
            if row.sum() == 0:
                continue

            # Find contiguous active segments (note-on regions)
            # by detecting rising and falling edges
            padded   = th.cat([th.zeros(1, device=row.device), row,
                                th.zeros(1, device=row.device)])
            edges    = padded[1:].float() - padded[:-1].float()
            onsets   = (edges > 0).nonzero(as_tuple=False).squeeze(-1)  # rising edges
            offsets  = (edges < 0).nonzero(as_tuple=False).squeeze(-1)  # falling edges

            for on, off in zip(onsets.tolist(), offsets.tolist()):
                dur = off - on  # original duration in frames

                # Perturb onset: shift by ±onset_jitter_frames
                jitter = random.randint(
                    -self.onset_jitter_frames, self.onset_jitter_frames)
                new_on = max(0, on + jitter)

                # Perturb duration: scale by random factor
                scale    = 1.0 + random.uniform(-self.duration_range,
                                                 self.duration_range)
                new_dur  = max(1, int(round(dur * scale)))
                new_off  = min(T, new_on + new_dur)

                if new_on < new_off:
                    out[pitch, new_on:new_off] = 1.0

        return out

    def forward(self, notes: th.Tensor) -> th.Tensor:
        """
        Args:
            notes: (batch, 256, T) or (batch, 2, 128, T)
        Returns:
            notes with per-note timing perturbation applied, same shape
        """
        if not self.training:
            return notes

        batch = notes.shape[0]
        # Normalise to (batch, 256, T) for processing
        orig_shape = notes.shape
        if notes.dim() == 4:
            notes_flat = notes.reshape(batch, -1, notes.shape[-1])
        else:
            notes_flat = notes  # already (batch, 256, T)

        out = notes_flat.clone()
        for b in range(batch):
            if random.random() > self.p:
                continue
            # Process each guitar's 128-note roll independently
            for g_start in range(0, notes_flat.shape[1], 128):
                g_end = g_start + 128
                out[b, g_start:g_end, :] = self._perturb_roll(
                    notes_flat[b, g_start:g_end, :])

        return out.reshape(orig_shape)


class FlipChannels(nn.Module):
    """Flip left-right channels of audio and swap guitar notes accordingly.

    When L/R channels are swapped, guitar1 (which was panned to one side)
    effectively becomes guitar2 in terms of spatial position. If the model
    uses score conditioning where guitar1 notes are in [:, 0:128, :] and
    guitar2 notes are in [:, 128:256, :], those slices must also be swapped.

    Returns (wav_flipped, notes_flipped). If notes is None, returns only wav.
    """
    def forward(self, wav, notes=None):
        batch, sources, channels, time = wav.size()
        if self.training and channels == 2:
            # Decide per batch item whether to flip
            do_flip = th.randint(2, (batch, 1, 1, 1), device=wav.device).bool()
            do_flip_wav = do_flip.expand(-1, sources, -1, time)

            wav_flipped = wav.clone()
            wav_flipped[do_flip.squeeze(-1).squeeze(-1).squeeze(-1)] =                 wav[do_flip.squeeze(-1).squeeze(-1).squeeze(-1)].flip(2)

            if notes is not None:
                # notes: (batch, 256, T) — swap guitar1 [0:128] and guitar2 [128:256]
                notes_out = notes.clone()
                for b in range(batch):
                    if do_flip[b].item():
                        g1 = notes[b, 0:128, :].clone()
                        g2 = notes[b, 128:256, :].clone()
                        notes_out[b, 0:128, :]   = g2
                        notes_out[b, 128:256, :] = g1
                return wav_flipped, notes_out

            return wav_flipped
        if notes is not None:
            return wav, notes
        return wav


class FlipSign(nn.Module):
    """Random sign flip."""
    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        if self.training:
            signs = th.randint(2, (batch, sources, 1, 1), device=wav.device, dtype=th.float32)
            wav = wav * (2 * signs - 1)
        return wav


class Remix(nn.Module):
    """Shuffle sources to make new mixes, with consistent notes remixing.

    When use_notes=True, the guitar notes must be remixed with the same
    permutation as the audio sources so that guitar1 notes always correspond
    to guitar1 audio and guitar2 notes to guitar2 audio.

    Notes layout: (batch, 256, T) where [0:128]=guitar1, [128:256]=guitar2.
    The permutation shuffles which track's guitar1 is paired with which
    track's guitar2, so we apply the same permutation to the notes slices.

    Returns (wav, notes) if notes is provided, else just wav.
    """
    def __init__(self, proba=1, group_size=4):
        super().__init__()
        self.proba = proba
        self.group_size = group_size

    def forward(self, wav, notes=None):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            group_size = self.group_size or batch
            if batch % group_size != 0:
                raise ValueError(
                    f"Batch size {batch} must be divisible by group size {group_size}")
            groups = batch // group_size

            wav = wav.view(groups, group_size, streams, channels, time)
            permutations = th.argsort(
                th.rand(groups, group_size, streams, 1, 1, device=device), dim=1)
            wav = wav.gather(1, permutations.expand(-1, -1, -1, channels, time))
            wav = wav.view(batch, streams, channels, time)

            if notes is not None:
                # notes: (batch, 256, T)
                # permutations: (groups, group_size, streams, 1, 1)
                # We need a per-item permutation over the batch (one perm per stem)
                # For notes, streams=2 (guitar1, guitar2), we use the same perm
                T_notes = notes.shape[-1]
                notes = notes.view(groups, group_size, 2, 128, T_notes)
                # Use stream-level permutation: perm over group_size dim per stream
                # Extract a (groups, group_size) permutation (take stream 0 as reference)
                perm_idx = permutations[:, :, 0, 0, 0]  # (groups, group_size)
                perm_exp = perm_idx.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
                    -1, -1, 2, 128, T_notes)
                notes = notes.gather(1, perm_exp)
                notes = notes.view(batch, 256, T_notes)

        if notes is not None:
            return wav, notes
        return wav


class Scale(nn.Module):
    def __init__(self, proba=1., min=0.25, max=1.25):
        super().__init__()
        self.proba = proba
        self.min = min
        self.max = max

    def forward(self, wav):
        batch, streams, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.proba:
            scales = th.empty(batch, streams, 1, 1, device=device).uniform_(self.min, self.max)
            wav *= scales
        return wav


class OppositePanning(nn.Module):
    """Apply opposite panning with random offset to paired guitars."""
    def __init__(self, offset_range=(-0.2, 0.2), p=0.2):
        super().__init__()
        self.offset_range = offset_range
        self.p = p

    def forward(self, wav):
        batch, sources, channels, time = wav.size()
        device = wav.device
        if self.training and random.random() < self.p and channels > 1:
            pan_factors = th.rand(batch, sources // 2, 1, 1, device=device) * 2 - 1
            offset = th.empty(
                batch, sources // 2, 1, 1, device=device).uniform_(*self.offset_range)
            pan_factors_opposite = th.clamp(pan_factors + offset, -1, 1)
            pan_factors = th.stack(
                [pan_factors, pan_factors_opposite], dim=2).view(batch, sources, 1, 1)
            gain_left  = th.clamp(1 - pan_factors, 0, 1)
            gain_right = th.clamp(1 + pan_factors, 0, 1)
            wav_left   = wav[:, :, 0:1, :] * gain_left
            wav_right  = wav[:, :, 1:2, :] * gain_right
            wav = th.cat([wav_left, wav_right], dim=2)
        return wav