from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from tqdm.auto import tqdm

from src.training.augment import (
    FlipChannels, FlipSign, NoteJitter, OppositePanning,
    Remix, Scale, Shift, ShiftWithNotes,
)
from src.training.checkpointing import save_checkpoint, save_training_history
from src.training.losses import build_pit_l1_loss
from src.utils.core import center_trim
from src.utils.paths import ensure_dir


@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float = 3e-4
    use_sum_loss: bool = False
    checkpoint_interval: int = 5
    use_notes: bool = False
    note_jitter_ms: float = 50.0   # max jitter in ms applied to notes during training
    note_jitter_p: float = 0.5     # probability of applying jitter each batch


def build_augmentation(device: torch.device, use_notes: bool = False):
    """Build augmentation pipeline.

    When use_notes=True, augmentations that affect note-audio correspondence
    (Shift, FlipChannels, Remix) are handled separately in prepare_batch since
    they need to modify both wav and notes consistently. The nn.Sequential
    returned here only contains audio-only safe augmentations.

    Returns:
        audio_augs:    nn.Sequential of audio-only augmentations (safe for notes)
        flip_channels: FlipChannels instance (handles notes jointly)
        remix:         Remix instance (handles notes jointly), or None
    """
    audio_only = [FlipSign(), Scale(), OppositePanning()]
    audio_augs = nn.Sequential(*audio_only).to(device)

    flip_channels = FlipChannels().to(device)
    remix = Remix().to(device) if not use_notes else Remix().to(device)
    # Always return Remix — it now handles notes correctly when passed

    return audio_augs, flip_channels, remix


def prepare_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
    audio_augs: nn.Module | None = None,
    flip_channels: FlipChannels | None = None,
    remix: Remix | None = None,
    shift_aug: ShiftWithNotes | None = None,
    jitter_aug: NoteJitter | None = None,
    use_notes: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare one batch for training.

    Augmentation order:
      1. ShiftWithNotes  — joint time shift of audio + notes (preserves alignment)
      2. FlipChannels    — L/R flip + guitar1/guitar2 notes swap
      3. Remix           — source shuffling + notes guitar slice permutation
      4. Audio-only augs — FlipSign, Scale, OppositePanning (notes-safe)
      5. NoteJitter      — per-note timing perturbation (notes only)
    """
    sources, notes = batch
    notes   = notes.to(device)
    sources = sources.to(device)
    sources = sources[:, 1:, :, :]   # drop mixture channel, keep stems only

    # 1. Joint time shift
    if shift_aug is not None:
        sources, notes = shift_aug(sources, notes)

    # 2. FlipChannels — swaps L/R audio and guitar1/guitar2 notes
    if flip_channels is not None:
        if use_notes:
            sources, notes = flip_channels(sources, notes)
        else:
            sources = flip_channels(sources)

    # 3. Remix — shuffles sources and guitar notes slices with same permutation
    if remix is not None and len(sources) >= 4:
        if use_notes:
            sources, notes = remix(sources, notes)
        else:
            sources = remix(sources)

    # 4. Audio-only augmentations (notes not involved)
    if audio_augs is not None and len(sources) >= 4:
        sources = audio_augs(sources)

    # 5. Note jitter — per-note timing perturbation
    if jitter_aug is not None and use_notes:
        notes = jitter_aug(notes)

    mixture = sources.sum(dim=1)
    return mixture, sources, notes


def train_one_epoch(
    model, dataloader, optimizer, loss_fn, device,
    audio_augs=None, flip_channels=None, remix=None,
    shift_aug=None, jitter_aug=None,
    use_sum_loss=False, use_notes=False,
):
    model.train()
    alpha = 0.7
    beta  = 0.3
    running_loss = 0.0
    steps = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        mixture, labels, notes = prepare_batch(
            batch, device=device,
            audio_augs=audio_augs,
            flip_channels=flip_channels,
            remix=remix,
            shift_aug=shift_aug,
            jitter_aug=jitter_aug,
            use_notes=use_notes,
        )
        optimizer.zero_grad()
        outputs = model(mixture, notes) if use_notes else model(mixture)
        outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
        labels  = center_trim(labels, outputs)
        loss    = loss_fn(outputs, labels)

        if use_sum_loss:
            loss_sum = torch.nn.functional.l1_loss(
                outputs.sum(dim=1), labels.sum(dim=1))
            loss = alpha * loss + beta * loss_sum

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{running_loss / steps:.4f}")

    return running_loss / max(steps, 1)


def evaluate_validation(model, dataloader, loss_fn, device, use_notes=False):
    model.eval()
    running_loss = 0.0
    steps = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="valid", leave=False)
        for batch in progress:
            batch_sources, notes = batch
            batch_sources = batch_sources.to(device)
            notes         = notes.to(device)
            inputs  = batch_sources[:, 0, :, :]
            labels  = batch_sources[:, 1:, :, :]
            outputs = model(inputs, notes) if use_notes else model(inputs)
            outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
            labels  = center_trim(labels, outputs)
            loss    = loss_fn(outputs, labels)
            running_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")

    return running_loss / max(steps, 1)


def train_model(model, train_loader, valid_loader, checkpoints_dir, log_dir,
                config: TrainConfig, device):
    checkpoints_dir = ensure_dir(checkpoints_dir)
    log_dir         = ensure_dir(log_dir)
    optimizer       = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn         = build_pit_l1_loss()

    # Build augmentation pipeline
    audio_augs, flip_channels, remix = build_augmentation(
        device, use_notes=config.use_notes)

    # Joint audio+notes shift — only used when notes are active
    shift_aug  = ShiftWithNotes(shift=8192).to(device) if config.use_notes else None

    # Note jitter — per-note timing perturbation for real-recording generalisation
    jitter_aug = NoteJitter(
        onset_jitter_ms=config.note_jitter_ms,
        duration_range=0.15,
        p=config.note_jitter_p,
    ).to(device) if config.use_notes else None

    history: list[dict] = []
    best_valid = float("inf")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            audio_augs=audio_augs,
            flip_channels=flip_channels,
            remix=remix,
            shift_aug=shift_aug,
            jitter_aug=jitter_aug,
            use_sum_loss=config.use_sum_loss,
            use_notes=config.use_notes,
        )
        valid_loss = evaluate_validation(
            model, valid_loader, loss_fn, device, use_notes=config.use_notes)

        metrics = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        history.append(metrics)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} "
              f"valid_loss={valid_loss:.4f} best_valid={min(best_valid, valid_loss):.4f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, metrics)
            print(f"saved best checkpoint: {checkpoints_dir / 'best.pt'}")

        if epoch == 0 or (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, metrics)
            print(f"saved periodic checkpoint: {checkpoints_dir / f'epoch_{epoch:03d}.pt'}")

    save_training_history(log_dir, history)
    return history