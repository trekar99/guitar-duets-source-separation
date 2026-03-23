from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
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
    note_jitter_ms: float = 50.0
    note_jitter_p: float = 0.5
    use_amp: bool = True
    patience: int = 0
    gradient_clip: float = 5.0
    sum_loss_alpha: float = 0.7
    sum_loss_beta: float = 0.3


def build_augmentation(device: torch.device, use_notes: bool = False):
    audio_only = [FlipSign(), Scale(), OppositePanning()]
    audio_augs = nn.Sequential(*audio_only).to(device)
    flip_channels = FlipChannels().to(device)
    remix = Remix().to(device)
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
    sources, notes = batch
    notes   = notes.to(device, non_blocking=True)
    sources = sources.to(device, non_blocking=True)
    sources = sources[:, 1:, :, :]

    if shift_aug is not None:
        sources, notes = shift_aug(sources, notes)

    if flip_channels is not None:
        if use_notes:
            sources, notes = flip_channels(sources, notes)
        else:
            sources = flip_channels(sources)

    if remix is not None and len(sources) >= 4:
        if use_notes:
            sources, notes = remix(sources, notes)
        else:
            sources = remix(sources)

    if audio_augs is not None and len(sources) >= 4:
        sources = audio_augs(sources)

    if jitter_aug is not None and use_notes:
        notes = jitter_aug(notes)

    mixture = sources.sum(dim=1)
    return mixture, sources, notes


def train_one_epoch(
    model, dataloader, optimizer, loss_fn, device,
    audio_augs=None, flip_channels=None, remix=None,
    shift_aug=None, jitter_aug=None,
    use_sum_loss=False, use_notes=False,
    scaler=None, gradient_clip=0.0,
    sum_loss_alpha=0.7, sum_loss_beta=0.3,
):
    model.train()
    running_loss = 0.0
    steps = 0
    use_amp = scaler is not None

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
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(mixture, notes) if use_notes else model(mixture)
            outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
            labels  = center_trim(labels, outputs)
            loss    = loss_fn(outputs, labels)

            if use_sum_loss:
                loss_sum = torch.nn.functional.l1_loss(
                    outputs.sum(dim=1), labels.sum(dim=1))
                loss = sum_loss_alpha * loss + sum_loss_beta * loss_sum

        if use_amp:
            scaler.scale(loss).backward()
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{running_loss / steps:.4f}")

    return running_loss / max(steps, 1)


def evaluate_validation(model, dataloader, loss_fn, device, use_notes=False, use_amp=False):
    model.eval()
    running_loss = 0.0
    steps = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="valid", leave=False)
        for batch in progress:
            batch_sources, notes = batch
            batch_sources = batch_sources.to(device, non_blocking=True)
            notes         = notes.to(device, non_blocking=True)
            inputs  = batch_sources[:, 0, :, :]
            labels  = batch_sources[:, 1:, :, :]

            with autocast(device_type=device.type, enabled=use_amp):
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
    scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.01)
    loss_fn         = build_pit_l1_loss()

    use_amp = config.use_amp and device.type == "cuda"
    scaler  = GradScaler() if use_amp else None

    audio_augs, flip_channels, remix = build_augmentation(
        device, use_notes=config.use_notes)

    shift_aug  = ShiftWithNotes(shift=8192).to(device) if config.use_notes else None
    jitter_aug = NoteJitter(
        onset_jitter_ms=config.note_jitter_ms,
        duration_range=0.15,
        p=config.note_jitter_p,
    ).to(device) if config.use_notes else None

    history: list[dict] = []
    best_valid = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1}/{config.epochs} (lr={current_lr:.2e})")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device,
            audio_augs=audio_augs,
            flip_channels=flip_channels,
            remix=remix,
            shift_aug=shift_aug,
            jitter_aug=jitter_aug,
            use_sum_loss=config.use_sum_loss,
            use_notes=config.use_notes,
            scaler=scaler,
            gradient_clip=config.gradient_clip,
            sum_loss_alpha=config.sum_loss_alpha,
            sum_loss_beta=config.sum_loss_beta,
        )
        valid_loss = evaluate_validation(
            model, valid_loader, loss_fn, device,
            use_notes=config.use_notes, use_amp=use_amp)

        scheduler.step()

        metrics = {"epoch": epoch, "train_loss": train_loss,
                   "valid_loss": valid_loss, "lr": current_lr}
        history.append(metrics)
        print(f"  train_loss={train_loss:.4f}  valid_loss={valid_loss:.4f}  "
              f"best_valid={min(best_valid, valid_loss):.4f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            patience_counter = 0
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, metrics)
            print(f"  -> saved best checkpoint")
        else:
            patience_counter += 1

        if epoch == 0 or (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, metrics)

        if config.patience > 0 and patience_counter >= config.patience:
            print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
            break

    save_training_history(log_dir, history)
    return history
