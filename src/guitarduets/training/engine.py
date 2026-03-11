from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from tqdm.auto import tqdm

from guitarduets.training.augment import FlipChannels, FlipSign, OppositePanning, Remix, Scale, Shift
from guitarduets.training.checkpointing import save_checkpoint, save_training_history
from guitarduets.training.losses import build_pit_l1_loss
from guitarduets.utils.core import center_trim
from guitarduets.utils.paths import ensure_dir


@dataclass
class TrainConfig:
    epochs: int
    learning_rate: float = 3e-4
    use_sum_loss: bool = False
    checkpoint_interval: int = 5
    use_notes: bool = False


def build_augmentation(device: torch.device) -> nn.Sequential:
    augmentations = [Shift(), FlipSign(), FlipChannels(), Scale(), Remix(), OppositePanning()]
    return nn.Sequential(*augmentations).to(device)


def prepare_batch(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device, augmentation: nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sources, notes = batch
    notes = notes.to(device)
    sources = sources.to(device)
    sources = sources[:, 1:, :, :]
    if augmentation is not None and len(sources) >= 4:
        sources = augmentation(sources)
    mixture = sources.sum(dim=1)
    return mixture, sources, notes


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, augmentation=None, use_sum_loss=False, use_notes=False):
    model.train()
    alpha = 0.7
    beta = 0.3
    running_loss = 0.0
    steps = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for batch in progress:
        mixture, labels, notes = prepare_batch(batch, device=device, augmentation=augmentation)
        optimizer.zero_grad()
        outputs = model(mixture, notes) if use_notes else model(mixture)
        outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
        labels = center_trim(labels, outputs)
        loss = loss_fn(outputs, labels)

        if use_sum_loss:
            loss_sum = torch.nn.functional.l1_loss(outputs.sum(dim=1), labels.sum(dim=1))
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
            notes = notes.to(device)
            inputs = batch_sources[:, 0, :, :]
            labels = batch_sources[:, 1:, :, :]
            outputs = model(inputs, notes) if use_notes else model(inputs)
            outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
            labels = center_trim(labels, outputs)
            loss = loss_fn(outputs, labels)
            running_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")

    return running_loss / max(steps, 1)


def train_model(model, train_loader, valid_loader, checkpoints_dir, log_dir, config: TrainConfig, device):
    checkpoints_dir = ensure_dir(checkpoints_dir)
    log_dir = ensure_dir(log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = build_pit_l1_loss()
    augmentation = build_augmentation(device)

    history: list[dict] = []
    best_valid = float("inf")

    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            augmentation=augmentation,
            use_sum_loss=config.use_sum_loss,
            use_notes=config.use_notes,
        )
        valid_loss = evaluate_validation(model, valid_loader, loss_fn, device, use_notes=config.use_notes)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        history.append(metrics)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} best_valid={min(best_valid, valid_loss):.4f}")

        if valid_loss < best_valid:
            best_valid = valid_loss
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, metrics)
            print(f"saved best checkpoint: {checkpoints_dir / 'best.pt'}")

        if epoch == 0 or (epoch + 1) % config.checkpoint_interval == 0:
            save_checkpoint(checkpoints_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, metrics)
            print(f"saved periodic checkpoint: {checkpoints_dir / f'epoch_{epoch:03d}.pt'}")

    save_training_history(log_dir, history)
    return history
