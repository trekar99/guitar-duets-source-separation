"""train_early_fusion.py
Fine-tune the unconditioned HTDemucs checkpoint with 4-channel early-fusion input.

The unconditioned checkpoint (`best_unconditioned.pt`) was trained with:
    audio_channels=2, extra_input_channels=0

This script:
1. Builds a new HTDemucs with audio_channels=2, extra_input_channels=2.
2. Inflates the first encoder weights to accept the two extra guide channels:
   - encoder[0].conv.weight  : (48, chin_z_old, 8, 1) → (48, chin_z_new, 8, 1)
   - tencoder[0].conv.weight : (48, chin_old,   8)    → (48, chin_new,   8)
   Old weights are copied to the first N channels; extra channels are initialised
   with small Gaussian noise (std = 0.01 × old_weight.std()).
3. All other weights are loaded directly from the checkpoint (shapes match).
4. Trains using EarlyFusionDataset, which returns (input_4ch, sources) pairs.

Usage
-----
    python train_early_fusion.py \
        --config configs/experiments/train_early_fusion.yaml \
        --checkpoint artifacts/checkpoints/train_guitarrecordings_unconditioned/best_unconditioned.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from _bootstrap import bootstrap

repo_root = bootstrap()

from data.dataset import EarlyFusionDataset
from data.manifests import load_manifest
from models.factory import build_model
from training.checkpointing import save_checkpoint, save_training_history
from training.engine import TrainConfig, train_one_epoch, evaluate_validation
from training.losses import build_pit_l1_loss
from utils.io import load_config
from utils.paths import ensure_dir


# ---------------------------------------------------------------------------
# Weight inflation helpers
# ---------------------------------------------------------------------------

def _inflate_conv_weight(
    old_weight: torch.Tensor,
    new_weight: torch.Tensor,
    noise_std_scale: float = 0.01,
) -> torch.Tensor:
    """Copy *old_weight* into the first channels of *new_weight* and fill the
    remaining channels with small Gaussian noise.

    Works for both 3-D (out, in, k) and 4-D (out, in, kH, kW) tensors.
    The channel dimension is always dim=1 (standard PyTorch Conv layout).
    """
    n_old = old_weight.shape[1]
    n_new = new_weight.shape[1]
    assert n_new >= n_old, "new_weight must have at least as many input channels as old_weight"

    noise_std = noise_std_scale * old_weight.std().item()
    with torch.no_grad():
        new_weight[:, :n_old, ...] = old_weight
        if n_new > n_old:
            new_weight[:, n_old:, ...].normal_(mean=0.0, std=max(noise_std, 1e-6))
    return new_weight


def inflate_checkpoint(
    model: torch.nn.Module,
    checkpoint_state: dict,
    noise_std_scale: float = 0.01,
) -> None:
    """Load *checkpoint_state* into *model*, inflating the first-encoder layers.

    All layers except encoder[0].conv.weight and tencoder[0].conv.weight are
    loaded with a strict shape match.  The two first-encoder layers are
    inflated to accommodate the extra guide channels.

    Prints a summary of which layers were inflated vs. copied directly.
    """
    model_state = model.state_dict()

    inflated_keys: list[str] = []
    direct_keys: list[str] = []
    skipped_keys: list[str] = []

    # Keys for the first-encoder convolution weights
    inflate_targets = {
        "encoder.0.conv.weight",
        "tencoder.0.conv.weight",
    }

    for key, ckpt_tensor in checkpoint_state.items():
        if key not in model_state:
            skipped_keys.append(key)
            continue

        model_tensor = model_state[key]

        if key in inflate_targets and ckpt_tensor.shape != model_tensor.shape:
            # The channel dimension (dim=1) may differ — inflate
            _inflate_conv_weight(ckpt_tensor, model_tensor, noise_std_scale)
            inflated_keys.append(
                f"  {key}: {list(ckpt_tensor.shape)} -> {list(model_tensor.shape)}"
            )
        else:
            if ckpt_tensor.shape != model_tensor.shape:
                skipped_keys.append(
                    f"{key}: checkpoint {list(ckpt_tensor.shape)} vs model {list(model_tensor.shape)} — SKIPPED"
                )
                continue
            model_state[key].copy_(ckpt_tensor)
            direct_keys.append(key)

    model.load_state_dict(model_state, strict=True)

    print(f"Inflated layers ({len(inflated_keys)}):")
    for msg in inflated_keys:
        print(msg)
    print(f"Loaded directly: {len(direct_keys)} layers")
    if skipped_keys:
        print(f"Skipped ({len(skipped_keys)}):")
        for msg in skipped_keys:
            print(f"  {msg}")


# ---------------------------------------------------------------------------
# Early-fusion training loop (adapted from engine.py)
# ---------------------------------------------------------------------------

def train_one_epoch_ef(model, dataloader, optimizer, loss_fn, device, use_sum_loss=False):
    """Training loop for EarlyFusionDataset.

    The dataset returns (input_4ch, sources) where:
        input_4ch : (B, 4, T) — 4-channel model input (already concatenated)
        sources   : (B, 2, 2, T) — guitar1 and guitar2 stereo targets
    """
    from tqdm.auto import tqdm
    from utils.core import center_trim

    model.train()
    alpha = 0.7
    beta = 0.3
    running_loss = 0.0
    steps = 0

    progress = tqdm(dataloader, desc="train", leave=False)
    for input_4ch, sources in progress:
        input_4ch = input_4ch.to(device)      # (B, 4, T)
        sources = sources.to(device)           # (B, 2, 2, T)

        optimizer.zero_grad()
        outputs = model(input_4ch)             # (B, 2*2, T) or (B, 2, 2, T)
        outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
        labels = center_trim(sources, outputs)
        loss = loss_fn(outputs, labels)

        if use_sum_loss:
            loss_sum = torch.nn.functional.l1_loss(
                outputs.sum(dim=1), labels.sum(dim=1)
            )
            loss = alpha * loss + beta * loss_sum

        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())
        steps += 1
        progress.set_postfix(loss=f"{running_loss / steps:.4f}")

    return running_loss / max(steps, 1)


def evaluate_validation_ef(model, dataloader, loss_fn, device):
    """Validation loop for EarlyFusionDataset."""
    from tqdm.auto import tqdm
    from utils.core import center_trim

    model.eval()
    running_loss = 0.0
    steps = 0

    with torch.no_grad():
        progress = tqdm(dataloader, desc="valid", leave=False)
        for input_4ch, sources in progress:
            input_4ch = input_4ch.to(device)
            sources = sources.to(device)
            outputs = model(input_4ch)
            outputs = outputs.view(outputs.size(0), 2, 2, outputs.size(-1))
            labels = center_trim(sources, outputs)
            loss = loss_fn(outputs, labels)
            running_loss += float(loss.item())
            steps += 1
            progress.set_postfix(loss=f"{running_loss / steps:.4f}")

    return running_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune HTDemucs with 4-channel early-fusion input."
    )
    parser.add_argument("--config", required=True, help="Path to train_early_fusion.yaml.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to unconditioned checkpoint to inflate (best_unconditioned.pt).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_kwargs = config["model"].get("kwargs", {})

    # ---- Load manifest and build datasets ------------------------------------
    manifest_entries = load_manifest(repo_root / config["dataset"]["manifest"])
    train_entries = [e for e in manifest_entries if e["split"] == config["dataset"]["train_split"]]
    valid_entries = [e for e in manifest_entries if e["split"] == config["dataset"]["valid_split"]]

    print("Early-Fusion Training")
    print(f"  config          : {args.config}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  run name        : {config['run']['name']}")
    print(f"  train tracks    : {len(train_entries)}")
    print(f"  val tracks      : {len(valid_entries)}")
    print(f"  segment seconds : {config['audio']['segment_seconds']}")
    print(f"  batch size      : {config['training']['batch_size']}")
    print(f"  epochs          : {config['training']['epochs']}")
    print(f"  learning rate   : {config['training'].get('learning_rate', 3e-4)}")
    print(f"  use sum loss    : {config['training'].get('use_sum_loss', False)}")

    segment_seconds = config["audio"]["segment_seconds"]
    stride_seconds = config["audio"].get("stride_seconds", segment_seconds)
    sample_rate = 44100
    stride_samples = int(stride_seconds * sample_rate)

    train_dataset = EarlyFusionDataset(
        train_entries,
        sample_length=segment_seconds,
        stride=stride_samples,
        normalize=config["dataset"]["normalize"],
    )
    valid_dataset = EarlyFusionDataset(
        valid_entries,
        sample_length=segment_seconds,
        stride=stride_samples,
        normalize=config["dataset"]["normalize"],
    )
    print(f"  train samples   : {len(train_dataset)}")
    print(f"  val samples     : {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    # ---- Build model and inflate from checkpoint ----------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nBuilding model with extra_input_channels={model_kwargs.get('extra_input_channels', 0)} ...")
    model = build_model(config["model"]["name"], model_kwargs)

    print(f"Loading and inflating checkpoint: {args.checkpoint}")
    payload = torch.load(args.checkpoint, map_location="cpu")
    if "model_state_dict" in payload:
        ckpt_state = payload["model_state_dict"]
    else:
        ckpt_state = payload
    inflate_checkpoint(model, ckpt_state)

    model.to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  device          : {device}")
    print(f"  parameters      : total={param_count:,}  trainable={trainable:,}")

    # ---- Training -----------------------------------------------------------
    checkpoints_dir = ensure_dir(repo_root / "artifacts" / "checkpoints" / config["run"]["name"])
    log_dir = ensure_dir(repo_root / "artifacts" / "logs" / config["run"]["name"])
    print(f"  checkpoints dir : {checkpoints_dir}")
    print(f"  log dir         : {log_dir}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"].get("learning_rate", 3e-4))
    loss_fn = build_pit_l1_loss()
    epochs = config["training"]["epochs"]
    use_sum_loss = config["training"].get("use_sum_loss", False)
    checkpoint_interval = config["training"].get("checkpoint_interval", 5)

    history: list[dict] = []
    best_valid = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss = train_one_epoch_ef(model, train_loader, optimizer, loss_fn, device, use_sum_loss)
        valid_loss = evaluate_validation_ef(model, valid_loader, loss_fn, device)

        metrics = {"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss}
        history.append(metrics)
        print(
            f"epoch={epoch + 1}  train_loss={train_loss:.4f}  "
            f"valid_loss={valid_loss:.4f}  best_valid={min(best_valid, valid_loss):.4f}"
        )

        if valid_loss < best_valid:
            best_valid = valid_loss
            save_checkpoint(checkpoints_dir / "best.pt", model, optimizer, epoch, metrics)
            print(f"  -> saved best checkpoint: {checkpoints_dir / 'best.pt'}")

        if epoch == 0 or (epoch + 1) % checkpoint_interval == 0:
            ckpt_path = checkpoints_dir / f"epoch_{epoch:03d}.pt"
            save_checkpoint(ckpt_path, model, optimizer, epoch, metrics)
            print(f"  -> saved periodic checkpoint: {ckpt_path}")

    save_training_history(log_dir, history)
    print(f"\nTraining complete. History: {len(history)} epochs.")


if __name__ == "__main__":
    main()
