from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from _bootstrap import bootstrap

repo_root = bootstrap()

from src.data.dataset import GuitarDataset
from src.data.manifests import load_manifest
from src.models.factory import build_model
from src.training.engine import TrainConfig, train_model
from src.utils.io import load_config


def describe_dataset(label: str, entries: list[dict]) -> None:
    total_seconds = sum(entry["length"] / entry["samplerate"] for entry in entries)
    notes_available = sum(1 for entry in entries if entry.get("notes_csv"))
    notes_suffix = f", notes_csv on {notes_available}/{len(entries)} tracks" if entries else ""
    print(f"{label}: {len(entries)} tracks, {total_seconds / 60:.2f} minutes{notes_suffix}")


def conditioning_label(model_kwargs: dict) -> str:
    time_cond = model_kwargs.get("time_conditioning", model_kwargs.get("note_conditioning", False))
    freq_cond = model_kwargs.get("freq_conditioning", False)
    if time_cond and freq_cond:
        return "time+freq"
    if time_cond:
        return "time"
    if freq_cond:
        return "freq"
    return "none"


def uses_notes(model_kwargs: dict) -> bool:
    time_cond = model_kwargs.get("time_conditioning", model_kwargs.get("note_conditioning", False))
    freq_cond = model_kwargs.get("freq_conditioning", False)
    return time_cond or freq_cond


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a source separation model.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to a .pt checkpoint to warm-start from.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    model_kwargs = config["model"].get("kwargs", {})
    manifest_entries = load_manifest(repo_root / config["dataset"]["manifest"], resolve_root=repo_root)
    train_entries = [e for e in manifest_entries if e["split"] == config["dataset"]["train_split"]]
    valid_entries = [e for e in manifest_entries if e["split"] == config["dataset"]["valid_split"]]

    run_name = config["run"]["name"]
    note_mode = uses_notes(model_kwargs)

    print("Training configuration")
    print(f"  config:           {args.config}")
    print(f"  run name:         {run_name}")
    print(f"  model:            {config['model']['name']}")
    print(f"  segment:          {config['audio']['segment_seconds']}s")
    print(f"  batch size:       {config['training']['batch_size']}")
    print(f"  epochs:           {config['training']['epochs']}")
    print(f"  learning rate:    {config['training'].get('learning_rate', 3e-4)}")
    print(f"  conditioning:     {conditioning_label(model_kwargs)}")
    print(f"  seed:             {args.seed}")
    describe_dataset("  train", train_entries)
    if valid_entries:
        describe_dataset("  valid", valid_entries)
    else:
        print("  valid: random 80/20 split from train")

    if not valid_entries:
        dataset = GuitarDataset(
            train_entries, sample_length=config["audio"]["segment_seconds"],
            normalize=config["dataset"]["normalize"], use_notes=note_mode,
        )
        train_size = int(len(dataset) * 0.8)
        valid_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, valid_dataset = random_split(
            dataset, [train_size, valid_size], generator=generator)
    else:
        train_dataset = GuitarDataset(
            train_entries, sample_length=config["audio"]["segment_seconds"],
            normalize=config["dataset"]["normalize"], use_notes=note_mode,
        )
        valid_dataset = GuitarDataset(
            valid_entries, sample_length=config["audio"]["segment_seconds"],
            normalize=config["dataset"]["normalize"], use_notes=note_mode,
        )

    print(f"  samples: train={len(train_dataset)} valid={len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config["training"]["batch_size"], shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = build_model(config["model"]["name"], model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            print(f"WARNING: checkpoint not found at {ckpt_path}, training from scratch.")
        else:
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(state.get("model_state_dict", state))
            print(f"  loaded checkpoint: {ckpt_path}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  device: {device}  params: {total_params:,} ({trainable:,} trainable)")

    checkpoints_dir = repo_root / "outputs" / "checkpoints" / run_name
    log_dir = repo_root / "outputs" / "logs" / run_name

    train_config = TrainConfig(
        epochs=config["training"]["epochs"],
        learning_rate=config["training"].get("learning_rate", 3e-4),
        use_sum_loss=config["training"].get("use_sum_loss", False),
        checkpoint_interval=config["training"].get("checkpoint_interval", 5),
        use_notes=note_mode,
        use_amp=config["training"].get("use_amp", True),
        patience=config["training"].get("patience", 0),
        gradient_clip=config["training"].get("gradient_clip", 5.0),
    )
    print(f"  AMP={train_config.use_amp}  grad_clip={train_config.gradient_clip}  "
          f"patience={train_config.patience if train_config.patience > 0 else 'off'}")
    print()

    history = train_model(
        model, train_loader, valid_loader,
        checkpoints_dir=checkpoints_dir, log_dir=log_dir,
        config=train_config, device=device,
    )
    print(f"Training finished. {len(history)} epochs completed.")


if __name__ == "__main__":
    main()
