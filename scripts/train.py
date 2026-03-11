from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from _bootstrap import bootstrap

repo_root = bootstrap()

from guitarduets.data.dataset import GuitarDataset
from guitarduets.data.manifests import load_manifest
from guitarduets.models.factory import build_model
from guitarduets.training.engine import TrainConfig, train_model
from guitarduets.utils.io import load_config


def describe_dataset(label: str, entries: list[dict]) -> None:
    total_seconds = sum(entry["length"] / entry["samplerate"] for entry in entries)
    notes_available = sum(1 for entry in entries if entry.get("notes_csv"))
    notes_suffix = f", notes_csv on {notes_available}/{len(entries)} tracks" if entries else ""
    print(f"{label}: {len(entries)} tracks, {total_seconds / 60:.2f} minutes{notes_suffix}")


def conditioning_label(model_kwargs: dict) -> str:
    time_conditioning = model_kwargs.get("time_conditioning", model_kwargs.get("note_conditioning", False))
    freq_conditioning = model_kwargs.get("freq_conditioning", False)
    if time_conditioning and freq_conditioning:
        return "time+freq"
    if time_conditioning:
        return "time"
    if freq_conditioning:
        return "freq"
    return "none"


def uses_notes(model_kwargs: dict) -> bool:
    time_conditioning = model_kwargs.get("time_conditioning", model_kwargs.get("note_conditioning", False))
    freq_conditioning = model_kwargs.get("freq_conditioning", False)
    return time_conditioning or freq_conditioning


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a source separation model.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    args = parser.parse_args()

    config = load_config(args.config)
    model_kwargs = config["model"].get("kwargs", {})
    manifest_entries = load_manifest(repo_root / config["dataset"]["manifest"])
    train_entries = [entry for entry in manifest_entries if entry["split"] == config["dataset"]["train_split"]]
    valid_entries = [entry for entry in manifest_entries if entry["split"] == config["dataset"]["valid_split"]]

    print("Training configuration")
    print(f"config: {args.config}")
    print(f"run name: {config['run']['name']}")
    print(f"model: {config['model']['name']}")
    print(f"segment seconds: {config['audio']['segment_seconds']}")
    print(f"batch size: {config['training']['batch_size']}")
    print(f"epochs: {config['training']['epochs']}")
    print(f"learning rate: {config['training'].get('learning_rate', 3e-4)}")
    print(f"use sum loss: {config['training'].get('use_sum_loss', False)}")
    print(f"conditioning: {conditioning_label(config['model'].get('kwargs', {}))}")
    print(f"normalize: {config['dataset']['normalize']}")
    print(f"manifest entries: {len(manifest_entries)}")
    describe_dataset("train split", train_entries)
    if valid_entries:
        describe_dataset("valid split", valid_entries)
    else:
        print("valid split: not provided, using random split from train split")

    if not valid_entries:
        dataset = GuitarDataset(
            train_entries,
            sample_length=config["audio"]["segment_seconds"],
            normalize=config["dataset"]["normalize"],
            use_notes=uses_notes(model_kwargs),
        )
        train_size = int(len(dataset) * 0.8)
        valid_size = len(dataset) - train_size
        train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
        print(f"random split samples: train={len(train_dataset)} valid={len(valid_dataset)}")
    else:
        train_dataset = GuitarDataset(
            train_entries,
            sample_length=config["audio"]["segment_seconds"],
            normalize=config["dataset"]["normalize"],
            use_notes=uses_notes(model_kwargs),
        )
        valid_dataset = GuitarDataset(
            valid_entries,
            sample_length=config["audio"]["segment_seconds"],
            normalize=config["dataset"]["normalize"],
            use_notes=uses_notes(model_kwargs),
        )
        print(f"dataset samples: train={len(train_dataset)} valid={len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    print(f"dataloader batches: train={len(train_loader)} valid={len(valid_loader)}")

    model = build_model(config["model"]["name"], model_kwargs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"device: {device}")
    print(f"model parameters: total={parameter_count:,} trainable={trainable_parameter_count:,}")

    checkpoints_dir = repo_root / "artifacts" / "checkpoints" / config["run"]["name"]
    log_dir = repo_root / "artifacts" / "logs" / config["run"]["name"]
    print(f"checkpoints dir: {checkpoints_dir}")
    print(f"log dir: {log_dir}")
    train_config = TrainConfig(
        epochs=config["training"]["epochs"],
        learning_rate=config["training"].get("learning_rate", 3e-4),
        use_sum_loss=config["training"].get("use_sum_loss", False),
        checkpoint_interval=config["training"].get("checkpoint_interval", 5),
        use_notes=uses_notes(model_kwargs),
    )
    history = train_model(
        model,
        train_loader,
        valid_loader,
        checkpoints_dir=checkpoints_dir,
        log_dir=log_dir,
        config=train_config,
        device=device,
    )
    print(f"Training finished. History entries: {len(history)}")


if __name__ == "__main__":
    main()
