from __future__ import annotations

from pathlib import Path

import torch

from utils.io import save_json
from utils.paths import ensure_dir


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: dict) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        checkpoint_path,
    )


def save_training_history(run_dir: str | Path, history: list[dict]) -> None:
    run_path = ensure_dir(run_dir)
    save_json(run_path / "history.json", history)

