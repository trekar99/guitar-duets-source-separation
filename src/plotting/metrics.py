from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from guitarduets.utils.io import save_json
from guitarduets.utils.paths import ensure_dir


def plot_training_history(history: list[dict], output_dir: str | Path) -> None:
    output_path = ensure_dir(output_dir)
    epochs = [item["epoch"] for item in history]
    train_losses = [item["train_loss"] for item in history]
    valid_losses = [item["valid_loss"] for item in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, valid_losses, label="valid")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training history")
    plt.grid()
    plt.legend()
    plt.savefig(output_path / "training_loss.png")
    plt.close()
    save_json(output_path / "history_snapshot.json", history)
