from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap

repo_root = bootstrap()

from src.data.manifests import load_manifest
from src.inference.separate import separate_tracks
from src.models.factory import build_model
from src.utils.io import load_config, save_json


def load_checkpoint_into_model(model, checkpoint_path, device="cpu"):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload.get("model_state_dict", payload))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-track separation.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_entries = load_manifest(repo_root / config["dataset"]["manifest"], resolve_root=repo_root)
    test_entries = [e for e in manifest_entries if e["split"] == config["dataset"]["test_split"]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config["model"]["name"], config["model"].get("kwargs", {}))
    model = load_checkpoint_into_model(model, args.checkpoint, device=device)
    model.to(device)

    checkpoint_stem = Path(args.checkpoint).stem
    output_dir = repo_root / "outputs" / "predictions" / config["run"]["name"] / checkpoint_stem
    written = separate_tracks(model, test_entries, output_dir, device)
    save_json(output_dir / "predictions_manifest.json", written)
    print(f"Wrote predictions to {output_dir}")


if __name__ == "__main__":
    main()
