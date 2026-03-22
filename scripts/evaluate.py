from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap

repo_root = bootstrap()

from data.manifests import load_manifest
from evaluation.metrics import evaluate_predictions
from utils.io import load_config, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate separated predictions.")
    parser.add_argument("--config", required=True, help="Path to experiment config.")
    parser.add_argument("--predictions", required=True, help="Predictions directory.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_entries = load_manifest(repo_root / config["dataset"]["manifest"])
    eval_entries = [
        entry
        for entry in manifest_entries
        if entry["split"] == config["dataset"]["test_split"]
    ]

    results, summary = evaluate_predictions(args.predictions, eval_entries)

    checkpoint_name = Path(args.predictions).name
    metrics_dir = repo_root / "artifacts" / "metrics" / config["run"]["name"] / checkpoint_name
    metrics_dir.mkdir(parents=True, exist_ok=True)

    save_json(metrics_dir / "per_track_metrics.json", results)
    save_json(metrics_dir / "summary.json", summary)
    print(f"Wrote metrics to {metrics_dir}")


if __name__ == "__main__":
    main()