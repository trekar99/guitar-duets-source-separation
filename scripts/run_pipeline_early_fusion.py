"""run_pipeline_early_fusion.py
Pipeline that runs separate_early_fusion.py followed by evaluate.py.

Usage
-----
    python run_pipeline_early_fusion.py \
        --config configs/experiments/eval_early_fusion.yaml \
        --checkpoint artifacts/checkpoints/early_fusion/best.pt
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from _bootstrap import bootstrap

repo_root = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run early-fusion separation then evaluation for a checkpoint."
    )
    parser.add_argument("--config", required=True, help="Path to eval_early_fusion.yaml.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    # ---- Step 1: Separation --------------------------------------------------
    print("=" * 60)
    print("Step 1: Separation (separate_early_fusion.py)")
    print("=" * 60)
    subprocess.check_call(
        [
            sys.executable,
            str(script_dir / "separate_early_fusion.py"),
            "--config", args.config,
            "--checkpoint", args.checkpoint,
        ]
    )

    # ---- Step 2: Evaluation --------------------------------------------------
    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    run_name = config["run"]["name"]
    checkpoint_stem = Path(args.checkpoint).stem
    predictions_dir = repo_root / "artifacts" / "predictions" / run_name / checkpoint_stem

    print("\n" + "=" * 60)
    print("Step 2: Evaluation (evaluate.py)")
    print("=" * 60)
    subprocess.check_call(
        [
            sys.executable,
            str(script_dir / "evaluate.py"),
            "--config", args.config,
            "--predictions", str(predictions_dir),
        ]
    )

    print("\nPipeline complete.")
    metrics_dir = repo_root / "artifacts" / "metrics" / run_name / checkpoint_stem
    print(f"Metrics written to: {metrics_dir}")


if __name__ == "__main__":
    main()
