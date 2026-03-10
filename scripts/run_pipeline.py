from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap

repo_root = bootstrap()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run separation and evaluation for a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    subprocess.check_call([sys.executable, str(script_dir / "separate.py"), "--config", args.config, "--checkpoint", args.checkpoint])
    checkpoint_stem = Path(args.checkpoint).stem
    predictions_dir = repo_root / "artifacts" / "predictions"

    import yaml

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_name = config["run"]["name"]
    final_predictions_dir = predictions_dir / run_name / checkpoint_stem
    subprocess.check_call([sys.executable, str(script_dir / "evaluate.py"), "--config", args.config, "--predictions", str(final_predictions_dir)])


if __name__ == "__main__":
    main()

