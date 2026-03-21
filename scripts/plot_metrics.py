from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap

repo_root = bootstrap()

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.plotting.metrics import plot_training_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved training history.")
    parser.add_argument("--history", required=True, help="Path to history.json")
    parser.add_argument("--output-dir", required=True, help="Directory for plots")
    args = parser.parse_args()

    with open(args.history, "r", encoding="utf-8") as handle:
        history = json.load(handle)
    plot_training_history(history, args.output_dir)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()

