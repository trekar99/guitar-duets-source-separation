from __future__ import annotations

import argparse

from _bootstrap import bootstrap

repo_root = bootstrap()

from src.data.metadata import build_manifest_from_split_roots
from src.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata and manifest files.")
    parser.add_argument("--config", required=True, help="Path to dataset config YAML.")
    args = parser.parse_args()

    config = load_config(args.config)
    split_roots = config["splits"]
    output_path = repo_root / config["manifest_output"]
    build_manifest_from_split_roots(split_roots, output_path)
    print(f"Wrote manifest to {output_path}")


if __name__ == "__main__":
    main()

