from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        elif config_path.suffix == ".json":
            data = json.load(handle)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    return data


def save_json(path: str | Path, data: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)

