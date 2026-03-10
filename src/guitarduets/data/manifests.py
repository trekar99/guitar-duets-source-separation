from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_manifest(path: str | Path) -> list[dict]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {manifest_path} must contain a JSON list.")
    return data


def save_manifest(entries: Iterable[dict], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(list(entries), handle, indent=2)

