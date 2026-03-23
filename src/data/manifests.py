from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_manifest(path: str | Path, resolve_root: str | Path | None = None) -> list[dict]:
    """Load a JSON manifest.

    Args:
        path: Path to the manifest file.
        resolve_root: If provided, relative file paths in the manifest entries
            (root, mix, sources/*, notes_csv) are resolved against this directory.
    """
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Manifest at {manifest_path} must contain a JSON list.")
    if resolve_root is not None:
        data = [_resolve_entry_paths(e, Path(resolve_root)) for e in data]
    return data


def _resolve_entry_paths(entry: dict, root: Path) -> dict:
    """Resolve relative paths in a manifest entry against *root*."""
    out = dict(entry)
    for key in ("root", "mix", "notes_csv"):
        val = out.get(key)
        if val and isinstance(val, str) and not Path(val).is_absolute():
            out[key] = str(root / val)
    if "sources" in out and isinstance(out["sources"], dict):
        out["sources"] = {
            k: str(root / v) if isinstance(v, str) and not Path(v).is_absolute() else v
            for k, v in out["sources"].items()
        }
    return out


def save_manifest(entries: Iterable[dict], path: str | Path) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(list(entries), handle, indent=2)

