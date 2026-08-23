"""Read/write the processed-dataset manifest (JSON, human-readable, Chinese-safe)."""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(path: Path | str, manifest: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def read_manifest(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
