from __future__ import annotations

from pathlib import Path


def resolve_path(path_str: str | Path) -> Path:
    return Path(path_str).expanduser().resolve()
