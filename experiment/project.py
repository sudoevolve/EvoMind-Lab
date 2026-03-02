from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_project_config(project: str) -> dict[str, Any]:
    base = _resolve_project_dir(project)
    cfg_path = base / "project.json"
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("project.json must be an object")
    return data


def _resolve_project_dir(project: str) -> Path:
    p = Path(project)
    if p.exists():
        return p if p.is_dir() else p.parent
    root = Path(__file__).resolve().parent.parent
    return root / "projects" / project
