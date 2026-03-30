"""Bootstrap helpers for direct script execution."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root_on_path() -> None:
    """Ensure repository root is importable when running scripts directly."""
    project_root = Path(__file__).resolve().parents[2]
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
