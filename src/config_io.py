"""
One loader for every YAML config read in the repo, with explicit encoding.

Why this exists
---------------
Text-mode ``open()`` without ``encoding=`` uses the platform codec — cp1252
on Windows. configs/crowd_flow.yaml is UTF-8 with non-ASCII characters
(arrows, dashes, superscripts), so each reader rolled its own dice: some
machines decoded fine, others died with "'charmap' codec can't decode byte
0x90" at job start. Centralising the read means the encoding is right by
construction and a future call site cannot forget it.

Callers keep their own policy for empty files / missing paths — this returns
whatever ``yaml.safe_load`` returns (``None`` for an empty document) and
raises ``FileNotFoundError`` naturally, so behaviour at each site is
unchanged apart from the codec fix.
"""

from __future__ import annotations

from typing import Any

import yaml


def load_yaml(path: str) -> Any:
    """Read and parse a YAML file as UTF-8."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_dict(path: str) -> dict:
    """``load_yaml`` for mapping documents: an empty file parses as {}. """
    result = load_yaml(path)
    return result if isinstance(result, dict) else {}
