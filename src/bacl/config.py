from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    return config


def resolve_path(value: str | Path, config: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_dir = Path(config["_config_path"]).parent
    return (config_dir / path).resolve()
