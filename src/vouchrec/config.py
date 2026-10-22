"""Configuration loading and data-path resolution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
DATA_ROOT_ENV = "VOUCHREC_DATA_ROOT"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ConfigError(f"Config file must contain a mapping at top level: {path}")
    return content


def load_config(name: str, config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    """Load ``configs/<name>.yaml``."""
    return load_yaml(config_dir / f"{name}.yaml")


def read_dotenv(path: Path) -> dict[str, str]:
    """Parse a minimal KEY=VALUE .env file. Lines starting with '#' are ignored."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


@dataclass(frozen=True)
class DataPaths:
    root: Path
    reports: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def splits(self) -> Path:
        return self.root / "splits"

    @property
    def attack_simulations(self) -> Path:
        return self.root / "attack_simulations"


def resolve_data_paths(
    data_config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
    repo_root: Path = REPO_ROOT,
) -> DataPaths:
    """Resolve the data root (env var > .env file > config) and the reports directory."""
    environ = os.environ if environ is None else environ
    paths_cfg = data_config.get("paths", {})

    root_value = environ.get(DATA_ROOT_ENV) or read_dotenv(repo_root / ".env").get(DATA_ROOT_ENV)
    if not root_value:
        root_value = paths_cfg.get("data_root", "data")
    root = Path(root_value)
    if not root.is_absolute():
        root = repo_root / root

    reports = Path(paths_cfg.get("reports_dir", "data/reports"))
    if not reports.is_absolute():
        reports = repo_root / reports
    return DataPaths(root=root, reports=reports)
