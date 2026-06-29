"""
Mutable configuration store with JSON persistence.

Allows runtime editing of pipeline settings through the web UI
without requiring environment variables or server restarts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

from .config import Config

logger = logging.getLogger(__name__)

_CONFIG_STORE: Optional["ConfigStore"] = None


class ConfigStore:
    """Wraps a Config instance with JSON persistence.

    On init, loads from ``data/config.json`` if it exists, otherwise
    creates a default Config from env vars and saves it.
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        self._path = self._config.data_dir / "config.json"

    @property
    def config(self) -> Config:
        return self._config

    def _serializable(self) -> dict[str, Any]:
        """Convert config to a JSON-serializable dict (Path → str)."""
        d = asdict(self._config)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def save(self) -> None:
        """Persist current config to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = self._serializable()
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Config saved to %s", self._path)

    def load(self) -> None:
        """Load config from disk, falling back to env vars."""
        if not self._path.exists():
            logger.info("No saved config at %s — using env var defaults", self._path)
            self.save()
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            # Apply saved values to the config object (only fields that exist)
            for key, value in data.items():
                if hasattr(self._config, key):
                    # Convert string paths back
                    if key.endswith("_dir") or key.endswith("_path"):
                        value = Path(value)
                    setattr(self._config, key, value)
            logger.info("Config loaded from %s", self._path)
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", self._path, exc)

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Apply partial updates and persist. Returns the full config dict."""
        for key, value in changes.items():
            if hasattr(self._config, key):
                # Type coercion for booleans and numbers
                current = getattr(self._config, key)
                if isinstance(current, bool):
                    value = str(value).lower() in ("true", "1", "yes", "on")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                setattr(self._config, key, value)
        self.save()
        return self._serializable()

    def as_dict(self) -> dict[str, Any]:
        return self._serializable()


def get_config_store(config: Optional[Config] = None) -> ConfigStore:
    """Return the module-level singleton ConfigStore."""
    global _CONFIG_STORE
    if _CONFIG_STORE is None:
        _CONFIG_STORE = ConfigStore(config)
        _CONFIG_STORE.load()
    return _CONFIG_STORE
