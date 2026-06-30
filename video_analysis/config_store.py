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

    # Fields derived from data_dir in __post_init__ — never persist these,
    # they are always recomputed from data_dir at startup.
    _DERIVED_PATH_FIELDS = frozenset(
        {
            "video_dir",
            "frames_dir",
            "audio_dir",
            "thumbnails_dir",
            "chroma_path",
            "clip_export_dir",
        }
    )

    def _serializable(self) -> dict[str, Any]:
        """Convert config to a JSON-serializable dict, skipping derived Path fields."""
        d = asdict(self._config)
        # Remove derived paths so they aren't persisted — they get
        # recomputed from data_dir on next startup.
        for k in self._DERIVED_PATH_FIELDS:
            d.pop(k, None)
        # Only persist user-settable Path fields (like data_dir itself)
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
        """Load config from disk, merging saved values over code defaults."""
        if not self._path.exists():
            logger.info("No saved config at %s — using code defaults", self._path)
            self._post_init_and_save()
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            saved_keys = set()
            for key, value in data.items():
                if key in self._DERIVED_PATH_FIELDS:
                    continue  # skip derived paths, recomputed from data_dir
                if hasattr(self._config, key):
                    if (key.endswith("_dir") or key.endswith("_path")) and not key.startswith("_"):
                        value = Path(value)
                    setattr(self._config, key, value)
                    saved_keys.add(key)
            # Recompute derived paths from data_dir
            self._post_init_and_save()
            current_keys = {k for k in self._serializable() if not k.startswith("_")}
            new_keys = current_keys - saved_keys
            if new_keys:
                logger.info(
                    "Config upgraded — %d new field(s): %s",
                    len(new_keys),
                    ", ".join(sorted(new_keys)),
                )
            logger.info(
                "Config loaded from %s (%d saved keys, %d total fields)",
                self._path,
                len(saved_keys),
                len(current_keys),
            )
        except Exception as exc:
            logger.warning("Failed to load config from %s: %s", self._path, exc)

    def _post_init_and_save(self) -> None:
        """Re-run __post_init__ to recompute derived paths, then save."""
        self._config.__post_init__()
        self.save()

    def update(self, changes: dict[str, Any]) -> dict[str, Any]:
        """Apply partial updates and persist. Returns the full config dict."""
        for key, value in changes.items():
            if hasattr(self._config, key):
                current = getattr(self._config, key)
                if isinstance(current, bool):
                    value = str(value).lower() in ("true", "1", "yes", "on")
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                setattr(self._config, key, value)
        # Recompute derived paths if data_dir changed
        if "data_dir" in changes:
            self._config.data_dir = Path(self._config.data_dir)
            self._config.__post_init__()
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
