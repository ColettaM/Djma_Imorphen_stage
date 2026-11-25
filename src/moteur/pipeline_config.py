"""Configuration partagee du pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_FILE = PROJECT_ROOT / "config" / "pipeline.json"
REQUIRED_KEYS = {"expected_images", "seed", "top_k", "max_bulletin_age_days"}


def load_pipeline_config(path: Path = DEFAULT_CONFIG_FILE) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(config)
    if missing:
        raise ValueError(f"Configuration du pipeline incomplete: {sorted(missing)}")
    for key in REQUIRED_KEYS:
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(
                f"Configuration du pipeline invalide pour {key}: {config[key]!r}"
            )
    return config


PIPELINE_CONFIG = load_pipeline_config()
EXPECTED_IMAGES = PIPELINE_CONFIG["expected_images"]
RANDOM_SEED = PIPELINE_CONFIG["seed"]
TOP_K = PIPELINE_CONFIG["top_k"]
MAX_BULLETIN_AGE_DAYS = PIPELINE_CONFIG["max_bulletin_age_days"]
