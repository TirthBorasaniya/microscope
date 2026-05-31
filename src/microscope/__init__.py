"""microscope — TopK SAE training, RelP circuit discovery, and a random-weights control.

The Gemma-2-2B architecture constants and a single YAML config loader live here so that
every downstream module imports its hyperparameters from one place rather than embedding
numeric literals in function bodies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# --- Gemma-2-2B architecture constants -------------------------------------------------
# Defined once, module-level, and imported everywhere these values are needed.
GEMMA2_HIDDEN_SIZE: int = 2304
GEMMA2_NUM_LAYERS: int = 26
SAE_HOOK_LAYER: int = 18  # residual stream after layer 18
SAE_HOOK_NAME: str = "blocks.18.hook_resid_post"
SAE_EXPANSION: int = 8  # SAE width = 8 x 2304 = 18432 features
SAE_TOP_K: int = 64  # features active per token
SAE_TRAIN_TOKENS: int = 2_000_000_000

__all__ = [
    "GEMMA2_HIDDEN_SIZE",
    "GEMMA2_NUM_LAYERS",
    "SAE_HOOK_LAYER",
    "SAE_HOOK_NAME",
    "SAE_EXPANSION",
    "SAE_TOP_K",
    "SAE_TRAIN_TOKENS",
    "load_yaml_config",
]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a plain dict.

    Every hyperparameter is read from ``configs/*.yaml`` through this single entry point so
    that the module CLIs never embed numeric literals of their own.
    """
    with open(path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping: got {type(config)!r}")
    return config
