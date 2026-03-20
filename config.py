"""Shared configuration helpers for reproducible local execution."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
DEFAULT_OMS_DOCS_DIR = DEFAULT_DATA_DIR / "oms_docs"
DEFAULT_FINETUNE_OUTPUT_DIR = DEFAULT_MODELS_DIR / "finetunes"


def init_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT_DIR / ".env")


def path_from_env(env_name: str, fallback: Path) -> Path:
    raw = os.getenv(env_name)
    if raw:
        return Path(raw).expanduser().resolve()
    return fallback


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
