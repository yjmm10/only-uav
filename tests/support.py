"""测试用辅助函数。"""

from __future__ import annotations

from pathlib import Path
from typing import List

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig


def compose_config(overrides: List[str] | None = None) -> DictConfig:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])
