from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf

from onlyuav.core.env_builder import EnvBuilder
from onlyuav.models import load_default_components

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class FixedSeedResetWrapper(gym.Wrapper):
    """每次 reset 使用固定 seed，便于复现任务流。"""

    def __init__(self, env: gym.Env, fixed_seed: int):
        super().__init__(env)
        self.fixed_seed = int(fixed_seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.fixed_seed
        return self.env.reset(**kwargs)


def to_jsonable(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_env(cfg: DictConfig, fixed_reset_seed: int | None) -> gym.Env:
    load_default_components()
    env = EnvBuilder.build(cfg.modules)
    if fixed_reset_seed is not None:
        env = FixedSeedResetWrapper(env, int(fixed_reset_seed))
    return env


def modules_dict_for_rllib(cfg: DictConfig) -> dict[str, Any]:
    return OmegaConf.to_container(cfg.modules, resolve=True)  # type: ignore[return-value]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def attach_train_file_logger(logger: logging.Logger, path: Path) -> logging.FileHandler:
    """将训练日志写入单一文件（与 SB3 控制台输出无关）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return fh
