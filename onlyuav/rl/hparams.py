from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from onlyuav.rl.builtin_defaults import get_builtin
from onlyuav.rl.common import CONFIG_DIR
from onlyuav.rl.registry import normalize_backend

# 可选：按库从 YAML 叠加（文件不存在则仅用内置 + 覆盖项）
BACKEND_PARAM_DIR: dict[str, str] = {
    "sb3": "algorithm/sb3",
    "rllib": "algorithm/rlib",
    "tianshou": "algorithm/tianshou",
    "di_engine": "algorithm/ding",
}


def resolve_training_algorithm_name(cfg: DictConfig) -> str:
    """当前训练使用的算法名（单算法模式）。优先级: train.algorithm > train.algo > 默认 ppo。"""
    a = OmegaConf.select(cfg, "experiment.train.algorithm")
    if a is not None and str(a).strip() and not str(a).startswith("${"):
        return str(a).lower()
    legacy = OmegaConf.select(cfg, "experiment.train.algo")
    if legacy is not None and str(legacy).strip() and not str(legacy).startswith("${"):
        return str(legacy).lower()
    return "ppo"


def load_algo_hparams(cfg: DictConfig, algo_name: str) -> dict[str, Any]:
    backend = normalize_backend(str(cfg.experiment.train.get("backend", "sb3")))
    if backend not in BACKEND_PARAM_DIR:
        raise ValueError(f"未知 backend={backend!r}，可选: {sorted(BACKEND_PARAM_DIR)}")

    key = algo_name.lower()
    merged: dict[str, Any] = deepcopy(get_builtin(backend, key))

    shared = OmegaConf.to_container(cfg.experiment.train.get("shared_algo_kwargs", {}), resolve=True) or {}
    merged.update(shared)

    folder = BACKEND_PARAM_DIR[backend]
    path = CONFIG_DIR / folder / f"{key}.yaml"
    if path.exists():
        file_cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True) or {}
        merged.update(file_cfg)

    selected = resolve_training_algorithm_name(cfg)
    if backend == "sb3" and key == selected:
        hydra_sb3 = OmegaConf.select(cfg, "algorithm.sb3")
        if hydra_sb3 is not None:
            merged.update(OmegaConf.to_container(hydra_sb3, resolve=True) or {})

    override = OmegaConf.to_container(cfg.experiment.train.get("algo_kwargs", {}), resolve=True) or {}
    merged.update(override)
    return merged
