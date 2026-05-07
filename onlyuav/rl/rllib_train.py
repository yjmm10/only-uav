from __future__ import annotations

import importlib
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Type

import ray
from omegaconf import DictConfig, OmegaConf
from ray.tune.registry import register_env

from onlyuav.rl.common import append_jsonl, attach_train_file_logger, modules_dict_for_rllib, to_jsonable
from onlyuav.rl.hparams import load_algo_hparams

_RLLIB_CONFIG_MODULES: dict[str, tuple[str, str]] = {
    "ppo": ("ray.rllib.algorithms.ppo", "PPOConfig"),
    "sac": ("ray.rllib.algorithms.sac", "SACConfig"),
    "a2c": ("ray.rllib.algorithms.a2c", "A2CConfig"),
    "td3": ("ray.rllib.algorithms.td3", "TD3Config"),
    "ddpg": ("ray.rllib.algorithms.ddpg", "DDPGConfig"),
}


def _load_rllib_config_class(algo: str) -> Type[Any]:
    key = algo.lower()
    if key not in _RLLIB_CONFIG_MODULES:
        raise ValueError(f"rllib 不支持算法 {algo!r}")
    mod_name, cls_name = _RLLIB_CONFIG_MODULES[key]
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"当前 Ray 安装未提供 {mod_name}.{cls_name}（算法 {key!r}）。"
            "较新版本 RLlib 可能已移除 TD3/DDPG/A2C 等，请使用 sb3/tianshou 后端，"
            "或为 rllib 安装仍包含该 Config 的 Ray 版本。"
        ) from exc


def _register_onlyuav_env() -> None:
    from onlyuav.core.env_builder import EnvBuilder
    from onlyuav.models import load_default_components
    from onlyuav.rl.common import FixedSeedResetWrapper

    def _creator(env_config: dict):
        load_default_components()
        env = EnvBuilder.build(OmegaConf.create(env_config["modules"]))
        if env_config.get("fixed_reset_seed") is not None:
            env = FixedSeedResetWrapper(env, int(env_config["fixed_reset_seed"]))
        return env

    register_env("onlyuav_drone", _creator)


def train_rllib(
    cfg: DictConfig,
    *,
    algo_name: str,
    total_timesteps: int,
    fixed_reset_seed: int | None,
    resume_default: bool,
    model_path: Path,
    log_path: Path,
    metric_jsonl_path: Path,
    log_every_steps: int,
    include_step_reward: bool,
    include_timestamp: bool,
    train_log_path: Path | None = None,
    session_dir: Path | None = None,
) -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        import ray.rllib  # noqa: F401
    except ImportError as exc:
        raise SystemExit("请先安装 RLlib：`uv sync --extra rllib`") from exc

    key = algo_name.lower()
    try:
        ConfigClass = _load_rllib_config_class(key)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    h = dict(load_algo_hparams(cfg, algo_name))
    rllib_over = OmegaConf.to_container(cfg.experiment.train.get("rllib", {}), resolve=True) or {}
    num_env_runners = int(rllib_over.get("num_env_runners", 0))
    num_cpus = int(rllib_over.get("num_cpus", 1))
    use_new_api = bool(rllib_over.get("use_new_api_stack", False))

    run_logger = logging.getLogger(f"onlyuav.train.rllib.{algo_name.lower()}")
    run_logger.handlers.clear()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = True
    if train_log_path is not None:
        attach_train_file_logger(run_logger, train_log_path)
    if session_dir is not None:
        run_logger.info("session_dir=%s metric_jsonl=%s", session_dir, metric_jsonl_path)

    ray_init_here = False
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus, num_gpus=0, ignore_reinit_error=True)
        ray_init_here = True

    _register_onlyuav_env()
    env_cfg = {
        "modules": modules_dict_for_rllib(cfg),
        "fixed_reset_seed": fixed_reset_seed,
    }

    train_common: dict[str, Any] = {}
    for src, dst in (("lr", "lr"), ("gamma", "gamma"), ("train_batch_size", "train_batch_size")):
        if src in h:
            train_common[dst] = h[src]

    train_kwargs = dict(train_common)
    if key in ("sac", "td3", "ddpg") and "tau" in h:
        train_kwargs["tau"] = h["tau"]

    config = (
        ConfigClass()
        .api_stack(
            enable_rl_module_and_learner=use_new_api,
            enable_env_runner_and_connector_v2=use_new_api,
        )
        .environment(env="onlyuav_drone", env_config=env_cfg)
        .env_runners(num_env_runners=num_env_runners)
        .framework("torch")
        .training(**train_kwargs)
    )
    build = getattr(config, "build_algo", None) or getattr(config, "build", None)
    if build is None:
        if ray_init_here:
            ray.shutdown()
        raise SystemExit("RLlib Config 缺少 build_algo / build，请检查 Ray 版本。")
    algo = build()

    ckpt_dir = Path(model_path)
    if ckpt_dir.suffix.lower() == ".zip":
        ckpt_dir = ckpt_dir.with_suffix("")
    ckpt_dir = Path(str(ckpt_dir) + "_rllib_ckpt")
    if resume_default and ckpt_dir.exists() and ckpt_dir.is_dir() and any(ckpt_dir.iterdir()):
        run_logger.info("RESUME rllib algo=%s checkpoint_dir=%s", key, ckpt_dir)
        algo.restore(str(ckpt_dir))
    else:
        run_logger.info("FRESH_START rllib algo=%s total_timesteps=%s", key, total_timesteps)

    start_t = time.time()
    last_logged = -1
    trained_steps = 0
    try:
        while trained_steps < total_timesteps:
            result = algo.train()
            trained_steps = int(
                result.get("num_env_steps_sampled")
                or result.get("timesteps_total")
                or trained_steps
            )
            if trained_steps - last_logged >= log_every_steps or trained_steps >= total_timesteps:
                last_logged = trained_steps
                row: dict = {
                    "library": "rllib",
                    "algorithm": key,
                    "timesteps": trained_steps,
                    "elapsed_s": round(time.time() - start_t, 3),
                }
                if include_timestamp:
                    row["timestamp"] = datetime.utcnow().isoformat()
                for rk, rv in result.items():
                    if isinstance(rv, (int, float, str, bool, type(None))):
                        row[f"result/{rk}"] = to_jsonable(rv)
                if not include_step_reward:
                    row.pop("step_reward", None)
                append_jsonl(metric_jsonl_path, row)
    finally:
        save_dir = str(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        algo.save(save_dir)
        algo.stop()
        if ray_init_here:
            ray.shutdown()

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "library": "rllib",
                "algorithm": key,
                "checkpoint_dir": save_dir,
                "timesteps_target": total_timesteps,
                "timesteps_trained": trained_steps,
            },
            f,
            indent=2,
        )
    run_logger.info("end training checkpoint=%s", save_dir)
    print(f"[rllib/{key}] Checkpoint dir: {save_dir}")
    print(f"[rllib/{key}] Metrics: {metric_jsonl_path}")
