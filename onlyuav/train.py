from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import hydra
import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from onlyuav.algorithms import get_algo_class
from onlyuav.core.env_builder import EnvBuilder
from onlyuav.models import load_default_components

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


class _TrainingLogger:
    def __init__(self):
        self.episode_rewards = []
        self.timesteps = []
        self._acc = 0.0

    def on_step(self, reward, done, t):
        self._acc += float(reward)
        if done:
            self.episode_rewards.append(self._acc)
            self.timesteps.append(int(t))
            self._acc = 0.0


class _FixedSeedResetWrapper(gym.Wrapper):
    """强制每次 reset 使用同一个 seed，复用同一任务流。"""

    def __init__(self, env, fixed_seed: int):
        super().__init__(env)
        self.fixed_seed = int(fixed_seed)

    def reset(self, **kwargs):
        kwargs["seed"] = self.fixed_seed
        return self.env.reset(**kwargs)

def _resolve_output_path(raw: str, algo: str, multi_mode: bool) -> Path:
    if "{algo}" in raw:
        return Path(raw.format(algo=algo))
    if multi_mode:
        base = Path(raw)
        return base.with_name(f"{base.name}_{algo}")
    return Path(raw)


def _find_checkpoint(model_path: Path) -> Path | None:
    candidates = [model_path]
    if model_path.suffix != ".zip":
        candidates.append(model_path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _to_jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (dict, list, tuple, str, int, float, bool)) or value is None:
        return value
    return str(value)


def _algo_hparams(cfg: DictConfig, algo_name: str) -> dict:
    selected_algo = str(cfg.experiment.train.algo).lower()
    if algo_name == selected_algo:
        by_algo = OmegaConf.to_container(cfg.algorithm, resolve=True) or {}
    else:
        cfg_path = Path(CONFIG_DIR) / "algorithm" / f"{algo_name}.yaml"
        if not cfg_path.exists():
            raise ValueError(f"Algorithm config not found: {cfg_path}")
        by_algo = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) or {}
    # 允许用户用 experiment.train.algo_kwargs 做最终覆盖
    override = OmegaConf.to_container(cfg.experiment.train.get("algo_kwargs", {}), resolve=True) or {}
    return {**by_algo, **override}


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_default_components()
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise SystemExit("请先安装依赖：`uv sync`") from exc

    class TrainingCallback(BaseCallback):
        def __init__(self, logger_obj, metric_jsonl_path: Path, run_logger: logging.Logger, verbose=0):
            super().__init__(verbose)
            self.logger_obj = logger_obj
            self.metric_jsonl_path = metric_jsonl_path
            self.run_logger = run_logger

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", 0.0)
            dones = self.locals.get("dones", False)
            reward = rewards[0] if hasattr(rewards, "__len__") else rewards
            done = dones[0] if hasattr(dones, "__len__") else dones
            self.logger_obj.on_step(reward, done, self.num_timesteps)
            return True

        def _on_rollout_end(self) -> None:
            metrics = {"timestamp": datetime.utcnow().isoformat(), "timesteps": int(self.num_timesteps)}
            logger_values = getattr(self.model.logger, "name_to_value", {})
            for key in (
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "train/approx_kl",
                "train/loss",
                "train/value_loss",
                "train/policy_gradient_loss",
                "train/explained_variance",
            ):
                if key in logger_values:
                    value = logger_values[key]
                    metrics[key] = _to_jsonable(value)
            with self.metric_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            # self.run_logger.info("rollout_end metrics=%s", metrics)

    algos_cfg = cfg.experiment.train.get("algos")
    if algos_cfg:
        algo_list = [str(a).lower() for a in algos_cfg]
    else:
        algo_list = [str(cfg.experiment.train.algo).lower()]

    multi_mode = len(algo_list) > 1
    total_timesteps = int(cfg.experiment.train.total_timesteps)
    fixed_reset_seed = cfg.experiment.train.get("fixed_reset_seed")
    resume_default = bool(cfg.experiment.train.get("resume", True))
    parallel_multi_algo = bool(cfg.experiment.train.get("parallel_multi_algo", True))

    def train_one(algo: str, force_multi_mode: bool = False):
        algo_class, algo_name = get_algo_class(algo)
        algo_kwargs = _algo_hparams(cfg, algo_name)
        env = EnvBuilder.build(cfg.modules)
        if fixed_reset_seed is not None:
            env = _FixedSeedResetWrapper(env, int(fixed_reset_seed))
        logger = _TrainingLogger()
        current_multi_mode = force_multi_mode or multi_mode
        model_path = _resolve_output_path(str(cfg.experiment.train.model_path), algo_name, current_multi_mode)
        log_path = _resolve_output_path(str(cfg.experiment.train.log_json), algo_name, current_multi_mode)
        exec_log_path = _resolve_output_path(str(cfg.experiment.train.exec_log), algo_name, current_multi_mode)
        metric_jsonl_path = _resolve_output_path(str(cfg.experiment.train.metric_jsonl), algo_name, current_multi_mode)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        exec_log_path.parent.mkdir(parents=True, exist_ok=True)
        metric_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        run_logger = logging.getLogger(f"onlyuav.train.{algo_name}")
        run_logger.handlers.clear()
        run_logger.setLevel(logging.INFO)
        run_logger.addHandler(logging.FileHandler(exec_log_path, mode="w", encoding="utf-8"))
        checkpoint = _find_checkpoint(model_path) if resume_default else None
        if checkpoint is not None:
            run_logger.info(
                "RESUME training algo=%s checkpoint=%s add_timesteps=%s fixed_reset_seed=%s",
                algo_name,
                checkpoint,
                total_timesteps,
                fixed_reset_seed,
            )
            model = algo_class.load(str(checkpoint), env=env)
            reset_num_timesteps = False
        else:
            run_logger.info(
                "FRESH_START training algo=%s total_timesteps=%s fixed_reset_seed=%s",
                algo_name,
                total_timesteps,
                fixed_reset_seed,
            )
            model = algo_class("MlpPolicy", env, verbose=1, seed=int(cfg.seed), **algo_kwargs)
            reset_num_timesteps = True

        model.learn(
            total_timesteps=total_timesteps,
            callback=TrainingCallback(logger, metric_jsonl_path, run_logger),
            reset_num_timesteps=reset_num_timesteps,
        )
        model.save(str(model_path))

        with log_path.open("w", encoding="utf-8") as f:
            json.dump({"algorithm": algo_name, "timesteps": logger.timesteps, "episode_rewards": logger.episode_rewards}, f, indent=2)

        run_logger.info("end training model=%s log_json=%s", model_path, log_path)
        print(f"[{algo_name}] Model saved: {model_path}.zip")
        print(f"[{algo_name}] Training log: {log_path}")
        print(f"[{algo_name}] Execution log: {exec_log_path}")
        print(f"[{algo_name}] Iteration metrics: {metric_jsonl_path}")

    if multi_mode and parallel_multi_algo:
        choices = HydraConfig.get().runtime.choices
        base_cmd = [
            sys.executable,
            "-m",
            "onlyuav.train",
            f"modules={choices['modules']}",
            f"experiment={choices['experiment']}",
            "experiment.train.algos=null",
            "experiment.train.parallel_multi_algo=false",
            f"experiment.train.total_timesteps={total_timesteps}",
            f"experiment.train.resume={'true' if resume_default else 'false'}",
        ]
        if fixed_reset_seed is None:
            base_cmd.append("experiment.train.fixed_reset_seed=null")
        else:
            base_cmd.append(f"experiment.train.fixed_reset_seed={int(fixed_reset_seed)}")

        processes: list[tuple[str, subprocess.Popen]] = []
        for algo in algo_list:
            cmd = base_cmd + [f"algorithm={algo}"]
            p = subprocess.Popen(cmd)
            processes.append((algo, p))

        failed = []
        for algo, p in processes:
            code = p.wait()
            if code != 0:
                failed.append((algo, code))
        if failed:
            details = ", ".join([f"{name}(exit={code})" for name, code in failed])
            raise SystemExit(f"并行训练失败: {details}")
        return

    for algo in algo_list:
        train_one(algo)


if __name__ == "__main__":
    main()
