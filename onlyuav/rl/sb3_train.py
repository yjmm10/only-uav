from __future__ import annotations

import json
import logging
import math
import time
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig
from stable_baselines3.common.callbacks import BaseCallback

from onlyuav.algorithms import get_algo_class
from onlyuav.rl.common import attach_train_file_logger, build_env, to_jsonable
from onlyuav.rl.hparams import load_algo_hparams


class _TrainingLogger:
    def __init__(self):
        self.episode_rewards: list[float] = []
        self.timesteps: list[int] = []
        self._acc = 0.0

    def on_step(self, reward, done, t):
        self._acc += float(reward)
        if done:
            self.episode_rewards.append(self._acc)
            self.timesteps.append(int(t))
            self._acc = 0.0


def _find_checkpoint(model_path: Path) -> Path | None:
    candidates = [model_path]
    if model_path.suffix != ".zip":
        candidates.append(model_path.with_suffix(".zip"))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def train_sb3(
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
    save_best_model: bool = True,
    best_metric_key: str = "rollout/ep_rew_mean",
    min_steps_between_best_saves: int = 5000,
) -> None:
    algo_class, key = get_algo_class(algo_name)
    algo_kwargs = dict(load_algo_hparams(cfg, key))
    policy_cls = str(algo_kwargs.pop("policy", "MlpPolicy"))

    env = build_env(cfg, fixed_reset_seed)
    logger = _TrainingLogger()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    metric_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    run_logger = logging.getLogger(f"onlyuav.train.sb3.{key}")
    run_logger.handlers.clear()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = True
    if train_log_path is not None:
        attach_train_file_logger(run_logger, train_log_path)

    best_model_base: Path | None = None
    if session_dir is not None and save_best_model:
        best_model_base = session_dir / "best_model"

    class TrainingCallback(BaseCallback):
        def __init__(self, **_kwargs):
            super().__init__(verbose=0)
            self.logger_obj = _kwargs["logger_obj"]
            self.metric_jsonl_path = _kwargs["metric_jsonl_path"]
            self.run_logger = _kwargs["run_logger"]
            self.algo_name = _kwargs["algo_name"]
            self.log_every_steps = max(int(_kwargs["log_every_steps"]), 1)
            self.include_step_reward = bool(_kwargs["include_step_reward"])
            self.include_timestamp = bool(_kwargs["include_timestamp"])
            self.save_best_model = bool(_kwargs.get("save_best_model", True))
            self.best_metric_key = str(_kwargs.get("best_metric_key", "rollout/ep_rew_mean"))
            self.min_steps_between_best_saves = max(int(_kwargs.get("min_steps_between_best_saves", 5000)), 0)
            self.best_model_base = _kwargs.get("best_model_base")
            self.best_reward = float("-inf")
            self.last_best_save_ts = -1
            self.last_metrics: dict[str, object] = {}
            self.last_logged_timestep = -1
            self.start_time = time.time()

        @staticmethod
        def _extract_metrics(logger_values: dict) -> dict[str, object]:
            tracked = (
                "rollout/ep_rew_mean",
                "rollout/ep_len_mean",
                "time/fps",
                "time/iterations",
                "time/time_elapsed",
                "time/total_timesteps",
                "train/approx_kl",
                "train/clip_fraction",
                "train/clip_range",
                "train/entropy_loss",
                "train/explained_variance",
                "train/learning_rate",
                "train/loss",
                "train/n_updates",
                "train/policy_gradient_loss",
                "train/std",
                "train/value_loss",
            )
            out: dict[str, object] = {}
            for k in tracked:
                if k in logger_values:
                    out[k] = to_jsonable(logger_values[k])
            return out

        def _emit_metrics(self, step_reward: float, done_flag: bool, force: bool = False) -> None:
            should_log = force or done_flag or (
                int(self.num_timesteps) - self.last_logged_timestep >= self.log_every_steps
            )
            if not should_log:
                return
            logger_values = getattr(self.model.logger, "name_to_value", {})
            extracted = self._extract_metrics(logger_values)
            if extracted:
                self.last_metrics.update(extracted)
            metrics = {
                "library": "sb3",
                "algorithm": self.algo_name,
                "timesteps": int(self.num_timesteps),
                "elapsed_s": round(time.time() - self.start_time, 3),
                "done": bool(done_flag),
                **self.last_metrics,
            }
            if self.include_timestamp:
                metrics["timestamp"] = datetime.utcnow().isoformat()
            if self.include_step_reward:
                metrics["step_reward"] = float(step_reward)
            infos = self.locals.get("infos")
            if infos and hasattr(infos, "__len__") and len(infos) > 0 and isinstance(infos[0], dict):
                ep_info = infos[0].get("episode")
                if isinstance(ep_info, dict):
                    if "r" in ep_info:
                        metrics["episode_reward"] = to_jsonable(ep_info["r"])
                    if "l" in ep_info:
                        metrics["episode_len"] = to_jsonable(ep_info["l"])
            with self.metric_jsonl_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            self.last_logged_timestep = int(self.num_timesteps)

        def _on_step(self) -> bool:
            rewards = self.locals.get("rewards", 0.0)
            dones = self.locals.get("dones", False)
            reward = rewards[0] if hasattr(rewards, "__len__") else rewards
            done = dones[0] if hasattr(dones, "__len__") else dones
            self.logger_obj.on_step(reward, done, self.num_timesteps)
            self._emit_metrics(step_reward=float(reward), done_flag=bool(done))
            return True

        def _on_training_end(self) -> None:
            self._emit_metrics(step_reward=0.0, done_flag=False, force=True)

    checkpoint = _find_checkpoint(model_path) if resume_default else None
    if checkpoint is not None:
        run_logger.info(
            "RESUME sb3 algo=%s checkpoint=%s add_timesteps=%s fixed_reset_seed=%s",
            key,
            checkpoint,
            total_timesteps,
            fixed_reset_seed,
        )
        model = algo_class.load(str(checkpoint), env=env)
        reset_num_timesteps = False
    else:
        run_logger.info(
            "FRESH_START sb3 algo=%s total_timesteps=%s fixed_reset_seed=%s",
            key,
            total_timesteps,
            fixed_reset_seed,
        )
        model = algo_class(policy_cls, env, verbose=1, seed=int(cfg.seed), **algo_kwargs)
        reset_num_timesteps = True

    cb = TrainingCallback(
        logger_obj=logger,
        metric_jsonl_path=metric_jsonl_path,
        run_logger=run_logger,
        algo_name=key,
        log_every_steps=log_every_steps,
        include_step_reward=include_step_reward,
        include_timestamp=include_timestamp,
    )
    model.learn(total_timesteps=total_timesteps, callback=cb, reset_num_timesteps=reset_num_timesteps)
    model.save(str(model_path))

    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"library": "sb3", "algorithm": key, "timesteps": logger.timesteps, "episode_rewards": logger.episode_rewards},
            f,
            indent=2,
        )
    run_logger.info("end training model=%s log_json=%s", model_path, log_path)
    print(f"[sb3/{key}] Model saved: {model_path}.zip")
    print(f"[sb3/{key}] Training log: {log_path}")
    print(f"[sb3/{key}] Iteration metrics: {metric_jsonl_path}")
