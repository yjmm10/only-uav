from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

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


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_default_components()
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:
        raise SystemExit("请先安装 examples 依赖：`uv sync --extra examples`") from exc

    class TrainingCallback(BaseCallback):
        def __init__(self, logger_obj, verbose=0):
            super().__init__(verbose)
            self.logger_obj = logger_obj

        def _on_step(self) -> bool:
            self.logger_obj.on_step(self.locals["rewards"][0], self.locals["dones"][0], self.num_timesteps)
            return True

    env = EnvBuilder.build(cfg.modules)
    model = PPO("MlpPolicy", env, verbose=1, seed=int(cfg.seed))
    logger = _TrainingLogger()
    model.learn(total_timesteps=int(cfg.experiment.train.total_timesteps), callback=TrainingCallback(logger))

    model_path = Path(cfg.experiment.train.model_path)
    log_path = Path(cfg.experiment.train.log_json)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))

    with log_path.open("w", encoding="utf-8") as f:
        json.dump({"timesteps": logger.timesteps, "episode_rewards": logger.episode_rewards}, f, indent=2)

    print(f"Model saved: {model_path}.zip")
    print(f"Training log: {log_path}")


if __name__ == "__main__":
    main()
