from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from onlyuav.core.env_builder import EnvBuilder
from onlyuav.evaluation.evaluator import evaluate_policy
from onlyuav.models import load_default_components

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")

@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_default_components()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("请先安装 examples 依赖：`uv sync --extra examples`") from exc

    model_path = Path(cfg.experiment.train.model_path)
    out_path = Path(cfg.experiment.eval.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = EnvBuilder.build(cfg.modules)
    env.reset(seed=int(cfg.seed))
    model = PPO.load(str(model_path))

    def policy_fn(obs):
        return model.predict(obs, deterministic=True)[0]

    summary, _ = evaluate_policy(env, policy_fn, num_episodes=int(cfg.experiment.eval.episodes), verbose=True)

    serializable = {}
    for key, value in summary.items():
        if isinstance(value, tuple):
            serializable[key] = {"mean": value[0], "std": value[1]}
        else:
            serializable[key] = value
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Evaluation JSON: {out_path}")


if __name__ == "__main__":
    main()
