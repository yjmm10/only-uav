from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from onlyuav.algorithms import get_algo_class
from onlyuav.core.env_builder import EnvBuilder
from onlyuav.evaluation.evaluator import evaluate_policy
from onlyuav.models import load_default_components
from onlyuav.rl.hparams import resolve_training_algorithm_name
from onlyuav.rl.registry import normalize_backend

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")


def _resolve_output_path(raw: str, algo: str, multi_mode: bool) -> Path:
    if "{algo}" in raw:
        return Path(raw.format(algo=algo))
    if multi_mode:
        base = Path(raw)
        return base.with_name(f"{base.stem}_{algo}{base.suffix}")
    return Path(raw)

@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    load_default_components()
    try:
        import stable_baselines3  # noqa: F401
    except ImportError as exc:
        raise SystemExit("请先安装依赖：`uv sync`") from exc

    backend = normalize_backend(str(cfg.experiment.train.get("backend", "sb3")))
    if backend != "sb3":
        raise SystemExit(
            f"当前 onlyuav.eval 仅支持 Stable-Baselines3 导出的 .zip 模型，"
            f"配置中 experiment.train.backend={backend!r}。"
            f"请使用 backend=sb3 训练评估，或为其它库单独编写加载与推理逻辑。"
        )

    eval_algos_cfg = cfg.experiment.eval.get("algos")
    eval_algo = cfg.experiment.eval.get("algo")
    train_algos_cfg = cfg.experiment.train.get("algos")
    if eval_algos_cfg:
        algo_list = [str(a).lower() for a in eval_algos_cfg]
    elif eval_algo:
        algo_list = [str(eval_algo).lower()]
    elif train_algos_cfg:
        algo_list = [str(a).lower() for a in train_algos_cfg]
    else:
        algo_list = [resolve_training_algorithm_name(cfg)]

    multi_mode = len(algo_list) > 1

    for algo in algo_list:
        algo_class, algo_name = get_algo_class(algo)
        model_path = _resolve_output_path(str(cfg.experiment.train.model_path), algo_name, multi_mode)
        out_path = _resolve_output_path(str(cfg.experiment.eval.output_json), algo_name, multi_mode)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        env = EnvBuilder.build(cfg.modules)
        env.reset(seed=int(cfg.seed))
        model = algo_class.load(str(model_path))

        def policy_fn(obs):
            return model.predict(obs, deterministic=True)[0]

        summary, _ = evaluate_policy(env, policy_fn, num_episodes=int(cfg.experiment.eval.episodes), verbose=True)

        serializable = {"algorithm": algo_name}
        for key, value in summary.items():
            if isinstance(value, tuple):
                serializable[key] = {"mean": value[0], "std": value[1]}
            else:
                serializable[key] = value
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

        print(f"[{algo_name}] Evaluation JSON: {out_path}")


if __name__ == "__main__":
    main()
