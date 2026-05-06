from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from onlyuav.rl.hparams import resolve_training_algorithm_name
from onlyuav.rl.registry import assert_algo_supported, normalize_backend


def _session_artifact_path(session_dir: Path, raw_template: str, algo: str) -> Path:
    """配置中的相对路径只取其文件名，落到本次会话目录（outputs/<algo>/<时间>/）下。"""
    s = str(raw_template)
    formatted = s.format(algo=algo.lower()) if "{algo}" in s else s
    return session_dir / Path(formatted).name


def _unique_timestamp_dir(parent: Path, stamp: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    candidate = parent / stamp
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate.resolve()
    for i in range(2, 10_000):
        alt = parent / f"{stamp}_{i}"
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=True)
            return alt.resolve()
    raise RuntimeError(f"无法为 {parent}/{stamp} 分配唯一目录")


def _resolve_train_artifacts(
    cfg: DictConfig,
    algo: str,
    multi_mode: bool,
    hydra_output_dir: Path,
) -> dict:
    train_cfg = cfg.experiment.train
    if bool(train_cfg.get("timestamped_output", True)):
        root = Path(get_original_cwd()) / str(train_cfg.get("output_root", "outputs"))
        fmt = str(train_cfg.get("output_time_format", "%Y%m%d%H%M"))
        stamp = datetime.now().strftime(fmt)
        session_dir = _unique_timestamp_dir(root / algo.lower(), stamp)
        return {
            "session_dir": session_dir,
            "model_path": _session_artifact_path(session_dir, str(train_cfg.model_path), algo),
            "log_path": _session_artifact_path(session_dir, str(train_cfg.log_json), algo),
            "metric_jsonl_path": _session_artifact_path(session_dir, str(train_cfg.metric_jsonl), algo),
            "train_log_path": session_dir / "train.log",
            "hydra_output_dir_effective": session_dir,
        }

    model_path = _resolve_output_path(str(train_cfg.model_path), algo, multi_mode)
    log_path = _resolve_output_path(str(train_cfg.log_json), algo, multi_mode)
    metric_jsonl_path = _resolve_metric_path(
        str(train_cfg.metric_jsonl),
        algo,
        multi_mode,
        hydra_output_dir=hydra_output_dir,
    )
    return {
        "session_dir": None,
        "model_path": model_path,
        "log_path": log_path,
        "metric_jsonl_path": metric_jsonl_path,
        "train_log_path": None,
        "hydra_output_dir_effective": hydra_output_dir,
    }


def _resolve_output_path(raw: str, algo: str, multi_mode: bool) -> Path:
    if "{algo}" in raw:
        return Path(raw.format(algo=algo))
    if multi_mode:
        base = Path(raw)
        return base.with_name(f"{base.name}_{algo}")
    return Path(raw)


def _resolve_metric_path(raw: str, algo: str, multi_mode: bool, hydra_output_dir: Path) -> Path:
    metric_path = _resolve_output_path(raw, algo, multi_mode)
    if metric_path.is_absolute():
        return metric_path
    return hydra_output_dir / metric_path


def run_training(cfg: DictConfig) -> None:
    algos_cfg = cfg.experiment.train.get("algos")
    if algos_cfg:
        algo_list = [str(a).lower() for a in algos_cfg]
    else:
        algo_list = [resolve_training_algorithm_name(cfg)]

    multi_mode = len(algo_list) > 1
    total_timesteps = int(cfg.experiment.train.total_timesteps)
    fixed_reset_seed = cfg.experiment.train.get("fixed_reset_seed")
    resume_default = bool(cfg.experiment.train.get("resume", True))
    parallel_multi_algo = bool(cfg.experiment.train.get("parallel_multi_algo", True))
    log_every_steps = int(cfg.experiment.train.get("metrics_log_every_steps", 100))
    include_step_reward = bool(cfg.experiment.train.get("metrics_include_step_reward", False))
    include_timestamp = bool(cfg.experiment.train.get("metrics_include_timestamp", True))
    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    backend = normalize_backend(str(cfg.experiment.train.get("backend", "sb3")))

    def train_one(algo: str, force_multi_mode: bool = False):
        assert_algo_supported(backend, algo)
        current_multi = force_multi_mode or multi_mode
        paths = _resolve_train_artifacts(cfg, algo, current_multi, hydra_output_dir)
        model_path = paths["model_path"]
        log_path = paths["log_path"]
        metric_jsonl_path = paths["metric_jsonl_path"]
        train_log_path = paths["train_log_path"]
        session_dir = paths["session_dir"]
        hydra_output_dir_effective = paths["hydra_output_dir_effective"]

        common = dict(
            cfg=cfg,
            algo_name=algo,
            total_timesteps=total_timesteps,
            fixed_reset_seed=int(fixed_reset_seed) if fixed_reset_seed is not None else None,
            resume_default=resume_default,
            model_path=model_path,
            log_path=log_path,
            metric_jsonl_path=metric_jsonl_path,
            train_log_path=train_log_path,
            session_dir=session_dir,
            log_every_steps=log_every_steps,
            include_step_reward=include_step_reward,
            include_timestamp=include_timestamp,
        )

        if backend == "sb3":
            from onlyuav.rl.sb3_train import train_sb3

            train_sb3(
                **common,
                save_best_model=bool(cfg.experiment.train.get("save_best_model", True)),
                best_metric_key=str(cfg.experiment.train.get("best_metric_key", "rollout/ep_rew_mean")),
                min_steps_between_best_saves=int(cfg.experiment.train.get("min_steps_between_best_saves", 5000)),
            )
        elif backend == "rllib":
            from onlyuav.rl.rllib_train import train_rllib

            train_rllib(**common)
        elif backend == "tianshou":
            from onlyuav.rl.tianshou_train import train_tianshou

            train_tianshou(**common)
        elif backend == "di_engine":
            from onlyuav.rl.di_engine_train import train_di_engine

            train_di_engine(**common, hydra_output_dir=hydra_output_dir_effective)
        else:
            raise ValueError(f"未知 backend: {backend!r}")

    if multi_mode and parallel_multi_algo:
        choices = HydraConfig.get().runtime.choices
        base_cmd = [
            sys.executable,
            "-m",
            "onlyuav.train",
            f"modules={choices['modules']}",
            f"experiment={choices['experiment']}",
            f"experiment.train.backend={backend}",
            "experiment.train.algos=null",
            "experiment.train.parallel_multi_algo=false",
            f"experiment.train.total_timesteps={total_timesteps}",
            f"experiment.train.resume={'true' if resume_default else 'false'}",
            f"experiment.train.metrics_log_every_steps={log_every_steps}",
            f"experiment.train.metrics_include_step_reward={'true' if include_step_reward else 'false'}",
            f"experiment.train.metrics_include_timestamp={'true' if include_timestamp else 'false'}",
        ]
        if fixed_reset_seed is None:
            base_cmd.append("experiment.train.fixed_reset_seed=null")
        else:
            base_cmd.append(f"experiment.train.fixed_reset_seed={int(fixed_reset_seed)}")

        processes: list[tuple[str, subprocess.Popen]] = []
        for algo in algo_list:
            cmd = base_cmd + [f"experiment.train.algorithm={algo}"]
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
