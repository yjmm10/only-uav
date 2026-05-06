from __future__ import annotations

import logging
from pathlib import Path

from easydict import EasyDict
from omegaconf import DictConfig, OmegaConf

from onlyuav.core.env_builder import EnvBuilder
from onlyuav.models import load_default_components
from onlyuav.rl.common import FixedSeedResetWrapper
from onlyuav.rl.ding_bundle import build_ppo_configs
from onlyuav.rl.hparams import load_algo_hparams


def train_di_engine(
    cfg: DictConfig,
    *,
    algo_name: str,
    total_timesteps: int,
    fixed_reset_seed: int | None,
    hydra_output_dir: Path,
    **_kwargs,
) -> None:
    try:
        from ding.entry.serial_entry_onpolicy import serial_pipeline_onpolicy
    except ImportError as exc:
        raise SystemExit("请先安装 DI-engine：`uv sync --extra di-engine`") from exc

    key = algo_name.lower()
    if key != "ppo":
        raise ValueError("di_engine 后端当前仅对接 PPO（可扩展 serial_pipeline 配置）")

    run_logger = logging.getLogger(f"onlyuav.train.di_engine.{key}")
    run_logger.handlers.clear()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = True

    hyper = load_algo_hparams(cfg, key)
    main_config, create_config = build_ppo_configs(hyper)
    exp_dir = hydra_output_dir / f"ding_{key}"
    main_config.exp_name = str(exp_dir)

    modules_container = OmegaConf.to_container(cfg.modules, resolve=True)

    def env_fn(env_cfg=None):
        load_default_components()
        env = EnvBuilder.build(OmegaConf.create(modules_container))
        if fixed_reset_seed is not None:
            env = FixedSeedResetWrapper(env, int(fixed_reset_seed))
        from ding.envs import DingEnvWrapper

        return DingEnvWrapper(env=env, cfg=EasyDict({}), is_gymnasium=True)

    cnum = int(main_config.env.collector_env_num)
    enum = int(main_config.env.evaluator_env_num)
    env_setting = [env_fn, [{} for _ in range(cnum)], [{} for _ in range(enum)]]

    run_logger.info("START di-engine PPO max_env_step=%s exp_name=%s", total_timesteps, main_config.exp_name)
    serial_pipeline_onpolicy(
        [main_config, create_config],
        seed=int(cfg.seed),
        env_setting=env_setting,
        max_env_step=int(total_timesteps),
    )
    print(f"[di_engine/{key}] 训练完成，输出目录: {main_config.exp_name}")
