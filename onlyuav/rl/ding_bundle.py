"""DI-engine 的 main_config / create_config 模板（PPO 连续控制）。"""
from __future__ import annotations

import torch.nn as nn
from copy import deepcopy
from typing import Any

from easydict import EasyDict


def build_ppo_configs(hyper: dict[str, Any]) -> tuple[EasyDict, EasyDict]:
    """根据 algorithm_ding/ppo.yaml 中的键合并到默认 PPO 配置。"""
    lr = float(hyper.get("learning_rate", 1e-3))
    batch_size = int(hyper.get("batch_size", 32))
    epoch_per_collect = int(hyper.get("epoch_per_collect", 10))
    collector_env_num = int(hyper.get("collector_env_num", 4))
    evaluator_env_num = int(hyper.get("evaluator_env_num", 2))
    n_evaluator_episode = int(hyper.get("n_evaluator_episode", 3))
    cuda = bool(hyper.get("cuda", False))
    obs_shape = int(hyper.get("obs_shape", 8))
    action_shape = int(hyper.get("action_shape", 4))
    hidden = list(hyper.get("encoder_hidden_size_list", [64, 64]))

    pendulum_like = dict(
        exp_name="onlyuav_ding_ppo",
        env=dict(
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            act_scale=False,
            n_evaluator_episode=n_evaluator_episode,
            stop_value=1e6,
        ),
        policy=dict(
            cuda=cuda,
            action_space="continuous",
            recompute_adv=True,
            model=dict(
                obs_shape=obs_shape,
                action_shape=action_shape,
                encoder_hidden_size_list=hidden,
                action_space="continuous",
                actor_head_layer_num=0,
                critic_head_layer_num=0,
                sigma_type="independent",
                activation=nn.Tanh(),
                bound_type="tanh",
            ),
            learn=dict(
                epoch_per_collect=epoch_per_collect,
                batch_size=batch_size,
                learning_rate=lr,
                value_weight=float(hyper.get("value_weight", 0.5)),
                entropy_weight=float(hyper.get("entropy_weight", 0.0)),
                clip_ratio=float(hyper.get("clip_ratio", 0.2)),
                adv_norm=True,
                value_norm=True,
                ignore_done=True,
            ),
            collect=dict(
                n_sample=int(hyper.get("n_sample", 4000)),
                unroll_len=1,
                discount_factor=float(hyper.get("gamma", 0.99)),
                gae_lambda=float(hyper.get("gae_lambda", 0.95)),
            ),
            eval=dict(evaluator=dict(eval_freq=int(hyper.get("eval_freq", 200)))),
        ),
    )
    main_config = EasyDict(deepcopy(pendulum_like))
    create_config = EasyDict(
        dict(
            env=dict(type="base"),
            env_manager=dict(type="base"),
            policy=dict(type="ppo"),
        )
    )
    return main_config, create_config
