"""
各 backend 下各算法的内置默认超参（与对应库 API 一致）。

切换算法时只需改 experiment.train.algorithm（或 experiment.train.algos 列表），
无需为每个算法单独建 YAML。可选：

- experiment.train.shared_algo_kwargs：同一 backend 下多算法共用的键（如 learning_rate、gamma）；
- configs/algorithm/<backend>/*.yaml：若存在则叠加覆盖（便于论文复现时微调）；
- experiment.train.algo_kwargs：最高优先级覆盖。
"""
from __future__ import annotations

from typing import Any

# ---- Stable-Baselines3 ----
SB3: dict[str, dict[str, Any]] = {
    "ppo": {
        "policy": "MlpPolicy",
        "learning_rate": 3.0e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    },
    "a2c": {
        "policy": "MlpPolicy",
        "learning_rate": 7.0e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "rms_prop_eps": 1.0e-5,
        "use_rms_prop": True,
    },
    "sac": {
        "policy": "MlpPolicy",
        "learning_rate": 3.0e-4,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "use_sde": False,
        "sde_sample_freq": -1,
    },
    "td3": {
        "policy": "MlpPolicy",
        "learning_rate": 1.0e-3,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 100,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
}

# ---- Ray RLlib (training() 常用键) ----
# 注：td3 / ddpg / a2c 依赖 Ray 自带模块；较新 Ray（约 2.38+）可能已移除，训练时会提示改用 sb3/tianshou 或降级 ray。
RLLIB: dict[str, dict[str, Any]] = {
    "ppo": {"lr": 3.0e-4, "gamma": 0.99, "train_batch_size": 4000},
    "sac": {"lr": 3.0e-4, "gamma": 0.99, "train_batch_size": 256, "tau": 0.005},
    "a2c": {"lr": 7.0e-4, "gamma": 0.99, "train_batch_size": 4000},
    "td3": {"lr": 1.0e-3, "gamma": 0.99, "train_batch_size": 100, "tau": 0.005},
    "ddpg": {"lr": 1.0e-4, "gamma": 0.99, "train_batch_size": 256, "tau": 0.005},
}

# ---- Tianshou ----
TIANSHOU: dict[str, dict[str, Any]] = {
    "ppo": {
        "lr": 3.0e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "step_per_collect": 2048,
        "repeat_per_update": 10,
        "training_num_envs": 1,
        "hidden_sizes": [64, 64],
        "max_grad_norm": 0.5,
        "eps_clip": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.0,
        "advantage_normalization": True,
        "recompute_advantage": True,
        "value_clip": True,
        "gae_lambda": 0.95,
    },
    "sac": {
        "lr": 3.0e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "step_per_collect": 2048,
        "training_num_envs": 1,
        "hidden_sizes": [64, 64],
        "learning_starts": 5000,
        "gradient_steps": 1,
        "auto_alpha": True,
        "target_entropy_scale": 1.0,
        "reward_normalization": False,
        "estimation_step": 1,
    },
    "a2c": {
        "lr": 7.0e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "step_per_collect": 2048,
        "repeat_per_update": 5,
        "training_num_envs": 1,
        "hidden_sizes": [64, 64],
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "gae_lambda": 1.0,
        "max_batchsize": 256,
    },
    "td3": {
        "lr": 1.0e-3,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 100,
        "step_per_collect": 2048,
        "training_num_envs": 1,
        "hidden_sizes": [64, 64],
        "learning_starts": 100,
        "gradient_steps": 1,
        "policy_noise": 0.2,
        "update_actor_freq": 2,
        "noise_clip": 0.5,
        "exploration_sigma": 0.1,
        "reward_normalization": False,
        "estimation_step": 1,
    },
    "ddpg": {
        "lr": 1.0e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 128,
        "step_per_collect": 2048,
        "training_num_envs": 1,
        "hidden_sizes": [64, 64],
        "learning_starts": 10000,
        "gradient_steps": 1,
        "exploration_sigma": 0.1,
        "reward_normalization": False,
        "estimation_step": 1,
    },
}

# ---- DI-engine PPO（供 ding_bundle 与 YAML 覆盖共用字段名）----
DI_ENGINE: dict[str, dict[str, Any]] = {
    "ppo": {
        "learning_rate": 1.0e-3,
        "batch_size": 32,
        "epoch_per_collect": 10,
        "collector_env_num": 4,
        "evaluator_env_num": 2,
        "n_evaluator_episode": 3,
        "n_sample": 4000,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_ratio": 0.2,
        "value_weight": 0.5,
        "entropy_weight": 0.0,
        "eval_freq": 200,
        "obs_shape": 8,
        "action_shape": 4,
        "encoder_hidden_size_list": [64, 64],
        "cuda": False,
    },
}

BUILTIN: dict[str, dict[str, dict[str, Any]]] = {
    "sb3": SB3,
    "rllib": RLLIB,
    "tianshou": TIANSHOU,
    "di_engine": DI_ENGINE,
}


def get_builtin(backend: str, algo: str) -> dict[str, Any]:
    b = backend.lower()
    a = algo.lower()
    if b not in BUILTIN or a not in BUILTIN[b]:
        raise KeyError(f"无内置默认: backend={b!r}, algorithm={a!r}")
    return dict(BUILTIN[b][a])
