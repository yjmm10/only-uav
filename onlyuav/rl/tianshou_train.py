from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from torch.distributions import Independent, Normal

from onlyuav.rl.common import append_jsonl, attach_train_file_logger, build_env, to_jsonable
from onlyuav.rl.hparams import load_algo_hparams


def train_tianshou(
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
    try:
        from tianshou.data import Collector, VectorReplayBuffer
        from tianshou.env import DummyVectorEnv
        from tianshou.policy import PPOPolicy, SACPolicy
        from tianshou.utils.net.common import Net
        from tianshou.utils.net.continuous import Actor, ActorProb, Critic
    except ImportError as exc:
        raise SystemExit("请先安装 Tianshou：`uv sync --extra tianshou`") from exc

    h = load_algo_hparams(cfg, algo_name)
    key = algo_name.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_env_num = int(h.get("training_num_envs", 1))
    step_per_collect = int(h.get("step_per_collect", 2048))
    batch_size = int(h.get("batch_size", 64))
    repeat = int(h.get("repeat_per_update", 10))
    lr = float(h.get("lr", 3e-4))
    learning_starts = int(h.get("learning_starts", 5000))
    gradient_steps = int(h.get("gradient_steps", 1))

    def _make():
        return build_env(cfg, fixed_reset_seed)

    env0 = _make()
    assert isinstance(env0.action_space, gym.spaces.Box)
    state_shape = env0.observation_space.shape
    action_shape = env0.action_space.shape
    max_action = float(np.max(env0.action_space.high))

    train_envs = DummyVectorEnv([_make for _ in range(train_env_num)])

    run_logger = logging.getLogger(f"onlyuav.train.tianshou.{key}")
    run_logger.handlers.clear()
    run_logger.setLevel(logging.INFO)
    run_logger.propagate = True
    if train_log_path is not None:
        attach_train_file_logger(run_logger, train_log_path)
    if session_dir is not None:
        run_logger.info("session_dir=%s metric_jsonl=%s", session_dir, metric_jsonl_path)
    run_logger.info("FRESH_START tianshou algo=%s total_timesteps=%s device=%s", key, total_timesteps, device)

    hidden = list(h.get("hidden_sizes", [64, 64]))

    def dist_fn(*args):
        return Independent(Normal(*args), 1)

    if key == "ppo":
        net_a = Net(state_shape, hidden_sizes=hidden, device=device)
        actor = ActorProb(net_a, action_shape, max_action=max_action, unbounded=True, device=device).to(device)
        net_c = Net(state_shape, hidden_sizes=hidden, device=device)
        critic = Critic(net_c, device=device).to(device)
        torch.nn.init.constant_(actor.sigma_param, -0.5)
        optim = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)
        policy = PPOPolicy(
            actor,
            critic,
            optim,
            dist_fn,
            discount_factor=float(h.get("gamma", 0.99)),
            max_grad_norm=float(h.get("max_grad_norm", 0.5)),
            eps_clip=float(h.get("eps_clip", 0.2)),
            vf_coef=float(h.get("vf_coef", 0.5)),
            ent_coef=float(h.get("ent_coef", 0.0)),
            advantage_normalization=bool(h.get("advantage_normalization", True)),
            recompute_advantage=bool(h.get("recompute_advantage", True)),
            value_clip=bool(h.get("value_clip", True)),
            action_space=env0.action_space,
            action_scaling=True,
            gae_lambda=float(h.get("gae_lambda", 0.95)),
        ).to(device)
    elif key == "sac":
        net_a = Net(state_shape, hidden_sizes=hidden, device=device)
        actor = Actor(net_a, action_shape, max_action=max_action, device=device).to(device)
        net_c1 = Net(state_shape, hidden_sizes=hidden, device=device)
        net_c2 = Net(state_shape, hidden_sizes=hidden, device=device)
        critic1 = Critic(net_c1, device=device).to(device)
        critic2 = Critic(net_c2, device=device).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=lr)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=lr)
        if bool(h.get("auto_alpha", True)):
            target_entropy = -float(h.get("target_entropy_scale", 1.0)) * float(np.prod(action_shape))
            log_alpha = torch.zeros(1, requires_grad=True, device=device)
            alpha_optim = torch.optim.Adam([log_alpha], lr=lr)
            alpha_arg: float | tuple = (target_entropy, log_alpha, alpha_optim)
        else:
            alpha_arg = float(h.get("alpha", 0.2))
        policy = SACPolicy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=float(h.get("tau", 0.005)),
            gamma=float(h.get("gamma", 0.99)),
            alpha=alpha_arg,
            action_space=env0.action_space,
            reward_normalization=bool(h.get("reward_normalization", False)),
            estimation_step=int(h.get("estimation_step", 1)),
        ).to(device)
    else:
        raise ValueError(f"tianshou 后端不支持算法 {algo_name!r}")

    buf_size = max(step_per_collect * 4, train_env_num * 2000)
    buffer = VectorReplayBuffer(buf_size, buffer_num=train_env_num)
    collector = Collector(policy, train_envs, buffer, exploration_noise=True)

    out_torch = Path(str(model_path))
    if out_torch.suffix.lower() == ".zip":
        out_torch = out_torch.with_suffix("")
    out_torch = Path(str(out_torch) + "_tianshou.pt")

    if resume_default and out_torch.exists():
        run_logger.info("RESUME tianshou checkpoint=%s", out_torch)
        ck = torch.load(out_torch, map_location=device)
        policy.load_state_dict(ck["policy"])

    metric_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_t = time.time()
    env_steps = 0
    last_log = -1
    ep_returns: list[float] = []

    collector.reset()
    while env_steps < total_timesteps:
        n = min(step_per_collect, total_timesteps - env_steps)
        result = collector.collect(n_step=n)
        env_steps += int(result["n/st"])

        if key == "ppo":
            policy.update(0, collector.buffer, batch_size=batch_size, repeat=repeat)
            collector.reset_buffer(keep_statistics=True)
        else:
            if env_steps >= learning_starts:
                for _ in range(gradient_steps):
                    policy.update(batch_size, collector.buffer)

        if "rews" in result and len(result["rews"]) > 0:
            ep_returns.extend([float(x) for x in result["rews"]])

        if env_steps - last_log >= log_every_steps or env_steps >= total_timesteps:
            last_log = env_steps
            row = {
                "library": "tianshou",
                "algorithm": key,
                "timesteps": env_steps,
                "elapsed_s": round(time.time() - start_t, 3),
                "collect/rew": float(result.get("rew", 0.0)),
                "collect/n_ep": int(result.get("n/ep", 0)),
            }
            if include_timestamp:
                row["timestamp"] = datetime.utcnow().isoformat()
            if include_step_reward:
                row["step_reward"] = None
            append_jsonl(metric_jsonl_path, row)

    torch.save({"policy": policy.state_dict(), "algo": key}, out_torch)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "library": "tianshou",
                "algorithm": key,
                "checkpoint": str(out_torch),
                "timesteps": env_steps,
                "episode_rewards_tail": to_jsonable(ep_returns[-50:]),
            },
            f,
            indent=2,
        )
    run_logger.info("end training checkpoint=%s", out_torch)
    print(f"[tianshou/{key}] Checkpoint: {out_torch}")
    print(f"[tianshou/{key}] Metrics: {metric_jsonl_path}")
