from __future__ import annotations

import hydra
import numpy as np
from omegaconf import DictConfig

from onlyuav.core.env_builder import EnvBuilder
from onlyuav.models import load_default_components


def print_composition(env):
    print("========== Environment Composition ==========")
    print(f"Mobility:           {type(env.mobility).__name__}")
    print(f"Channel:            {type(env.channel).__name__}")
    print(f"Power:              {type(env.power).__name__}")
    print(f"Task Generator:     {type(env.task_gen).__name__}")
    print(f"Computing:          {type(env.computing).__name__}")
    print(f"Energy:             {type(env.energy_model).__name__}")
    print(f"Reward:             {type(env.reward_model).__name__}")
    print(f"Observation:        {type(env.obs_model).__name__}")
    print(f"Action Interpreter: {type(env.action_interp).__name__}")
    print(f"Server position:    {env.server_pos}")
    print("=============================================")


def run_random_episode(env):
    _, _ = env.reset()
    done = False
    reward_sum = 0.0
    while not done:
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        reward_sum += float(reward)
        done = terminated or truncated
    return reward_sum


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    load_default_components()
    env = EnvBuilder.build(cfg.modules)
    print_composition(env)
    rewards = [run_random_episode(env) for _ in range(2)]
    print(f"Random policy average reward (2 episodes): {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
