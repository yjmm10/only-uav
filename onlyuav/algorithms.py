from __future__ import annotations


def get_algo_class(algo_name: str):
    from stable_baselines3 import A2C, PPO, SAC, TD3

    mapping = {
        "ppo": PPO,
        "a2c": A2C,
        "sac": SAC,
        "td3": TD3,
    }
    key = algo_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported algo '{algo_name}'. Supported: {sorted(mapping)}")
    return mapping[key], key
