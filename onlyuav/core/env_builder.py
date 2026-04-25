from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


class ComponentRegistry:
    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(component_class: type):
            cls._registry[name] = component_class
            return component_class

        return decorator

    @classmethod
    def instantiate(cls, cfg: dict[str, Any]):
        comp_type = cfg["type"]
        params = cfg.get("params", {})
        if comp_type not in cls._registry:
            raise ValueError(f"Module type '{comp_type}' not registered.")
        return cls._registry[comp_type](**params)


class EnvBuilder:
    @staticmethod
    def build(module_cfg: DictConfig | dict[str, Any]):
        cfg = OmegaConf.to_container(module_cfg, resolve=True) if isinstance(module_cfg, DictConfig) else module_cfg
        modules: dict[str, Any] = {}
        for module_name, sub_cfg in cfg.items():
            if module_name in ("env_params", "meta"):
                continue
            if isinstance(sub_cfg, dict) and "type" in sub_cfg:
                modules[module_name] = ComponentRegistry.instantiate(sub_cfg)

        env_params = cfg.get("env_params", {})
        from onlyuav.envs.drone_env import DroneEnv

        return DroneEnv(
            mobility=modules["mobility"],
            channel=modules["channel"],
            power=modules["power"],
            task_gen=modules["task"],
            computing=modules["computing"],
            energy=modules["energy"],
            reward_model=modules["reward"],
            obs_model=modules["observation"],
            action_interp=modules["action_interpreter"],
            server_pos=env_params.get("server_pos", [300, 0, 50]),
            max_steps=env_params.get("max_steps", 200),
            dt=env_params.get("dt", 1.0),
        )
