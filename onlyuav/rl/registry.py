from __future__ import annotations

SUPPORTED_ALGOS: dict[str, frozenset[str]] = {
    "sb3": frozenset({"ppo", "a2c", "sac", "td3"}),
    "rllib": frozenset({"ppo", "sac"}),
    "tianshou": frozenset({"ppo", "sac"}),
    "di_engine": frozenset({"ppo"}),
}


def normalize_backend(name: str) -> str:
    key = name.strip().lower().replace("-", "_")
    aliases = {"di": "di_engine", "ding": "di_engine", "sb3": "sb3"}
    return aliases.get(key, key)


def assert_algo_supported(backend: str, algo: str) -> None:
    algo_l = algo.lower()
    if backend not in SUPPORTED_ALGOS:
        raise ValueError(f"不支持的 backend: {backend!r}。可选: {sorted(SUPPORTED_ALGOS)}")
    if algo_l not in SUPPORTED_ALGOS[backend]:
        raise ValueError(
            f"backend={backend!r} 不支持算法 {algo!r}。"
            f"该库可用: {sorted(SUPPORTED_ALGOS[backend])}"
        )
