"""RL 注册表、内置超参与 Hydra 合并路径的单元测试。"""

from __future__ import annotations

import importlib.util

import pytest

from onlyuav.models import load_default_components
from onlyuav.rl.builtin_defaults import get_builtin
from onlyuav.rl.hparams import load_algo_hparams
from onlyuav.rl.registry import SUPPORTED_ALGOS, assert_algo_supported, normalize_backend
from tests.support import compose_config


@pytest.mark.parametrize(
    "backend,algo",
    [(b, a) for b, algos in sorted(SUPPORTED_ALGOS.items()) for a in sorted(algos)],
)
def test_get_builtin_non_empty(backend: str, algo: str):
    h = get_builtin(backend, algo)
    assert isinstance(h, dict) and len(h) > 0


def test_normalize_backend_aliases():
    assert normalize_backend("sb3") == "sb3"
    assert normalize_backend("Di-Engine") == "di_engine"
    assert normalize_backend("ding") == "di_engine"
    assert normalize_backend("rllib") == "rllib"


def test_assert_algo_supported_accepts_registered():
    assert_algo_supported("tianshou", "ddpg")
    assert_algo_supported("rllib", "a2c")


def test_assert_algo_supported_rejects_unknown():
    with pytest.raises(ValueError, match="不支持算法"):
        assert_algo_supported("sb3", "not_an_algo")
    with pytest.raises(ValueError, match="不支持的 backend"):
        assert_algo_supported("unknown_backend", "ppo")


def test_load_algo_hparams_di_engine_ppo():
    load_default_components()
    cfg = compose_config(["experiment.train.backend=di_engine", "experiment.train.algorithm=ppo"])
    h = load_algo_hparams(cfg, "ppo")
    assert "learning_rate" in h and "batch_size" in h


def test_load_rllib_config_unknown_algorithm_raises():
    from onlyuav.rl.rllib_train import _load_rllib_config_class

    with pytest.raises(ValueError, match="不支持算法"):
        _load_rllib_config_class("not_an_algo")


def test_load_rllib_config_ppo_when_ray_available():
    if importlib.util.find_spec("ray.rllib.algorithms.ppo") is None:
        pytest.skip("Ray RLlib 未安装")
    from onlyuav.rl.rllib_train import _load_rllib_config_class

    cls = _load_rllib_config_class("ppo")
    assert cls.__name__ == "PPOConfig"


def test_load_rllib_config_td3_import_or_clear_error():
    """TD3 在较新 Ray 中可能已移除：成功导入或抛出带说明的 ValueError。"""
    if importlib.util.find_spec("ray.rllib") is None:
        pytest.skip("Ray RLlib 未安装")
    from onlyuav.rl.rllib_train import _load_rllib_config_class

    try:
        cls = _load_rllib_config_class("td3")
        assert "TD3" in cls.__name__
    except ValueError as exc:
        msg = str(exc).lower()
        assert "td3" in msg or "ray" in msg or "提供" in msg
