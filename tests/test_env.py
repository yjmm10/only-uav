from pathlib import Path

from hydra import compose, initialize_config_dir

from onlyuav.core.env_builder import EnvBuilder
from onlyuav.models import load_default_components


def _compose(overrides=None):
    config_dir = str((Path(__file__).resolve().parents[1] / "configs"))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        return compose(config_name="config", overrides=overrides or [])


def test_env_runs():
    load_default_components()
    cfg = _compose()
    env = EnvBuilder.build(cfg.modules)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape


def test_hydra_override_local_only():
    load_default_components()
    cfg = _compose(["experiment=local_only", "modules/computing=local_only"])
    env = EnvBuilder.build(cfg.modules)
    assert type(env.computing).__name__ == "LocalOnly"
