from pathlib import Path

import pytest
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


@pytest.mark.parametrize(
    "overrides",
    [
        ["modules/channel=spectrum_sharing"],
        ["modules/channel=prob_los"],
        ["modules/channel=interference_limited"],
        ["modules/mobility=random_waypoint"],
        ["modules/task=trace_driven"],
        ["modules/energy=infinite_battery"],
        ["modules/energy=energy_harvesting"],
        ["modules/observation=partial_obs"],
        ["modules/reward=sparse_completion"],
        ["modules/reward=constrained"],
        ["modules/computing=queued_edge"],
        [
            "modules/channel=spectrum_sharing",
            "modules/computing=queued_edge",
            "modules/reward=constrained",
        ],
    ],
)
def test_planning_doc_module_variants_run(overrides):
    load_default_components()
    cfg = _compose(overrides)
    env = EnvBuilder.build(cfg.modules)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (8,)
    for _ in range(10):
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (8,)


def test_trace_driven_loads_json_trace(tmp_path):
    load_default_components()
    trace = tmp_path / "trace.json"
    trace.write_text(
        '[{"data_size": 900000.0, "req_cycles": 360000000.0, "max_delay": 2.0}]',
        encoding="utf-8",
    )
    from onlyuav.core.env_builder import ComponentRegistry

    gen = ComponentRegistry.instantiate(
        {"type": "TraceDriven", "params": {"trace_file": str(trace), "loop": False}}
    )
    gen.reset()
    assert len(gen.sample(0)) == 1
    assert len(gen.sample(1)) == 0


def test_algo_hparams_per_backend():
    from onlyuav.rl.hparams import load_algo_hparams

    load_default_components()
    base = ["experiment.train.backend=sb3", "experiment.train.algorithm=ppo"]
    cfg = _compose(base)
    h = load_algo_hparams(cfg, "ppo")
    assert "policy" in h and "learning_rate" in h

    cfg2 = _compose(["experiment.train.backend=tianshou", "experiment.train.algorithm=ppo"])
    h2 = load_algo_hparams(cfg2, "ppo")
    assert "batch_size" in h2

    cfg3 = _compose(["experiment.train.backend=rllib", "experiment.train.algorithm=ppo"])
    h3 = load_algo_hparams(cfg3, "ppo")
    assert "lr" in h3

    cfg4 = _compose(["experiment.train.backend=sb3", "experiment.train.algorithm=sac"])
    h4 = load_algo_hparams(cfg4, "sac")
    assert h4["policy"] == "MlpPolicy" and "buffer_size" in h4
