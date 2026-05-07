import pytest

from onlyuav.core.env_builder import EnvBuilder
from onlyuav.models import load_default_components

from tests.support import compose_config


def test_env_runs():
    load_default_components()
    cfg = compose_config()
    env = EnvBuilder.build(cfg.modules)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    for _ in range(5):
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert obs.shape == env.observation_space.shape


def test_hydra_override_local_only():
    load_default_components()
    cfg = compose_config(["experiment=local_only", "modules/computing=local_only"])
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
    cfg = compose_config(overrides)
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
    cfg = compose_config(base)
    h = load_algo_hparams(cfg, "ppo")
    assert "policy" in h and "learning_rate" in h

    cfg2 = compose_config(["experiment.train.backend=tianshou", "experiment.train.algorithm=ppo"])
    h2 = load_algo_hparams(cfg2, "ppo")
    assert "batch_size" in h2

    cfg3 = compose_config(["experiment.train.backend=rllib", "experiment.train.algorithm=ppo"])
    h3 = load_algo_hparams(cfg3, "ppo")
    assert "lr" in h3

    cfg4 = compose_config(["experiment.train.backend=sb3", "experiment.train.algorithm=sac"])
    h4 = load_algo_hparams(cfg4, "sac")
    assert h4["policy"] == "MlpPolicy" and "buffer_size" in h4

    cfg5 = compose_config(["experiment.train.backend=tianshou", "experiment.train.algorithm=td3"])
    h5 = load_algo_hparams(cfg5, "td3")
    assert "policy_noise" in h5 and h5["update_actor_freq"] == 2

    cfg6 = compose_config(["experiment.train.backend=rllib", "experiment.train.algorithm=ddpg"])
    h6 = load_algo_hparams(cfg6, "ddpg")
    assert h6["train_batch_size"] == 256 and h6["tau"] == 0.005


def test_load_algo_hparams_merges_hydra_algorithm_sb3():
    """Hydra 组 algorithm/sb3 映射到 cfg.algorithm.sb3，并参与 SB3 超参合并。"""
    from onlyuav.rl.hparams import load_algo_hparams

    load_default_components()
    cfg = compose_config(
        [
            "experiment.train.backend=sb3",
            "experiment.train.algorithm=ppo",
            "+algorithm/sb3=ppo",
            "algorithm.sb3.learning_rate=0.001",
        ]
    )
    h = load_algo_hparams(cfg, "ppo")
    assert h["learning_rate"] == 0.001
