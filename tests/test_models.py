"""onlyuav.models 各组件：注册、实例化与核心数值行为的单元测试。"""

from __future__ import annotations

import json

import numpy as np
import pytest

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.models import load_default_components


@pytest.fixture(scope="module", autouse=True)
def _registered_components():
    load_default_components()


def _make(type_name: str, params: dict | None = None):
    return ComponentRegistry.instantiate({"type": type_name, "params": params or {}})


def test_registry_rejects_unknown_type():
    with pytest.raises(ValueError, match="not registered"):
        ComponentRegistry.instantiate({"type": "NoSuchComponent", "params": {}})


def test_free_space_rate_positive_and_distance_trend():
    ch = _make("FreeSpace")
    near = np.array([0.0, 0.0, 50.0])
    far = np.array([300.0, 0.0, 50.0])
    server = np.array([100.0, 0.0, 50.0])
    r_near = ch.rate(near, server)
    r_far = ch.rate(far, server)
    assert r_near > 0 and r_far > 0
    assert r_near > r_far


def test_spectrum_sharing_vs_free_space_scaling():
    fs = _make("FreeSpace", {"freq": 2.4e9, "tx_power_dbm": 20.0, "bandwidth": 10e6})
    ss = _make(
        "SpectrumSharing",
        {"freq": 2.4e9, "tx_power_dbm": 20.0, "bandwidth": 10e6, "interference_factor": 0.35},
    )
    tx = np.array([0.0, 0.0, 50.0])
    rx = np.array([200.0, 0.0, 50.0])
    r_fs = fs.rate(tx, rx)
    r_ss = ss.rate(tx, rx)
    assert r_ss == pytest.approx(r_fs * 0.65)


def test_interference_limited_positive():
    ch = _make("InterferenceLimited", {"orthogonality_factor": 0.5})
    r = ch.rate(np.zeros(3), np.array([100.0, 0.0, 50.0]))
    assert r > 0


def test_probabilistic_los_rate_positive():
    ch = _make("ProbabilisticLOS")
    r = ch.rate(np.zeros(3), np.array([150.0, 0.0, 50.0]))
    assert r > 0


def test_simple_point_mass_clip_and_state():
    mob = _make("SimplePointMass", {"bounds": 10.0, "max_speed": 5.0, "dt": 1.0})
    mob.reset(init_pos=[0.0, 0.0, 50.0])
    for _ in range(50):
        mob.step([10.0, 0.0])
    st = mob.state()
    assert np.all(st["pos"][:2] >= -10.0) and np.all(st["pos"][:2] <= 10.0)
    assert "vel" in st and st["vel"].shape == (3,)


def test_random_waypoint_moves_and_fixed_altitude():
    np.random.seed(0)
    mob = _make("RandomWaypoint", {"bounds": 100.0, "max_speed": 5.0, "waypoint_radius": 200.0})
    mob.reset(init_pos=[0.0, 0.0, 50.0])
    p0 = mob.state()["pos"].copy()
    mob.step(np.zeros(3))
    p1 = mob.state()["pos"]
    assert not np.allclose(p0[:2], p1[:2])
    assert p1[2] == pytest.approx(50.0)


def test_simple_power_increases_with_speed_and_load():
    pwr = _make("SimplePower")
    e0 = pwr.compute(np.zeros(3), 0.0)
    e1 = pwr.compute(np.array([10.0, 0.0, 0.0]), 0.0)
    e2 = pwr.compute(np.zeros(3), 1e9)
    assert e1 > e0 and e2 > e0


def test_poisson_arrival_deterministic_with_seed():
    np.random.seed(123)
    gen = _make("PoissonArrival", {"arrival_rate": 1.5})
    gen.reset()
    tasks = gen.sample(0)
    assert isinstance(tasks, list)
    for t in tasks:
        assert {"id", "data_size", "req_cycles", "max_delay", "arrival_time"} <= set(t.keys())


def test_edge_offloading_local_vs_offload():
    comp = _make("EdgeOffloading")
    task = {"data_size": 1e6, "req_cycles": 3e8}
    loc = comp.process(task, 0, 1e9)
    edge = comp.process(task, 1, 1e9)
    assert loc["success"] and edge["success"]
    assert loc["exec_time"] > 0 and edge["exec_time"] > 0


def test_edge_offloading_zero_channel_fails_offload():
    comp = _make("EdgeOffloading")
    task = {"data_size": 1e6, "req_cycles": 3e8}
    bad = comp.process(task, 1, 0.0)
    assert bad["success"] is False


def test_local_only_ignores_offload_flag():
    comp = _make("LocalOnly", {"local_freq": 1e9})
    task = {"req_cycles": 1e9}
    a = comp.process(task, 0, 1e9)
    b = comp.process(task, 1, 1e9)
    assert a["exec_time"] == b["exec_time"]


def test_queued_edge_second_task_includes_queue_wait():
    comp = _make("QueuedEdgeOffloading")
    task = {"data_size": 1e5, "req_cycles": 1e9}
    rate = 5e6
    first = comp.process(task, 1, rate)
    second = comp.process(task, 1, rate)
    assert second["exec_time"] >= first["exec_time"]


def test_finite_battery_consume_and_clamp():
    bat = _make("FiniteBattery", {"capacity": 100.0})
    bat.reset()
    bat.consume(30.0, dt=1.0)
    assert bat.remaining() == pytest.approx(70.0)
    bat.consume(1e9, dt=1.0)
    assert bat.remaining() == 0.0


def test_infinite_battery_nominal():
    bat = _make("InfiniteBattery", {"nominal_level": 42.0})
    bat.consume(999.0, dt=1.0)
    assert bat.remaining() == 42.0


def test_energy_harvesting_bounds():
    bat = _make("EnergyHarvesting", {"capacity": 100.0, "harvest_rate": 50.0})
    bat.reset()
    bat.consume(200.0, dt=1.0)
    assert 0.0 <= bat.remaining() <= 100.0


def test_weighted_sum_reward():
    rw = _make("WeightedSum", {"w_throughput": 2.0, "w_energy": 1.0, "w_delay": 1.0})
    r = rw.compute({"completed": 3, "energy_cost": 1.0, "delay": 0.5})
    assert r == pytest.approx(3 * 2.0 - 1.0 - 0.5)


def test_sparse_completion_reward():
    rw = _make("SparseCompletion", {"completion_bonus": 5.0})
    assert rw.compute({"completed": 1}) == 5.0
    assert rw.compute({"completed": 0}) == 0.0


def test_constrained_reward_penalty_when_over_limits():
    rw = _make(
        "ConstrainedReward",
        {
            "delay_soft_limit": 1.0,
            "energy_step_soft_limit": 10.0,
            "penalty_delay": 1.0,
            "penalty_energy": 1.0,
        },
    )
    low = rw.compute({"completed": 0, "energy_cost": 1.0, "delay": 0.5})
    high_delay = rw.compute({"completed": 0, "energy_cost": 1.0, "delay": 2.0})
    assert high_delay < low


def test_full_obs_shape_and_partial_masks_server():
    env_state = {
        "pos": np.array([0.0, 0.0, 50.0]),
        "vel": np.array([1.0, 2.0, 0.0]),
        "energy": 100.0,
        "queue_len": 2,
        "server_pos": np.array([400.0, 0.0, 50.0]),
    }
    full = _make("FullObs")
    assert full.get_obs(env_state).shape == (8,)
    partial = _make("PartialObs", {"visible_range_m": 50.0})
    obs = partial.get_obs(env_state)
    assert obs[-2] == 0.0 and obs[-1] == 0.0


def test_standard_interpreter():
    it = _make("StandardInterpreter", {"fixed_local_cpu_freq": 2e9})
    move, offload, freq = it.interpret(np.array([0.1, -0.2, 0.9], dtype=np.float32))
    assert np.allclose(move, [0.1, -0.2])
    assert offload == 1
    assert freq == 2e9


def test_trace_driven_json(tmp_path):
    trace = tmp_path / "t.json"
    trace.write_text(
        json.dumps(
            [
                {"data_size": 100.0, "req_cycles": 200.0, "max_delay": 1.0},
            ]
        ),
        encoding="utf-8",
    )
    gen = _make("TraceDriven", {"trace_file": str(trace), "loop": True})
    gen.reset()
    assert len(gen.sample(0)) == 1
    assert len(gen.sample(1)) == 1
