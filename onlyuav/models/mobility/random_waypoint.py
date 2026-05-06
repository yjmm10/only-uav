"""随机路点机动：每步朝当前路点匀速飞行，到达后重新采样；忽略 RL 给出的移动指令（适用于非受控目标或基线轨迹）。"""

from __future__ import annotations

import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IMobility


@ComponentRegistry.register("RandomWaypoint")
class RandomWaypointMobility(IMobility):
    def __init__(
        self,
        dt: float = 1.0,
        max_speed: float = 8.0,
        bounds: float = 500.0,
        waypoint_radius: float = 20.0,
        init_pos: list[float] | None = None,
        fixed_altitude: float = 50.0,
    ):
        self.dt = dt
        self.max_speed = max_speed
        self.bounds = bounds
        self.waypoint_radius = waypoint_radius
        self.default_init_pos = init_pos if init_pos is not None else [0.0, 0.0, fixed_altitude]
        self.fixed_altitude = fixed_altitude
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.target = np.zeros(3, dtype=np.float64)

    def reset(self, init_pos=None, **kwargs):
        start = init_pos if init_pos is not None else self.default_init_pos
        self.pos = np.array(start, dtype=np.float64)
        self.pos[2] = self.fixed_altitude
        self.vel = np.zeros(3, dtype=np.float64)
        self._sample_waypoint()

    def _sample_waypoint(self):
        xy = np.random.uniform(-self.bounds, self.bounds, size=2)
        self.target = np.array([xy[0], xy[1], self.fixed_altitude], dtype=np.float64)

    def step(self, action):
        del action
        delta = self.target[:2] - self.pos[:2]
        dist = float(np.linalg.norm(delta))
        if dist < self.waypoint_radius:
            self._sample_waypoint()
            delta = self.target[:2] - self.pos[:2]
            dist = float(np.linalg.norm(delta)) + 1e-6
        direction = delta / dist
        self.vel[:2] = direction * self.max_speed
        self.vel[2] = 0.0
        self.pos[:2] += self.vel[:2] * self.dt
        self.pos[2] = self.fixed_altitude
        self.pos[:2] = np.clip(self.pos[:2], -self.bounds, self.bounds)
        return self.state()

    def state(self) -> dict:
        return {"pos": self.pos.copy(), "vel": self.vel.copy()}
