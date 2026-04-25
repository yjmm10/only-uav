import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IMobility


@ComponentRegistry.register("SimplePointMass")
class SimplePointMass(IMobility):
    def __init__(self, dt=1.0, max_speed=10.0, bounds=500.0, init_pos=None):
        self.dt = dt
        self.max_speed = max_speed
        self.bounds = bounds
        self.default_init_pos = init_pos if init_pos is not None else [0.0, 0.0, 50.0]
        self.reset()

    def reset(self, init_pos=None, **kwargs):
        start = init_pos if init_pos is not None else self.default_init_pos
        self.pos = np.array(start, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)

    def step(self, action):
        acc = np.clip(np.array(action, dtype=np.float64), -2.0, 2.0)
        self.vel[:2] += acc * self.dt
        speed = np.linalg.norm(self.vel[:2])
        if speed > self.max_speed:
            self.vel[:2] = self.vel[:2] / speed * self.max_speed
        self.pos += self.vel * self.dt
        self.pos[:2] = np.clip(self.pos[:2], -self.bounds, self.bounds)
        return self.state()

    def state(self) -> dict:
        return {"pos": self.pos.copy(), "vel": self.vel.copy()}
