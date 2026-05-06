"""部分可观测：超出可见距离时掩蔽服务器位置（置零），其余与 FullObs 一致，保持 8 维观测。"""

from __future__ import annotations

import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IObservation


@ComponentRegistry.register("PartialObs")
class PartialObservation(IObservation):
    def __init__(self, visible_range_m: float = 250.0):
        self.visible_range_m = float(visible_range_m)

    def reset(self, **kwargs):
        return None

    def get_obs(self, env_state):
        pos = env_state["pos"]
        server = env_state["server_pos"]
        dist = float(np.linalg.norm(pos[:2] - server[:2]))
        sx, sy = float(server[0]), float(server[1])
        if dist > self.visible_range_m:
            sx, sy = 0.0, 0.0

        return np.array(
            [
                float(pos[0]),
                float(pos[1]),
                float(env_state["vel"][0]),
                float(env_state["vel"][1]),
                float(env_state["energy"]),
                float(env_state["queue_len"]),
                sx,
                sy,
            ],
            dtype=np.float32,
        )
