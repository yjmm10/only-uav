import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IObservation


@ComponentRegistry.register("FullObs")
class FullObservation(IObservation):
    def reset(self, **kwargs):
        return None

    def get_obs(self, env_state):
        return np.array(
            [
                env_state["pos"][0],
                env_state["pos"][1],
                env_state["vel"][0],
                env_state["vel"][1],
                env_state["energy"],
                env_state["queue_len"],
                env_state["server_pos"][0],
                env_state["server_pos"][1],
            ],
            dtype=np.float32,
        )
