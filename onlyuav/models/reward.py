from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IReward


@ComponentRegistry.register("WeightedSum")
class WeightedSumReward(IReward):
    def __init__(self, w_throughput=10.0, w_energy=0.001, w_delay=1.0):
        self.w_t = w_throughput
        self.w_e = w_energy
        self.w_d = w_delay

    def reset(self, **kwargs):
        return None

    def compute(self, state):
        return float(
            state["completed"] * self.w_t
            - state["energy_cost"] * self.w_e
            - state["delay"] * self.w_d
        )
