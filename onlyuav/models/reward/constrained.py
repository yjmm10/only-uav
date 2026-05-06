from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IReward


@ComponentRegistry.register("ConstrainedReward")
class ConstrainedReward(IReward):
    """加权和 + 对时延/能耗超过软阈值的二次惩罚（规划文档中的约束型奖励简化版）。"""

    def __init__(
        self,
        w_throughput: float = 10.0,
        w_energy: float = 0.001,
        w_delay: float = 1.0,
        delay_soft_limit: float = 1.5,
        energy_step_soft_limit: float = 500.0,
        penalty_delay: float = 5.0,
        penalty_energy: float = 0.01,
    ):
        self.w_t = w_throughput
        self.w_e = w_energy
        self.w_d = w_delay
        self.delay_soft_limit = delay_soft_limit
        self.energy_step_soft_limit = energy_step_soft_limit
        self.penalty_delay = penalty_delay
        self.penalty_energy = penalty_energy

    def reset(self, **kwargs):
        return None

    def compute(self, state):
        base = (
            state["completed"] * self.w_t
            - state["energy_cost"] * self.w_e
            - state["delay"] * self.w_d
        )
        pen = 0.0
        if state["delay"] > self.delay_soft_limit:
            excess = state["delay"] - self.delay_soft_limit
            pen += self.penalty_delay * (excess**2)
        if state["energy_cost"] > self.energy_step_soft_limit:
            excess_e = state["energy_cost"] - self.energy_step_soft_limit
            pen += self.penalty_energy * (excess_e**2)
        return float(base - pen)
