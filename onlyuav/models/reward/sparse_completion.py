from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IReward


@ComponentRegistry.register("SparseCompletion")
class SparseCompletionReward(IReward):
    """仅在任务成功完成时给予固定正奖励，否则为 0。"""

    def __init__(self, completion_bonus: float = 1.0):
        self.completion_bonus = float(completion_bonus)

    def reset(self, **kwargs):
        return None

    def compute(self, state):
        return float(self.completion_bonus) if int(state.get("completed", 0)) else 0.0
