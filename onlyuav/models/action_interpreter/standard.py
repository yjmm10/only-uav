from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IActionInterpreter


@ComponentRegistry.register("StandardInterpreter")
class StandardInterpreter(IActionInterpreter):
    """动作仅含平面移动与卸载二值；机载 CPU 工作频率为固定硬件参数，不参与策略输出。"""

    def __init__(self, fixed_local_cpu_freq: float = 1e9):
        self.fixed_local_cpu_freq = float(fixed_local_cpu_freq)

    def reset(self, **kwargs):
        return None

    def interpret(self, raw_action):
        move = raw_action[:2]
        offload_target = int(raw_action[2] > 0.5)
        return move, offload_target, self.fixed_local_cpu_freq
