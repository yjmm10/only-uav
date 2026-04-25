from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IActionInterpreter


@ComponentRegistry.register("StandardInterpreter")
class StandardInterpreter(IActionInterpreter):
    def __init__(self, cpu_freq_min=0.5e9, cpu_freq_max=2e9):
        self.cpu_freq_min = cpu_freq_min
        self.cpu_freq_max = cpu_freq_max

    def reset(self, **kwargs):
        return None

    def interpret(self, raw_action):
        move = raw_action[:2]
        offload_target = int(raw_action[2] > 0.5)
        cpu_scale = max(float(raw_action[3]), 0.0)
        cpu_freq = self.cpu_freq_min + cpu_scale * (self.cpu_freq_max - self.cpu_freq_min)
        return move, offload_target, cpu_freq
