from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IComputing


@ComponentRegistry.register("LocalOnly")
class LocalOnly(IComputing):
    def __init__(self, local_freq=1e9):
        self.local_freq = local_freq

    def reset(self, **kwargs):
        return None

    def process(self, task, offload_target, channel_rate):
        exec_time = task["req_cycles"] / self.local_freq
        energy = exec_time * 1e-11 * self.local_freq**2
        return {"exec_time": exec_time, "energy": energy, "success": True}
