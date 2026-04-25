from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IComputing


@ComponentRegistry.register("EdgeOffloading")
class EdgeOffloading(IComputing):
    def __init__(self, edge_freq=3e9, local_freq=1e9, tx_power_w=2.0, edge_latency_bias=0.02):
        self.edge_freq = edge_freq
        self.local_freq = local_freq
        self.tx_power_w = tx_power_w
        self.edge_latency_bias = edge_latency_bias

    def reset(self, **kwargs):
        return None

    def process(self, task, offload_target, channel_rate):
        if offload_target == 0:
            exec_time = task["req_cycles"] / self.local_freq
            energy = exec_time * 1e-11 * self.local_freq**2
            return {"exec_time": exec_time, "energy": energy, "success": True}

        if channel_rate <= 0:
            return {"exec_time": 1e6, "energy": 0.0, "success": False}

        tx_time = task["data_size"] / channel_rate
        proc_time = task["req_cycles"] / self.edge_freq + self.edge_latency_bias
        total_time = tx_time + proc_time
        comm_energy = tx_time * self.tx_power_w
        return {"exec_time": total_time, "energy": comm_energy, "success": True}
