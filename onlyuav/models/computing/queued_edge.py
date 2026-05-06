"""边缘侧 FIFO 排队：卸载任务在边缘服务器单队列等待，本步服务时间 = 传输 + 排队等待 + 计算。"""

from __future__ import annotations

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IComputing


@ComponentRegistry.register("QueuedEdgeOffloading")
class QueuedEdgeOffloading(IComputing):
    def __init__(
        self,
        edge_freq: float = 3e9,
        local_freq: float = 1e9,
        tx_power_w: float = 2.0,
        edge_latency_bias: float = 0.02,
    ):
        self.edge_freq = edge_freq
        self.local_freq = local_freq
        self.tx_power_w = tx_power_w
        self.edge_latency_bias = edge_latency_bias
        self._edge_remaining_compute: float = 0.0

    def reset(self, **kwargs):
        self._edge_remaining_compute = 0.0

    def process(self, task, offload_target, channel_rate):
        if offload_target == 0:
            exec_time = task["req_cycles"] / self.local_freq
            energy = exec_time * 1e-11 * self.local_freq**2
            return {"exec_time": exec_time, "energy": energy, "success": True}

        if channel_rate <= 0:
            return {"exec_time": 1e6, "energy": 0.0, "success": False}

        tx_time = task["data_size"] / channel_rate
        queue_wait = self._edge_remaining_compute
        proc_time = task["req_cycles"] / self.edge_freq + self.edge_latency_bias
        total_time = tx_time + queue_wait + proc_time
        comm_energy = tx_time * self.tx_power_w

        self._edge_remaining_compute = queue_wait + proc_time

        return {"exec_time": total_time, "energy": comm_energy, "success": True}
