"""简化的同频干扰：在自由空间 Shannon 容量上乘以 (0,1] 的频谱效率因子，模拟多用户共享导致的有效速率下降。"""

from __future__ import annotations

import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IChannel


@ComponentRegistry.register("InterferenceLimited")
class InterferenceLimitedChannel(IChannel):
    def __init__(
        self,
        freq: float = 2.4e9,
        tx_power_dbm: float = 20.0,
        bandwidth: float = 10e6,
        orthogonality_factor: float = 0.65,
    ):
        self.freq = freq
        self.tx_power = 10 ** ((tx_power_dbm - 30) / 10)
        self.bandwidth = bandwidth
        self.noise_power = 1.38e-23 * 290 * bandwidth
        self.orthogonality_factor = float(np.clip(orthogonality_factor, 1e-6, 1.0))

    def reset(self, **kwargs):
        return None

    def rate(self, tx_pos, rx_pos):
        d = float(np.linalg.norm(np.array(tx_pos) - np.array(rx_pos)) + 1e-6)
        path_loss = (4 * np.pi * d * self.freq / 3e8) ** 2
        rx_power = self.tx_power / path_loss
        snr = rx_power / self.noise_power
        cap = self.bandwidth * np.log2(1 + snr)
        return float(cap * self.orthogonality_factor)
