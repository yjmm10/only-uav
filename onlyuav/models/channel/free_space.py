import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IChannel


@ComponentRegistry.register("FreeSpace")
class FreeSpaceChannel(IChannel):
    def __init__(self, freq=2.4e9, tx_power_dbm=20.0, bandwidth=10e6):
        self.freq = freq
        self.tx_power = 10 ** ((tx_power_dbm - 30) / 10)
        self.bandwidth = bandwidth
        self.noise_power = 1.38e-23 * 290 * bandwidth

    def reset(self, **kwargs):
        return None

    def rate(self, tx_pos, rx_pos):
        d = np.linalg.norm(np.array(tx_pos) - np.array(rx_pos)) + 1e-6
        path_loss = (4 * np.pi * d * self.freq / 3e8) ** 2
        rx_power = self.tx_power / path_loss
        snr = rx_power / self.noise_power
        return float(self.bandwidth * np.log2(1 + snr))
