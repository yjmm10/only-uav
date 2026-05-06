"""概率视距信道：自由空间路径损耗 + 分段 LOS 概率；用期望接收功率得到确定性的可达速率。"""

from __future__ import annotations

import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IChannel


@ComponentRegistry.register("ProbabilisticLOS")
class ProbabilisticLOSChannel(IChannel):
    def __init__(
        self,
        freq: float = 2.4e9,
        tx_power_dbm: float = 20.0,
        bandwidth: float = 10e6,
        los_prob_suburban: float = 0.85,
        los_prob_urban: float = 0.45,
        distance_threshold_m: float = 200.0,
        nlos_extra_loss_db: float = 15.0,
    ):
        self.freq = freq
        self.tx_power = 10 ** ((tx_power_dbm - 30) / 10)
        self.bandwidth = bandwidth
        self.noise_power = 1.38e-23 * 290 * bandwidth
        self.los_prob_suburban = los_prob_suburban
        self.los_prob_urban = los_prob_urban
        self.distance_threshold_m = distance_threshold_m
        self.nlos_loss_linear = 10 ** (nlos_extra_loss_db / 10)

    def reset(self, **kwargs):
        return None

    def rate(self, tx_pos, rx_pos):
        d = float(np.linalg.norm(np.array(tx_pos) - np.array(rx_pos)) + 1e-6)
        path_loss = (4 * np.pi * d * self.freq / 3e8) ** 2
        p_los = self.los_prob_suburban if d < self.distance_threshold_m else self.los_prob_urban
        # E[1/L] 意义下对两类链路取接收功率期望：P_rx = P_tx/path_loss * (p_los + (1-p_los)/L_nlos)
        rx_scale = p_los + (1.0 - p_los) / self.nlos_loss_linear
        rx_power = self.tx_power / path_loss * rx_scale
        snr = rx_power / self.noise_power
        return float(self.bandwidth * np.log2(1 + snr))
