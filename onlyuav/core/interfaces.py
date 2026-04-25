from __future__ import annotations

from abc import abstractmethod
from typing import Any

from onlyuav.core.base_component import BaseComponent


class IMobility(BaseComponent):
    @abstractmethod
    def step(self, action: Any) -> dict:
        """根据动作返回新的位置、速度等。"""

    @abstractmethod
    def state(self) -> dict:
        """返回当前状态快照。"""


class IChannel(BaseComponent):
    @abstractmethod
    def rate(self, tx_pos, rx_pos) -> float:
        """返回信道速率 bps。"""


class IPower(BaseComponent):
    @abstractmethod
    def compute(self, velocity, computation_load) -> float:
        """返回总功耗。"""


class ITaskGenerator(BaseComponent):
    @abstractmethod
    def sample(self, current_time) -> list:
        """返回当前时间步到达的任务列表。"""


class IComputing(BaseComponent):
    @abstractmethod
    def process(self, task, offload_target, channel_rate) -> dict:
        """处理任务，返回执行时间、能耗、是否成功。"""


class IEnergy(BaseComponent):
    @abstractmethod
    def consume(self, power, dt) -> None:
        """消耗能量。"""

    @abstractmethod
    def remaining(self) -> float:
        """剩余能量。"""


class IReward(BaseComponent):
    @abstractmethod
    def compute(self, state: dict) -> float:
        """返回奖励。"""


class IObservation(BaseComponent):
    @abstractmethod
    def get_obs(self, env_state: dict):
        """返回观测向量。"""


class IActionInterpreter(BaseComponent):
    @abstractmethod
    def interpret(self, raw_action) -> tuple:
        """将原始动作映射为移动、卸载目标与资源分配。"""
