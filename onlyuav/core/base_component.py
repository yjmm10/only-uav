from abc import ABC, abstractmethod


class BaseComponent(ABC):
    """所有可替换模块的根接口。"""

    @abstractmethod
    def reset(self, **kwargs) -> None:
        """重置内部状态到初始条件。"""

    def step(self, **kwargs):
        """单步更新（按需由子类实现）。"""
        raise NotImplementedError
