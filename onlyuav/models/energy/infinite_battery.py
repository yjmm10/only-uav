from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IEnergy


@ComponentRegistry.register("InfiniteBattery")
class InfiniteBattery(IEnergy):
    """不消耗、不终止；remaining 恒为正值，便于消融「能量约束」。"""

    def __init__(self, nominal_level: float = 1.0):
        self.nominal_level = float(nominal_level)

    def reset(self, **kwargs):
        return None

    def consume(self, power, dt=1.0):
        return None

    def remaining(self):
        return float(self.nominal_level)
