from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IEnergy


@ComponentRegistry.register("EnergyHarvesting")
class EnergyHarvestingBattery(IEnergy):
    """有限电量 + 每步按 harvest_rate 补能（简化太阳能模型），电量裁剪在 [0, capacity]。"""

    def __init__(self, capacity: float = 5000.0, harvest_rate: float = 2.0):
        self.capacity = float(capacity)
        self.harvest_rate = float(harvest_rate)
        self.energy = self.capacity

    def reset(self, **kwargs):
        self.energy = self.capacity

    def consume(self, power, dt=1.0):
        self.energy -= float(power) * float(dt)
        self.energy += self.harvest_rate * float(dt)
        self.energy = max(0.0, min(self.energy, self.capacity))

    def remaining(self):
        return float(self.energy)
