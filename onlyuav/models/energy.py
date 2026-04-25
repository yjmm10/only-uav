from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IEnergy


@ComponentRegistry.register("FiniteBattery")
class FiniteBattery(IEnergy):
    def __init__(self, capacity=5000.0):
        self.capacity = capacity
        self.energy = capacity

    def reset(self, **kwargs):
        self.energy = self.capacity

    def consume(self, power, dt=1.0):
        self.energy -= power * dt
        self.energy = max(self.energy, 0.0)

    def remaining(self):
        return float(self.energy)
