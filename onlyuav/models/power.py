import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import IPower


@ComponentRegistry.register("SimplePower")
class SimplePower(IPower):
    def __init__(self, hover_power=100.0, speed_coeff=5.0, comp_coeff=1e-11):
        self.hover = hover_power
        self.speed_coeff = speed_coeff
        self.comp_coeff = comp_coeff

    def reset(self, **kwargs):
        return None

    def compute(self, velocity, computation_load):
        speed = np.linalg.norm(velocity)
        propulsion = self.hover + self.speed_coeff * speed**2
        return float(propulsion + computation_load * self.comp_coeff)
