import numpy as np

from onlyuav.core.env_builder import ComponentRegistry
from onlyuav.core.interfaces import ITaskGenerator


@ComponentRegistry.register("PoissonArrival")
class PoissonArrival(ITaskGenerator):
    def __init__(self, arrival_rate=0.3, data_size_mean=1e6, req_cycles_mean=5e8, max_delay=2.0):
        self.rate = arrival_rate
        self.data_mean = data_size_mean
        self.cycles_mean = req_cycles_mean
        self.max_delay = max_delay
        self.next_id = 0

    def reset(self, **kwargs):
        self.next_id = 0

    def sample(self, current_time):
        n_tasks = np.random.poisson(self.rate)
        tasks = []
        for _ in range(n_tasks):
            tasks.append(
                {
                    "id": self.next_id,
                    "data_size": float(np.random.exponential(self.data_mean)),
                    "req_cycles": float(np.random.exponential(self.cycles_mean)),
                    "max_delay": self.max_delay,
                    "arrival_time": current_time,
                }
            )
            self.next_id += 1
        return tasks
