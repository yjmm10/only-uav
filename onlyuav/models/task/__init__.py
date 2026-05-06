"""Task generation models."""

from onlyuav.models.task.poisson_arrival import PoissonArrival  # noqa: F401
from onlyuav.models.task.trace_driven import TraceDrivenTasks  # noqa: F401

__all__ = ["PoissonArrival", "TraceDrivenTasks"]
