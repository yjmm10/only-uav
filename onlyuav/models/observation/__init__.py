"""Observation models."""

from onlyuav.models.observation.full_obs import FullObservation  # noqa: F401
from onlyuav.models.observation.partial_obs import PartialObservation  # noqa: F401

__all__ = ["FullObservation", "PartialObservation"]
