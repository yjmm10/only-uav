"""Computation / offloading models."""

from onlyuav.models.computing.edge_offloading import EdgeOffloading  # noqa: F401
from onlyuav.models.computing.local_only import LocalOnly  # noqa: F401
from onlyuav.models.computing.queued_edge import QueuedEdgeOffloading  # noqa: F401

__all__ = ["EdgeOffloading", "LocalOnly", "QueuedEdgeOffloading"]
