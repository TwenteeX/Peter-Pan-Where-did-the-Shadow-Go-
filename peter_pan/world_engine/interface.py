"""
World Engine interface. Implementations build a unified 3D world from mesh/calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from peter_pan.protocols import WorldState


class WorldEngineInterface(ABC):
    """
    Input: 3D mesh / point cloud, calibration data.
    Output: WorldState (surfaces, global coords).
    """

    @abstractmethod
    def update(self, point_cloud_path: Optional[str] = None, calibration: Optional[dict[str, Any]] = None) -> None:
        """Update world from new geometry or calibration."""
        ...

    @abstractmethod
    def get_state(self) -> WorldState:
        """Return current world state (surfaces, coords)."""
        ...
