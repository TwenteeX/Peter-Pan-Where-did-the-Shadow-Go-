"""
Stub World Engine for MVP: flat 2D table as single surface. No depth yet.
"""

from __future__ import annotations

from typing import Any, Optional

from peter_pan.protocols import WorldState, SurfacePatch
from .interface import WorldEngineInterface


class StubWorldEngine(WorldEngineInterface):
    """Single flat ground surface; no 3D mesh. For Phase 1–2 MVP."""

    def __init__(self, table_bounds_xy: Optional[list[list[float]]] = None):
        # Default: 1m x 1m table in XY, Z=0
        self._bounds = table_bounds_xy or [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]

    def update(
        self,
        point_cloud_path: Optional[str] = None,
        calibration: Optional[dict[str, Any]] = None,
    ) -> None:
        # Ignore for stub; could apply calibration to _bounds later
        pass

    def get_state(self) -> WorldState:
        # One surface: ground table
        vertices = [[x, y, 0.0] for x, y in self._bounds]
        return WorldState(
            surfaces=[
                SurfacePatch(
                    surface_id="table",
                    role="ground",
                    vertices_xyz=vertices,
                    normal_xyz=[0, 0, 1],
                )
            ]
        )
