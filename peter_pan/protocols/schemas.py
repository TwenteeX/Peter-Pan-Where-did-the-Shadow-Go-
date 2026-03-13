"""
Data schemas for inter-module I/O. All fields are defined so that:
- Perception Hub, World Engine, Agent Brain, and Render Bridge share the same contracts.
- Implementations remain hardware- and engine-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import json


# ---------------------------------------------------------------------------
# Perception Hub outputs
# ---------------------------------------------------------------------------


@dataclass
class SemanticLabel:
    """Object identity from VLM (e.g. 'a small white cat'). Used as seed for shadow behavior."""

    description: str
    raw_response: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"description": self.description, "raw_response": self.raw_response, "extra": self.extra},
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, s: str) -> "SemanticLabel":
        d = json.loads(s)
        return cls(
            description=d.get("description", ""),
            raw_response=d.get("raw_response"),
            extra=d.get("extra", {}),
        )


@dataclass
class ShadowMaskOutput:
    """2D shadow contour: mask image and/or polygon coordinates."""

    mask_image_path: Optional[str] = None  # path to saved mask (e.g. PNG)
    polygon_xy: Optional[list[list[float]]] = None  # list of [x,y] in image coords
    bounds_xyxy: Optional[tuple[float, float, float, float]] = None  # (xmin, ymin, xmax, ymax)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionOutput:
    """Full output of Perception Hub for one frame or snapshot."""

    semantic: Optional[SemanticLabel] = None
    shadow_mask: Optional[ShadowMaskOutput] = None
    point_cloud_path: Optional[str] = None  # path to 3D point cloud file (e.g. .ply)
    object_visible: bool = True  # False when object removed -> "Severed" / awakening
    frame_timestamp_ms: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic": self.semantic.to_json() if self.semantic else None,
            "shadow_mask": {
                "mask_image_path": self.shadow_mask.mask_image_path if self.shadow_mask else None,
                "polygon_xy": self.shadow_mask.polygon_xy if self.shadow_mask else None,
                "bounds_xyxy": self.shadow_mask.bounds_xyxy if self.shadow_mask else None,
            } if self.shadow_mask else None,
            "point_cloud_path": self.point_cloud_path,
            "object_visible": self.object_visible,
            "frame_timestamp_ms": self.frame_timestamp_ms,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# World Engine I/O
# ---------------------------------------------------------------------------


@dataclass
class SurfacePatch:
    """A patch of surface in global 3D (e.g. table, book) with interaction type."""

    surface_id: str
    role: str  # "ground" | "climbable" | "obstacle" | "target"
    vertices_xyz: list[list[float]]  # list of [x,y,z]
    normal_xyz: Optional[list[float]] = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldState:
    """Unified world state: global 3D coords and interactive surfaces."""

    surfaces: list[SurfacePatch] = field(default_factory=list)
    origin_calibration: Optional[dict[str, Any]] = None  # sensor-to-world transform info
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surfaces": [
                {
                    "surface_id": s.surface_id,
                    "role": s.role,
                    "vertices_xyz": s.vertices_xyz,
                    "normal_xyz": s.normal_xyz,
                    "extra": s.extra,
                }
                for s in self.surfaces
            ],
            "origin_calibration": self.origin_calibration,
            "extra": self.extra,
        }


# ---------------------------------------------------------------------------
# Agent Brain output -> Render Bridge input
# ---------------------------------------------------------------------------


@dataclass
class ActionIntent:
    """Structured action from Agent Brain (e.g. climb, move_to)."""

    action: str  # "idle" | "walk" | "climb" | "crouch" | "awaken" | "react" | ...
    target_coord: Optional[list[float]] = None  # [x, y, z] in world coords
    duration_sec: Optional[float] = None
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "target_coord": self.target_coord,
            "duration_sec": self.duration_sec,
            "params": self.params,
        }


# ---------------------------------------------------------------------------
# Render Bridge: command to engine (OSC/WebSocket payload)
# ---------------------------------------------------------------------------


@dataclass
class RenderCommand:
    """Command sent to rendering engine (UE/Unity): trigger animation or pose."""

    command_type: str  # "animation" | "pose" | "position" | "event"
    name: str  # e.g. "stand", "walk", "climb", "awaken"
    value: Optional[float] = None
    position_xyz: Optional[list[float]] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_osc_args(self) -> list:
        """Flatten for OSC message (e.g. /peter_pan/animation stand 1.0)."""
        args = [self.name]
        if self.value is not None:
            args.append(float(self.value))
        if self.position_xyz:
            args.extend(float(x) for x in self.position_xyz)
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_type": self.command_type,
            "name": self.name,
            "value": self.value,
            "position_xyz": self.position_xyz,
            "extra": self.extra,
        }
