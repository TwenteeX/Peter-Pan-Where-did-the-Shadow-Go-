"""
Shared I/O contracts and data schemas for all system modules.
Hardware-agnostic: only data flow formats are defined.
"""

from .schemas import (
    SemanticLabel,
    ShadowMaskOutput,
    PerceptionOutput,
    SurfacePatch,
    WorldState,
    ActionIntent,
    RenderCommand,
)

__all__ = [
    "SemanticLabel",
    "ShadowMaskOutput",
    "PerceptionOutput",
    "SurfacePatch",
    "WorldState",
    "ActionIntent",
    "RenderCommand",
]
