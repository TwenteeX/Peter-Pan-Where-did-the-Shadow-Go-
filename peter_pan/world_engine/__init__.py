"""
World Engine — sync physical topology in digital space, unified 3D coordinate system.
Input: 3D mesh, calibration data.
Output: Global 3D coords, interactive/collision surfaces.
"""

from .interface import WorldEngineInterface
from .stub import StubWorldEngine

__all__ = ["WorldEngineInterface", "StubWorldEngine"]
