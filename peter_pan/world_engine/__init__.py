"""
World Engine — sync physical topology in digital space, unified 3D coordinate system.
Input: 3D mesh, calibration data.
Output: Global 3D coords, interactive/collision surfaces.
"""

from .homography_plane import HomographyPlane
from .interface import WorldEngineInterface
from .stub import StubWorldEngine

__all__ = ["HomographyPlane", "WorldEngineInterface", "StubWorldEngine"]
