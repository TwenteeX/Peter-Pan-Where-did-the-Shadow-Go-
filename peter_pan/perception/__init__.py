"""
Perception Hub — capture geometry and semantics, extract shadow contour.
Input: RGB stream, optional depth.
Output: 2D shadow mask, semantic labels (JSON), optional 3D point cloud.
"""

from .interface import PerceptionHubInterface, run_perception_loop

__all__ = ["PerceptionHubInterface", "run_perception_loop"]
