"""
Perception Hub — capture geometry and semantics, extract shadow contour.
Input: RGB stream, optional depth.
Output: 2D shadow mask, semantic labels (JSON), optional 3D point cloud.
"""

from .interface import PerceptionHubInterface, run_perception_loop
from .shadow_detector import ShadowDetector, ShadowDetectorConfig
from .environment_builder import (
    build_environment,
    save_environment,
    load_environment,
    EnvironmentBuilderDefaults,
)

__all__ = [
    "PerceptionHubInterface",
    "run_perception_loop",
    "ShadowDetector",
    "ShadowDetectorConfig",
    "build_environment",
    "save_environment",
    "load_environment",
    "EnvironmentBuilderDefaults",
]
