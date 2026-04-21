"""
Image pixels -> plane coordinates (2D).

- If homography enabled in config: cv2.getPerspectiveTransform from 4 image points to 4 plane points.
- Else: plane = normalized image coordinates (u,v) in [0,1]² (no physical units).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


class HomographyPlane:
    """Maps camera image (x,y) pixels to plane coordinates (X, Y)."""

    def __init__(self, world_cfg: dict[str, Any]):
        self._H: Optional[np.ndarray] = None
        self._use_uv_fallback = True
        pcfg = world_cfg.get("plane_space", {}) or {}
        hcfg = pcfg.get("homography", {}) or {}
        if not bool(hcfg.get("enabled", False)):
            return
        if cv2 is None:
            return
        img_pts = hcfg.get("image_points")
        plane_pts = hcfg.get("plane_points")
        if not img_pts or not plane_pts or len(img_pts) != 4 or len(plane_pts) != 4:
            return
        src = np.array(img_pts, dtype=np.float32)
        dst = np.array(plane_pts, dtype=np.float32)
        self._H = cv2.getPerspectiveTransform(src, dst)
        self._use_uv_fallback = False

    def pixel_to_plane(
        self, x: float, y: float, frame_wh: Tuple[int, int]
    ) -> Tuple[float, float]:
        """Single image pixel -> plane (X, Y). frame_wh = (w, h) for UV fallback."""
        w, h = int(frame_wh[0]), int(frame_wh[1])
        if w <= 0 or h <= 0:
            return 0.0, 0.0
        if self._H is None or self._use_uv_fallback:
            return float(x) / w, float(y) / h
        if cv2 is None:
            return float(x) / w, float(y) / h
        pt = np.array([[[float(x), float(y)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def polygon_pixels_to_plane(
        self,
        polygon_xy: List[List[float]],
        frame_wh: Tuple[int, int],
    ) -> List[List[float]]:
        fw, fh = int(frame_wh[0]), int(frame_wh[1])
        out: List[List[float]] = []
        for pt in polygon_xy:
            if len(pt) < 2:
                continue
            x, y = float(pt[0]), float(pt[1])
            px, py = self.pixel_to_plane(x, y, (fw, fh))
            out.append([px, py])
        return out
