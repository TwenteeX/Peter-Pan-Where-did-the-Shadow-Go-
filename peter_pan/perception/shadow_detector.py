"""
Shadow contour recognition (Sprint PRD Goal A).
OpenCV: Lab/HSV luminance, background subtraction, morphology, contours, polygon + JSON.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None  # type: ignore

from peter_pan.geometry import (
    bbox_from_points,
    image_point_to_table,
    image_polygon_to_table,
    normalize_table_roi,
)
from peter_pan.config import load_config


@dataclass
class ShadowDetectorConfig:
    """Tunable shadow detection (defaults align with config.yaml)."""

    table_roi: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]]
    )
    table_boundary: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]]
    )
    color_space: str = "lab"  # lab | hsv | gray
    darkness_threshold: int = 18  # bg L - cur L > threshold
    min_area_px: int = 800
    max_regions: int = 3
    morph_open_kernel: int = 3
    morph_close_kernel: int = 7
    poly_epsilon_ratio: float = 0.002  # approxPolyDP epsilon vs perimeter
    new_event_centroid_px: float = 55.0  # new blob if no match within this distance
    new_event_area_ratio: float = 0.35  # sudden area change triggers event


class ShadowDetector:
    """
    FR-A1..A6: ROI, segmentation, contours, noise, events, JSON.
    Coordinates in output JSON for agent: table space (polygon, bbox, centroid).
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        detector_cfg: Optional[ShadowDetectorConfig] = None,
    ):
        if cv2 is None:
            raise RuntimeError("OpenCV required: pip install opencv-python")
        self._yaml = config or load_config()
        sd = self._yaml.get("shadow_detector", {})
        self.cfg = detector_cfg or ShadowDetectorConfig(
            table_roi=sd.get("table_roi", ShadowDetectorConfig().table_roi),
            table_boundary=sd.get("table_boundary", ShadowDetectorConfig().table_boundary),
            color_space=sd.get("color_space", "lab"),
            darkness_threshold=int(sd.get("darkness_threshold", 18)),
            min_area_px=int(sd.get("min_area_px", 800)),
            max_regions=int(sd.get("max_regions", 3)),
            morph_open_kernel=int(sd.get("morph_open_kernel", 3)),
            morph_close_kernel=int(sd.get("morph_close_kernel", 7)),
            poly_epsilon_ratio=float(sd.get("poly_epsilon_ratio", 0.002)),
            new_event_centroid_px=float(sd.get("new_event_centroid_px", 55)),
            new_event_area_ratio=float(sd.get("new_event_area_ratio", 0.35)),
        )
        self._background_l: Optional[np.ndarray] = None
        self._prev_centroids_table: list[tuple[float, float]] = []
        self._prev_areas: list[float] = []

    def set_background(self, frame_bgr: np.ndarray) -> None:
        """Capture background with no shadow / arm (FR-A1)."""
        self._background_l = self._luminance(frame_bgr)

    def load_background(self, path: str | Path) -> None:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        self.set_background(img)

    def _luminance(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self.cfg.color_space == "gray":
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self.cfg.color_space == "hsv":
            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            return hsv[:, :, 2]
        # lab L channel
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        return lab[:, :, 0]

    def _roi_mask(self, shape: tuple[int, ...]) -> np.ndarray:
        h, w = shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        roi = self.cfg.table_roi
        pts = np.array([[[int(p[0]), int(p[1])]] for p in roi], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def detect(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        """
        Run detection on one BGR frame. Returns PRD §5.3 JSON structure (shadow_regions in table coords).
        """
        ts = time.time()
        if self._background_l is None:
            raise RuntimeError(
                "Background not set. Call set_background() or load_background() with a clean tabletop (no shadow)."
            )
        cur_l = self._luminance(frame_bgr)
        bg = self._background_l
        if bg.shape != cur_l.shape:
            raise RuntimeError("Frame size does not match background; recapture background.")

        # Shadow = darker than background (FR-A2)
        diff = bg.astype(np.int16) - cur_l.astype(np.int16)
        _, dark = cv2.threshold(
            diff.astype(np.float32),
            self.cfg.darkness_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        dark_u8 = np.clip(dark, 0, 255).astype(np.uint8)

        roi_mask = self._roi_mask(frame_bgr.shape)
        dark_u8 = cv2.bitwise_and(dark_u8, dark_u8, mask=roi_mask)

        # FR-A4 morphology
        k1 = max(1, self.cfg.morph_open_kernel | 1)
        k2 = max(1, self.cfg.morph_close_kernel | 1)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k1, k1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k2, k2))
        dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_OPEN, kernel1)
        dark_u8 = cv2.morphologyEx(dark_u8, cv2.MORPH_CLOSE, kernel2)

        contours, _ = cv2.findContours(dark_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.cfg.min_area_px:
                continue
            candidates.append(c)
        candidates.sort(key=cv2.contourArea, reverse=True)
        candidates = candidates[: self.cfg.max_regions]

        shadow_regions = []
        table_roi = self.cfg.table_roi
        table_boundary = self.cfg.table_boundary

        new_centroids_table: list[tuple[float, float]] = []
        new_areas: list[float] = []

        for idx, cnt in enumerate(candidates):
            area_px = float(cv2.contourArea(cnt))
            peri = cv2.arcLength(cnt, True)
            eps = max(1.0, self.cfg.poly_epsilon_ratio * peri)
            approx = cv2.approxPolyDP(cnt, eps, True)
            poly_img = [[float(p[0][0]), float(p[0][1])] for p in approx]
            if len(poly_img) < 3:
                hull = cv2.convexHull(cnt)
                poly_img = [[float(p[0][0]), float(p[0][1])] for p in hull]

            poly_table = image_polygon_to_table(poly_img, table_roi, table_boundary)
            bx, by, bw, bh = bbox_from_points(poly_table)
            m = cv2.moments(cnt)
            if m["m00"] > 1e-6:
                cx_i = m["m10"] / m["m00"]
                cy_i = m["m01"] / m["m00"]
            else:
                cx_i = sum(p[0] for p in poly_img) / len(poly_img)
                cy_i = sum(p[1] for p in poly_img) / len(poly_img)
            cx_t, cy_t = image_point_to_table(cx_i, cy_i, table_roi, table_boundary)
            new_centroids_table.append((cx_t, cy_t))
            new_areas.append(area_px)

            conf = min(0.99, 0.55 + 0.45 * min(1.0, area_px / (self.cfg.min_area_px * 8)))
            is_new = self._is_new_event((cx_t, cy_t), area_px, idx)

            shadow_regions.append(
                {
                    "id": f"shadow_{idx + 1}",
                    "polygon": poly_table,
                    "bbox": [bx, by, bw, bh],
                    "centroid": [cx_t, cy_t],
                    "area_px": int(area_px),
                    "confidence": round(conf, 2),
                    "is_new_event": is_new,
                }
            )

        self._prev_centroids_table = new_centroids_table
        self._prev_areas = new_areas

        out = {
            "timestamp": ts,
            "table_roi": table_roi,
            "shadow_regions": shadow_regions,
            "_debug": {
                "mask": dark_u8,
                "contours_image": candidates,
            },
        }
        return out

    def _is_new_event(self, centroid_table: tuple[float, float], area_px: float, idx: int) -> bool:
        """FR-A5: new valid shadow or significant change."""
        if not self._prev_centroids_table:
            return True
        matched = False
        for i, prev in enumerate(self._prev_centroids_table):
            d = np.hypot(centroid_table[0] - prev[0], centroid_table[1] - prev[1])
            if d < self.cfg.new_event_centroid_px:
                matched = True
                if i < len(self._prev_areas) and self._prev_areas[i] > 1:
                    ar = abs(area_px - self._prev_areas[i]) / self._prev_areas[i]
                    if ar > self.cfg.new_event_area_ratio:
                        return True
                break
        if not matched:
            return True
        return False

    @staticmethod
    def to_json(data: dict[str, Any], include_debug: bool = False) -> str:
        """Serialize; _debug holds numpy arrays and is never written to JSON."""
        del include_debug
        clean = {k: v for k, v in data.items() if k != "_debug"}
        return json.dumps(clean, ensure_ascii=False, indent=2)

    def save_json(self, data: dict[str, Any], path: str | Path) -> None:
        Path(path).write_text(self.to_json(data), encoding="utf-8")
