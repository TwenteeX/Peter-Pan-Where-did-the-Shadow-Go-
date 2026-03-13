"""
Minimal Perception Hub implementation: OpenCV camera + background subtraction for shadow.
MVP Phase 2: image capture, background subtract, optional VLM call.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None  # type: ignore

from peter_pan.protocols import (
    PerceptionOutput,
    SemanticLabel,
    ShadowMaskOutput,
)
from peter_pan.config import load_config
from .interface import PerceptionHubInterface


class CameraShadowPerception(PerceptionHubInterface):
    """
    Uses one RGB camera. Background subtraction for shadow mask; optional VLM for semantic label.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._cap = None
        self._bg = None
        self._last_semantic: Optional[SemanticLabel] = None

    def start_stream(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required: pip install opencv-python")
        idx = self.config.get("perception", {}).get("camera_index", 0)
        self._cap = cv2.VideoCapture(int(idx))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {idx}")
        # Capture initial background (no object)
        _, frame = self._cap.read()
        if frame is not None:
            self._bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def stop_stream(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._bg = None

    def capture_frame(self) -> Optional[PerceptionOutput]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None

        ts = time.time() * 1000
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        shadow_mask = None
        object_visible = True

        if self._bg is not None:
            cfg = self.config.get("perception", {}).get("shadow", {})
            thresh = cfg.get("threshold", 30)
            diff = cv2.absdiff(self._bg, gray)
            _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
            # Optional: contour -> polygon
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygon_xy = None
            bounds_xyxy = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 100:
                    polygon_xy = [[float(p[0][0]), float(p[0][1])] for p in largest]
                    x, y, w, h = cv2.boundingRect(largest)
                    bounds_xyxy = (float(x), float(y), float(x + w), float(y + h))
                else:
                    object_visible = False
            shadow_mask = ShadowMaskOutput(
                polygon_xy=polygon_xy,
                bounds_xyxy=bounds_xyxy,
                extra={"contour_count": len(contours)},
            )

        return PerceptionOutput(
            semantic=self._last_semantic,
            shadow_mask=shadow_mask,
            object_visible=object_visible,
            frame_timestamp_ms=ts,
            extra={"frame_shape": list(frame.shape)},
        )

    def set_background(self, frame_bgr=None) -> None:
        """Update background from current frame (call when table is empty)."""
        if frame_bgr is not None:
            self._bg = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        elif self._cap is not None:
            _, f = self._cap.read()
            if f is not None:
                self._bg = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    def set_semantic_from_vlm(self, label: SemanticLabel) -> None:
        """After calling VLM, set the semantic seed for subsequent frames."""
        self._last_semantic = label
