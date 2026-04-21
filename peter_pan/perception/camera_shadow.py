"""
Minimal Perception Hub implementation: OpenCV camera + background subtraction for shadow.
MVP Phase 2: image capture, background subtract, optional VLM call.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    import numpy as np

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
from .detector_sam2 import DetectorSAM2Refiner
from .detector_yolo import DetectorYOLO
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
        self._vlm_lock = threading.Lock()
        self._vlm_last_schedule = 0.0
        self._vlm_inflight = False
        self._last_bounds_xyxy: Optional[tuple[float, float, float, float]] = None
        self._detector_refiner: Optional[DetectorSAM2Refiner] = None
        self._detector_yolo: Optional[DetectorYOLO] = None
        self._vlm_semantics_by_track: Dict[int, SemanticLabel] = {}
        self._sam2_pending = False

    def start_stream(self) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required: pip install opencv-python")
        perception_cfg = self.config.get("perception", {})
        idx = perception_cfg.get("camera_index", 0)
        self._cap = cv2.VideoCapture(int(idx))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {idx}")
        width = int(perception_cfg.get("image_width", 0) or 0)
        height = int(perception_cfg.get("image_height", 0) or 0)
        if width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # Capture initial background (no object)
        _, frame = self._cap.read()
        if frame is not None:
            self._bg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def stop_stream(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._bg = None

    def process_frame_bgr(
        self, frame_bgr: "np.ndarray"
    ) -> Tuple[PerceptionOutput, Optional["np.ndarray"], Optional["np.ndarray"]]:
        """
        Run perception on one BGR frame. Returns (output, diff_gray, mask_gray) for debugging;
        diff/mask are None when background is not set.
        """
        ts = time.time() * 1000
        shadow_cfg = self.config.get("perception", {}).get("shadow", {})
        method = str(shadow_cfg.get("method", "background_subtraction")).lower()
        if method == "detector_sam2":
            (
                shadow_mask,
                object_visible,
                diff_vis,
                mask_vis,
                bounds_xyxy,
            ) = self._process_detector_sam2(frame_bgr)
        elif method == "detector_yolo":
            (
                shadow_mask,
                object_visible,
                diff_vis,
                mask_vis,
                bounds_xyxy,
            ) = self._process_detector_yolo(frame_bgr)
        else:
            (
                shadow_mask,
                object_visible,
                diff_vis,
                mask_vis,
                bounds_xyxy,
            ) = self._process_background_subtraction(frame_bgr)

        with self._vlm_lock:
            sem = self._last_semantic
        out = PerceptionOutput(
            semantic=sem,
            shadow_mask=shadow_mask,
            object_visible=object_visible,
            frame_timestamp_ms=ts,
            extra={"frame_shape": list(frame_bgr.shape)},
        )
        self._last_bounds_xyxy = bounds_xyxy
        self._maybe_run_vlm_thread(frame_bgr, object_visible, bounds_xyxy)
        return out, diff_vis, mask_vis

    def _process_background_subtraction(
        self, frame_bgr: "np.ndarray"
    ) -> tuple[
        Optional[ShadowMaskOutput],
        bool,
        Optional["np.ndarray"],
        Optional["np.ndarray"],
        Optional[tuple[float, float, float, float]],
    ]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        object_visible = True
        diff_vis: Optional["np.ndarray"] = None
        mask_vis: Optional["np.ndarray"] = None
        bounds_xyxy: Optional[tuple[float, float, float, float]] = None
        shadow_mask: Optional[ShadowMaskOutput] = None
        if self._bg is not None:
            cfg = self.config.get("perception", {}).get("shadow", {})
            thresh = cfg.get("threshold", 30)
            diff = cv2.absdiff(self._bg, gray)
            diff_vis = diff
            _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
            mask_vis = mask
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            polygon_xy = None
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest) > 100:
                    polygon_xy = [[float(p[0][0]), float(p[0][1])] for p in largest]
                    x, y, w, h = cv2.boundingRect(largest)
                    bounds_xyxy = (float(x), float(y), float(x + w), float(y + h))
                else:
                    object_visible = False
            else:
                object_visible = False
            shadow_mask = ShadowMaskOutput(
                polygon_xy=polygon_xy,
                bounds_xyxy=bounds_xyxy,
                extra={"contour_count": len(contours), "method": "background_subtraction"},
            )
        return shadow_mask, object_visible, diff_vis, mask_vis, bounds_xyxy

    def _process_detector_sam2(
        self, frame_bgr: "np.ndarray"
    ) -> tuple[
        Optional[ShadowMaskOutput],
        bool,
        Optional["np.ndarray"],
        Optional["np.ndarray"],
        Optional[tuple[float, float, float, float]],
    ]:
        if self._detector_refiner is None:
            self._detector_refiner = DetectorSAM2Refiner(
                self.config.get("perception", {}).get("shadow", {})
            )
        run_sam = False
        if self._sam2_pending:
            self._sam2_pending = False
            run_sam = True
        result = self._detector_refiner.predict(frame_bgr, run_sam=run_sam)
        return (
            result["shadow_mask"],
            bool(result["object_visible"]),
            None,
            result.get("mask_vis"),
            result.get("bounds_xyxy"),
        )

    def _process_detector_yolo(
        self, frame_bgr: "np.ndarray"
    ) -> tuple[
        Optional[ShadowMaskOutput],
        bool,
        Optional["np.ndarray"],
        Optional["np.ndarray"],
        Optional[tuple[float, float, float, float]],
    ]:
        if self._detector_yolo is None:
            self._detector_yolo = DetectorYOLO(
                self.config.get("perception", {}).get("shadow", {})
            )
        result = self._detector_yolo.predict(frame_bgr)
        return (
            result["shadow_mask"],
            bool(result["object_visible"]),
            None,
            result.get("mask_vis"),
            result.get("bounds_xyxy"),
        )

    def _maybe_run_vlm_thread(
        self,
        frame_bgr: "np.ndarray",
        object_visible: bool,
        bounds_xyxy: Optional[tuple[float, float, float, float]],
    ) -> None:
        vlm = self.config.get("perception", {}).get("vlm", {})
        if not vlm.get("enabled", False):
            return
        if str(vlm.get("trigger_mode", "auto")).lower() != "auto":
            return
        if str(vlm.get("provider", "openai")).lower() != "openai":
            return
        if vlm.get("only_when_object_visible", True) and not object_visible:
            return
        interval = float(vlm.get("interval_sec", 3.0))
        now = time.time()
        if now - self._vlm_last_schedule < interval:
            return
        self.request_vlm_once(frame_bgr, bounds_xyxy=bounds_xyxy)

    def request_vlm_once(
        self,
        frame_bgr: "np.ndarray",
        *,
        bounds_xyxy: Optional[tuple[float, float, float, float]] = None,
    ) -> bool:
        """Manually trigger one VLM request. Returns False when skipped."""
        vlm = self.config.get("perception", {}).get("vlm", {})
        if not vlm.get("enabled", False):
            return False
        if str(vlm.get("provider", "openai")).lower() != "openai":
            return False
        if self._vlm_inflight:
            return False

        self._vlm_last_schedule = time.time()
        self._vlm_inflight = True
        frame_copy = frame_bgr.copy()
        bounds_copy = bounds_xyxy if bounds_xyxy is not None else self._last_bounds_xyxy

        def worker() -> None:
            try:
                from peter_pan.perception.vlm_openai import describe_frame_openai

                label = describe_frame_openai(
                    frame_copy, self.config, bounds_xyxy=bounds_copy
                )
                with self._vlm_lock:
                    self._last_semantic = label
            except Exception as e:  # noqa: BLE001 — surface API errors as semantic
                err = SemanticLabel(
                    description="",
                    raw_response=str(e),
                    extra={"vlm_error": True},
                )
                with self._vlm_lock:
                    self._last_semantic = err
            finally:
                self._vlm_inflight = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    def get_vlm_semantic_for_track(self, track_id: int) -> Optional[SemanticLabel]:
        with self._vlm_lock:
            return self._vlm_semantics_by_track.get(track_id)

    def snapshot_vlm_semantics_by_track(self) -> Dict[int, SemanticLabel]:
        with self._vlm_lock:
            return dict(self._vlm_semantics_by_track)

    def request_sam2_once(self) -> None:
        """Next frame with method detector_sam2 will run SAM2 once (if ready)."""
        self._sam2_pending = True

    def request_vlm_for_all_tracked_objects(
        self,
        frame_bgr: "np.ndarray",
        detections: List[Dict[str, Any]],
    ) -> bool:
        """
        One VLM API call per detection with valid track_id (cropped ROI).
        Stores results in _vlm_semantics_by_track[track_id].
        """
        vlm = self.config.get("perception", {}).get("vlm", {})
        if not vlm.get("enabled", False):
            return False
        if str(vlm.get("provider", "openai")).lower() != "openai":
            return False
        if self._vlm_inflight:
            return False

        max_n = int(vlm.get("max_objects_per_batch", 10))
        items: List[Tuple[int, tuple]] = []
        for det in detections:
            tid = int(det.get("track_id", -1))
            if tid < 0:
                continue
            b = det.get("bounds_xyxy")
            if not b or len(b) != 4:
                continue
            items.append((tid, (float(b[0]), float(b[1]), float(b[2]), float(b[3]))))
        items = items[:max_n]

        if not items:
            return self.request_vlm_once(
                frame_bgr,
                bounds_xyxy=(
                    detections[0].get("bounds_xyxy")
                    if detections
                    else self._last_bounds_xyxy
                ),
            )

        self._vlm_last_schedule = time.time()
        self._vlm_inflight = True
        frame_copy = frame_bgr.copy()
        items_copy = list(items)

        def worker() -> None:
            try:
                from peter_pan.perception.vlm_openai import describe_frame_openai

                last_label: Optional[SemanticLabel] = None
                for tid, bounds in items_copy:
                    crop = self._crop_bgr(frame_copy, bounds)
                    if crop is None:
                        continue
                    try:
                        label = describe_frame_openai(
                            crop, self.config, bounds_xyxy=None
                        )
                    except Exception as e:  # noqa: BLE001
                        label = SemanticLabel(
                            description="",
                            raw_response=str(e),
                            extra={"vlm_error": True, "track_id": tid},
                        )
                    with self._vlm_lock:
                        self._vlm_semantics_by_track[tid] = label
                        last_label = label
                if last_label is not None:
                    with self._vlm_lock:
                        self._last_semantic = last_label
            finally:
                self._vlm_inflight = False

        threading.Thread(target=worker, daemon=True).start()
        return True

    @staticmethod
    def _crop_bgr(
        frame_bgr: "np.ndarray", bounds_xyxy: tuple[float, float, float, float]
    ) -> Optional["np.ndarray"]:
        if cv2 is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = (
            int(bounds_xyxy[0]),
            int(bounds_xyxy[1]),
            int(bounds_xyxy[2]),
            int(bounds_xyxy[3]),
        )
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return frame_bgr[y1:y2, x1:x2].copy()

    def capture_frame(self) -> Optional[PerceptionOutput]:
        if self._cap is None or not self._cap.isOpened():
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        out, _, _ = self.process_frame_bgr(frame)
        return out

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
        with self._vlm_lock:
            self._last_semantic = label
