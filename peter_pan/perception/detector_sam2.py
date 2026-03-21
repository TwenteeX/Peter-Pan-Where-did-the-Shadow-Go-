"""
YOLO detector + SAM2 mask refinement for object contour extraction.

This module is optional and only used when:
  config.perception.shadow.method == "detector_sam2"
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from peter_pan.protocols import ShadowMaskOutput


class DetectorSAM2Refiner:
    def __init__(self, shadow_cfg: dict[str, Any]):
        self.cfg = shadow_cfg
        self._yolo = None
        self._sam2_predictor = None
        self._ready = False
        self._init_error: Optional[str] = None
        self._init_models()

    def _init_models(self) -> None:
        if cv2 is None:
            self._init_error = "OpenCV is required."
            return
        try:
            from ultralytics import YOLO
        except Exception as e:  # noqa: BLE001
            self._init_error = f"ultralytics import failed: {e}"
            return

        det_cfg = self.cfg.get("detector", {})
        yolo_model = det_cfg.get("model", "yolov8n.pt")
        try:
            self._yolo = YOLO(yolo_model)
        except Exception as e:  # noqa: BLE001
            self._init_error = f"YOLO init failed for model={yolo_model}: {e}"
            return

        sam2_cfg = self.cfg.get("sam2", {})
        model_cfg = sam2_cfg.get("model_cfg")
        checkpoint = sam2_cfg.get("checkpoint")
        if not model_cfg or not checkpoint:
            self._init_error = (
                "SAM2 config missing: set shadow.sam2.model_cfg and shadow.sam2.checkpoint."
            )
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:  # noqa: BLE001
            self._init_error = f"SAM2 import failed: {e}"
            return

        device = sam2_cfg.get("device", "cpu")
        try:
            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:  # noqa: BLE001
            self._init_error = f"SAM2 init failed: {e}"
            return
        self._ready = True

    def predict(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if not self._ready:
            return {
                "shadow_mask": ShadowMaskOutput(
                    polygon_xy=None,
                    bounds_xyxy=None,
                    extra={"method": "detector_sam2", "error": self._init_error},
                ),
                "object_visible": False,
                "mask_vis": None,
                "bounds_xyxy": None,
            }

        det_cfg = self.cfg.get("detector", {})
        min_conf = float(det_cfg.get("min_conf", 0.25))
        min_area = float(self.cfg.get("min_area", 100.0))

        yolo_out = self._yolo.predict(frame_bgr, verbose=False)
        if not yolo_out:
            return self._empty("no_yolo_output")
        boxes = yolo_out[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return self._empty("no_detection")

        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        xyxy = boxes.xyxy.detach().cpu().numpy()
        if confs is not None:
            valid = np.where(confs >= min_conf)[0]
            if valid.size == 0:
                return self._empty("no_detection_above_conf")
            best_idx = int(valid[np.argmax(confs[valid])])
            best_conf = float(confs[best_idx])
        else:
            best_idx = 0
            best_conf = -1.0
        box = xyxy[best_idx]
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        bbox_xyxy = (x1, y1, x2, y2)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._sam2_predictor.set_image(rgb)
        masks, scores, _ = self._sam2_predictor.predict(
            box=np.array([x1, y1, x2, y2], dtype=np.float32),
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return self._empty("sam2_no_mask", bbox_xyxy=bbox_xyxy, det_conf=best_conf)

        score_idx = int(np.argmax(scores))
        chosen_mask = masks[score_idx].astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            chosen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return self._empty(
                "sam2_mask_no_contour", bbox_xyxy=bbox_xyxy, det_conf=best_conf
            )
        largest = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(largest))
        if area < min_area:
            return self._empty(
                "sam2_mask_too_small", bbox_xyxy=bbox_xyxy, det_conf=best_conf
            )

        polygon_xy = [[float(p[0][0]), float(p[0][1])] for p in largest]
        x, y, w, h = cv2.boundingRect(largest)
        bounds_xyxy = (float(x), float(y), float(x + w), float(y + h))
        shadow = ShadowMaskOutput(
            polygon_xy=polygon_xy,
            bounds_xyxy=bounds_xyxy,
            extra={
                "method": "detector_sam2",
                "detector_conf": best_conf,
                "sam2_score": float(scores[score_idx]),
                "contour_count": len(contours),
                "mask_area": area,
            },
        )
        return {
            "shadow_mask": shadow,
            "object_visible": True,
            "mask_vis": chosen_mask,
            "bounds_xyxy": bounds_xyxy,
        }

    def _empty(
        self,
        reason: str,
        *,
        bbox_xyxy: Optional[tuple[float, float, float, float]] = None,
        det_conf: Optional[float] = None,
    ) -> dict[str, Any]:
        extra: dict[str, Any] = {"method": "detector_sam2", "reason": reason}
        if det_conf is not None:
            extra["detector_conf"] = det_conf
        return {
            "shadow_mask": ShadowMaskOutput(
                polygon_xy=None, bounds_xyxy=bbox_xyxy, extra=extra
            ),
            "object_visible": False,
            "mask_vis": None,
            "bounds_xyxy": bbox_xyxy,
        }
