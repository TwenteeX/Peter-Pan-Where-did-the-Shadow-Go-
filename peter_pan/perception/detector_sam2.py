"""
YOLO + SAM2: YOLO every frame; SAM2 only when requested or run_each_frame.

Used when config.perception.shadow.method == "detector_sam2".

- YOLO loads even if SAM2 fails (lazy / on-demand still works).
- shadow.sam2.run_each_frame: true = SAM every frame (heavy); false = SAM only when
  CameraShadowPerception.request_sam2_once() / debug button / key [s].
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from peter_pan.protocols import ShadowMaskOutput

from .detector_yolo import _SimpleIoUTracker, _area_xyxy


def _draw_yolo_overlay(frame_bgr: np.ndarray, detections: List[dict[str, Any]]) -> np.ndarray:
    vis = frame_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = [int(v) for v in d["bounds_xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tid = int(d.get("track_id", -1))
        label = (
            f"id:{tid} {d['class_name']} {d['conf']:.2f}"
            if tid >= 0
            else f"{d['class_name']} {d['conf']:.2f}"
        )
        cv2.putText(
            vis,
            label,
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return vis


def _mask_to_display(mask_u8: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
    if mask_u8.ndim == 3:
        gray = cv2.cvtColor(mask_u8, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_u8
    if gray.shape[0] != frame_h or gray.shape[1] != frame_w:
        gray = cv2.resize(gray, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _mask_binary_fullres(mask: np.ndarray, fh: int, fw: int) -> np.ndarray:
    """SAM mask (any dtype/shape) -> uint8 {0,255} at (fh, fw)."""
    m = np.asarray(mask)
    m = np.squeeze(m)
    if m.ndim > 2:
        m = m[..., 0]
    if m.dtype != np.uint8:
        mf = m.astype(np.float64)
        if mf.size and mf.max() <= 1.0 + 1e-6:
            m = (mf > 0.5).astype(np.uint8) * 255
        else:
            m = (mf > 0).astype(np.uint8) * 255
    else:
        m = (m > 127).astype(np.uint8) * 255
    if m.shape[0] != fh or m.shape[1] != fw:
        m = cv2.resize(m, (fw, fh), interpolation=cv2.INTER_NEAREST)
    return m


def _palette_bgr(n: int) -> List[Tuple[int, int, int]]:
    """Distinct BGR colors for n instance masks."""
    if n <= 0:
        return []
    colors: List[Tuple[int, int, int]] = []
    for i in range(n):
        hue = int(179 * i / max(n, 1)) % 180
        hsv = np.uint8([[[hue, 220, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


class DetectorSAM2Refiner:
    def __init__(self, shadow_cfg: dict[str, Any]):
        self.cfg = shadow_cfg
        self._yolo: Any = None
        self._sam2_predictor: Any = None
        self._simple_tracker: Optional[_SimpleIoUTracker] = None
        self._yolo_ready = False
        self._sam_ready = False
        self._yolo_init_error: Optional[str] = None
        self._sam_init_error: Optional[str] = None
        self._cached_mask_bgr: Optional[np.ndarray] = None
        # After a successful SAM pass, keep contour for plane capture on later frames (lazy SAM).
        self._last_sam_polygon_xy: Optional[List[List[float]]] = None
        self._last_sam_bounds_xyxy: Optional[Tuple[float, float, float, float]] = None
        self._init_models()

    def _init_models(self) -> None:
        if cv2 is None:
            self._yolo_init_error = "OpenCV is required."
            return
        try:
            from ultralytics import YOLO
        except Exception as e:  # noqa: BLE001
            self._yolo_init_error = f"ultralytics import failed: {e}"
            return

        det_cfg = self.cfg.get("detector", {})
        yolo_model = det_cfg.get("model", "yolov8n.pt")
        try:
            self._yolo = YOLO(yolo_model)
        except Exception as e:  # noqa: BLE001
            self._yolo_init_error = f"YOLO init failed for model={yolo_model}: {e}"
            return
        self._yolo_ready = True

        tr_cfg = self.cfg.get("tracker", {})
        if bool(tr_cfg.get("enabled", True)) and str(
            tr_cfg.get("backend", "bytetrack")
        ).lower() in ("simple",):
            self._simple_tracker = _SimpleIoUTracker(
                float(tr_cfg.get("iou_threshold", 0.25)),
                float(tr_cfg.get("center_dist_px", 120.0)),
                int(tr_cfg.get("max_age_frames", 20)),
            )

        sam2_cfg = self.cfg.get("sam2", {})
        model_cfg = sam2_cfg.get("model_cfg")
        checkpoint = sam2_cfg.get("checkpoint")
        if not model_cfg or not checkpoint:
            self._sam_init_error = (
                "SAM2 config missing: set shadow.sam2.model_cfg and shadow.sam2.checkpoint."
            )
            return
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as e:  # noqa: BLE001
            self._sam_init_error = f"SAM2 import failed: {e}"
            return

        device = sam2_cfg.get("device", "cpu")
        try:
            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
        except Exception as e:  # noqa: BLE001
            self._sam_init_error = f"SAM2 init failed: {e}"
            return
        self._sam_ready = True

    def _yolo_detections(self, frame_bgr: np.ndarray) -> Tuple[List[dict[str, Any]], Any]:
        det_cfg = self.cfg.get("detector", {})
        tr_cfg = self.cfg.get("tracker", {})
        min_conf = float(det_cfg.get("min_conf", 0.25))
        max_objects = int(det_cfg.get("max_objects", 10))
        min_area = float(self.cfg.get("min_area", 100.0))
        use_tracker = bool(tr_cfg.get("enabled", True))
        backend = str(tr_cfg.get("backend", "bytetrack")).lower()

        if use_tracker and backend == "bytetrack":
            tracker_yaml = str(tr_cfg.get("bytetrack_cfg", "bytetrack.yaml"))
            results = self._yolo.track(
                frame_bgr,
                persist=True,
                tracker=tracker_yaml,
                verbose=False,
            )
        else:
            results = self._yolo.predict(frame_bgr, verbose=False)

        if not results:
            return [], None
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return [], r0

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
        names = getattr(r0, "names", None) or getattr(self._yolo, "names", {}) or {}

        track_ids: Optional[np.ndarray] = None
        if use_tracker and backend == "bytetrack" and getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.detach().cpu().numpy().astype(int)

        raw: List[dict[str, Any]] = []
        for i in range(len(xyxy)):
            c = float(confs[i]) if confs is not None else 1.0
            if c < min_conf:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy[i].tolist()]
            bb = (x1, y1, x2, y2)
            if _area_xyxy(bb) < min_area:
                continue
            cid = int(clss[i]) if clss is not None else 0
            cname = str(names.get(cid, str(cid)))
            tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1
            raw.append(
                {
                    "bounds_xyxy": bb,
                    "conf": c,
                    "class_id": cid,
                    "class_name": cname,
                    "track_id": tid,
                }
            )

        raw.sort(key=lambda d: -d["conf"])
        raw = raw[:max_objects]

        if use_tracker and backend == "simple" and self._simple_tracker is not None:
            for d in raw:
                d["track_id"] = -1
            self._simple_tracker.update(raw)
        elif not use_tracker:
            for d in raw:
                d["track_id"] = -1

        return raw, r0

    def _yolo_shadow_and_vis(
        self, frame_bgr: np.ndarray, detections: List[dict[str, Any]], *, extra: dict[str, Any]
    ) -> dict[str, Any]:
        largest = max(detections, key=lambda d: _area_xyxy(d["bounds_xyxy"]))
        lx1, ly1, lx2, ly2 = largest["bounds_xyxy"]
        bounds_xyxy = (lx1, ly1, lx2, ly2)
        polygon_xy = [
            [lx1, ly1],
            [lx2, ly1],
            [lx2, ly2],
            [lx1, ly2],
        ]
        shadow = ShadowMaskOutput(
            polygon_xy=polygon_xy,
            bounds_xyxy=bounds_xyxy,
            extra=extra,
        )
        vis = _draw_yolo_overlay(frame_bgr, detections)
        return {
            "shadow_mask": shadow,
            "object_visible": True,
            "mask_vis": vis,
            "bounds_xyxy": bounds_xyxy,
        }

    def predict(self, frame_bgr: np.ndarray, *, run_sam: bool = False) -> dict[str, Any]:
        sam2_cfg = self.cfg.get("sam2", {})
        run_each_frame = bool(sam2_cfg.get("run_each_frame", False))
        do_sam = run_sam or run_each_frame

        if not self._yolo_ready or self._yolo is None:
            return {
                "shadow_mask": ShadowMaskOutput(
                    polygon_xy=None,
                    bounds_xyxy=None,
                    extra={
                        "method": "detector_sam2",
                        "error": self._yolo_init_error or "YOLO not initialized",
                    },
                ),
                "object_visible": False,
                "mask_vis": None,
                "bounds_xyxy": None,
            }

        detections, _r0 = self._yolo_detections(frame_bgr)
        if not detections:
            self._last_sam_polygon_xy = None
            self._last_sam_bounds_xyxy = None
            return {
                "shadow_mask": ShadowMaskOutput(
                    polygon_xy=None,
                    bounds_xyxy=None,
                    extra={"method": "detector_sam2", "reason": "no_detection"},
                ),
                "object_visible": False,
                "mask_vis": self._cached_mask_bgr,
                "bounds_xyxy": None,
            }

        fh, fw = frame_bgr.shape[:2]
        min_area = float(self.cfg.get("min_area", 100.0))

        base_extra: dict[str, Any] = {
            "method": "detector_sam2",
            "detections": detections,
            "detected_count": len(detections),
            "contour_count": len(detections),
        }
        if self._sam_init_error and not self._sam_ready:
            base_extra["sam2_error"] = self._sam_init_error

        if not do_sam:
            base_extra["sam2_skipped"] = True
            base_extra["sam2_note"] = (
                "SAM2 skipped (lazy); use SAM button / [s] or set sam2.run_each_frame."
            )
            mask_vis = self._cached_mask_bgr
            if mask_vis is None or mask_vis.shape[0] != fh or mask_vis.shape[1] != fw:
                mask_vis = _draw_yolo_overlay(frame_bgr, detections)
            out = self._yolo_shadow_and_vis(frame_bgr, detections, extra=base_extra)
            # Plane capture / TD: keep using last SAM contour until the next SAM run.
            if (
                self._last_sam_polygon_xy is not None
                and len(self._last_sam_polygon_xy) >= 3
            ):
                sm = out["shadow_mask"]
                if sm is not None:
                    sm.polygon_xy = [list(p) for p in self._last_sam_polygon_xy]
                    if self._last_sam_bounds_xyxy is not None:
                        sm.bounds_xyxy = self._last_sam_bounds_xyxy
                    ex = dict(sm.extra or {})
                    ex["polygon_source"] = "last_sam_cache"
                    sm.extra = ex
            out["mask_vis"] = mask_vis
            return out

        if not self._sam_ready or self._sam2_predictor is None:
            base_extra["sam2_note"] = "SAM2 unavailable; YOLO-only."
            return self._yolo_shadow_and_vis(frame_bgr, detections, extra=base_extra)

        max_sam = int(
            sam2_cfg.get(
                "max_objects_per_sam_pass",
                self.cfg.get("detector", {}).get("max_objects", 10),
            )
        )
        max_sam = max(1, max_sam)
        dets_for_sam = detections[:max_sam]
        palette = _palette_bgr(len(dets_for_sam))

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._sam2_predictor.set_image(rgb)

        composite = np.zeros((fh, fw, 3), dtype=np.uint8)
        best_largest: Optional[Any] = None
        best_area = 0.0
        best_score = 0.0
        best_conf = 0.0
        applied = 0
        scores_list: List[float] = []

        # SAM per detection; paint low-conf first so overlaps show higher-conf color last.
        paint_order = sorted(
            range(len(dets_for_sam)),
            key=lambda i: float(dets_for_sam[i]["conf"]),
        )

        for i in paint_order:
            det = dets_for_sam[i]
            x1, y1, x2, y2 = det["bounds_xyxy"]
            dconf = float(det["conf"])
            masks, scores, _ = self._sam2_predictor.predict(
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=True,
            )
            if masks is None or len(masks) == 0:
                continue
            score_idx = int(np.argmax(scores))
            bin_full = _mask_binary_fullres(masks[score_idx], fh, fw)
            if not np.any(bin_full > 127):
                continue
            color = palette[i]
            fg = bin_full > 127
            composite[fg] = color
            applied += 1
            scores_list.append(float(scores[score_idx]))

            contours, _ = cv2.findContours(
                bin_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            largest = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(largest))
            if area >= min_area and area > best_area:
                best_area = area
                best_largest = largest
                best_score = float(scores[score_idx])
                best_conf = dconf

        if applied == 0:
            base_extra["reason"] = "sam2_no_mask"
            base_extra["detector_conf"] = float(dets_for_sam[0]["conf"])
            return self._yolo_shadow_and_vis(frame_bgr, detections, extra=base_extra)

        self._cached_mask_bgr = composite.copy()

        if best_largest is None:
            base_extra["reason"] = "sam2_mask_too_small"
            base_extra["sam2_masks_applied"] = applied
            out = self._yolo_shadow_and_vis(frame_bgr, detections, extra=base_extra)
            out["mask_vis"] = composite
            return out

        polygon_xy = [[float(p[0][0]), float(p[0][1])] for p in best_largest]
        x, y, w, h = cv2.boundingRect(best_largest)
        bounds_xyxy = (float(x), float(y), float(x + w), float(y + h))
        self._last_sam_polygon_xy = polygon_xy
        self._last_sam_bounds_xyxy = bounds_xyxy

        sam_extra = {
            **base_extra,
            "detector_conf": best_conf,
            "sam2_score": best_score,
            "mask_area": best_area,
            "sam2_ran": True,
            "sam2_masks_applied": applied,
            "sam2_scores": scores_list,
        }
        shadow = ShadowMaskOutput(
            polygon_xy=polygon_xy,
            bounds_xyxy=bounds_xyxy,
            extra=sam_extra,
        )
        return {
            "shadow_mask": shadow,
            "object_visible": True,
            "mask_vis": composite,
            "bounds_xyxy": bounds_xyxy,
        }
