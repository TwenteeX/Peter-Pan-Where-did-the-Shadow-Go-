"""
Ultralytics YOLO detection + optional tracking (ByteTrack or simple IoU).

Used when config.perception.shadow.method == "detector_yolo".
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from peter_pan.protocols import ShadowMaskOutput


def _area_xyxy(b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xyxy(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _area_xyxy(a) + _area_xyxy(b) - inter
    return inter / ua if ua > 0 else 0.0


def _center_dist_xyxy(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    dx, dy = acx - bcx, acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)


class _SimpleIoUTracker:
    """Lightweight IoU / center-distance association when ByteTrack is disabled."""

    def __init__(
        self,
        iou_threshold: float,
        center_dist_px: float,
        max_age_frames: int,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.center_dist_px = center_dist_px
        self.max_age_frames = max_age_frames
        self._next_id = 0
        self._tracks: dict[int, dict[str, Any]] = {}

    def update(self, detections: list[dict[str, Any]]) -> None:
        """Mutates each dict in `detections` to set `track_id`."""
        n = len(detections)
        if n == 0:
            for tid in list(self._tracks.keys()):
                self._tracks[tid]["missed"] = int(self._tracks[tid].get("missed", 0)) + 1
                if self._tracks[tid]["missed"] > self.max_age_frames:
                    del self._tracks[tid]
            return

        used_det: set[int] = set()

        for tid in list(self._tracks.keys()):
            tbox = self._tracks[tid]["box"]
            best_i = -1
            best_metric = -1.0
            for i in range(n):
                if i in used_det:
                    continue
                dbox = detections[i]["bounds_xyxy"]
                iou = _iou_xyxy(tbox, dbox)
                cd = _center_dist_xyxy(tbox, dbox)
                if iou >= self.iou_threshold or cd <= self.center_dist_px:
                    metric = iou if iou >= self.iou_threshold else max(
                        0.0, 1.0 - cd / max(self.center_dist_px, 1e-6)
                    )
                    if metric > best_metric:
                        best_metric = metric
                        best_i = i
            if best_i >= 0:
                self._tracks[tid]["box"] = detections[best_i]["bounds_xyxy"]
                self._tracks[tid]["missed"] = 0
                detections[best_i]["track_id"] = tid
                used_det.add(best_i)
            else:
                self._tracks[tid]["missed"] = int(self._tracks[tid].get("missed", 0)) + 1
                if self._tracks[tid]["missed"] > self.max_age_frames:
                    del self._tracks[tid]

        for i in range(n):
            if i in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = {
                "box": detections[i]["bounds_xyxy"],
                "missed": 0,
            }
            detections[i]["track_id"] = tid


class DetectorYOLO:
    def __init__(self, shadow_cfg: dict[str, Any]):
        self.cfg = shadow_cfg
        self._yolo: Any = None
        self._ready = False
        self._init_error: Optional[str] = None
        self._simple_tracker: Optional[_SimpleIoUTracker] = None
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

        tr_cfg = self.cfg.get("tracker", {})
        if bool(tr_cfg.get("enabled", True)) and str(
            tr_cfg.get("backend", "bytetrack")
        ).lower() in ("simple",):
            self._simple_tracker = _SimpleIoUTracker(
                float(tr_cfg.get("iou_threshold", 0.25)),
                float(tr_cfg.get("center_dist_px", 120.0)),
                int(tr_cfg.get("max_age_frames", 20)),
            )
        self._ready = True

    def _empty(self, reason: str) -> dict[str, Any]:
        return {
            "shadow_mask": ShadowMaskOutput(
                polygon_xy=None,
                bounds_xyxy=None,
                extra={"method": "detector_yolo", "reason": reason, "error": self._init_error},
            ),
            "object_visible": False,
            "mask_vis": None,
            "bounds_xyxy": None,
        }

    def predict(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if not self._ready or self._yolo is None:
            return self._empty("init_failed")

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
            return self._empty("no_yolo_output")
        r0 = results[0]
        boxes = r0.boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return self._empty("no_detection")

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy() if boxes.conf is not None else None
        clss = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None

        names = getattr(r0, "names", None) or getattr(self._yolo, "names", {}) or {}

        track_ids: Optional[np.ndarray] = None
        if use_tracker and backend == "bytetrack" and getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.detach().cpu().numpy().astype(int)

        raw: list[dict[str, Any]] = []
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

        if not raw:
            return self._empty("no_detection_after_filter")

        largest = max(raw, key=lambda d: _area_xyxy(d["bounds_xyxy"]))
        lx1, ly1, lx2, ly2 = largest["bounds_xyxy"]
        bounds_xyxy = (lx1, ly1, lx2, ly2)
        polygon_xy = [
            [lx1, ly1],
            [lx2, ly1],
            [lx2, ly2],
            [lx1, ly2],
        ]

        extra: dict[str, Any] = {
            "method": "detector_yolo",
            "detections": raw,
            "detected_count": len(raw),
            "contour_count": len(raw),
        }
        if backend == "bytetrack":
            extra["tracker_backend"] = "bytetrack"
        elif backend == "simple":
            extra["tracker_backend"] = "simple"
        else:
            extra["tracker_backend"] = "none"

        shadow = ShadowMaskOutput(
            polygon_xy=polygon_xy,
            bounds_xyxy=bounds_xyxy,
            extra=extra,
        )

        vis = frame_bgr.copy()
        for d in raw:
            x1, y1, x2, y2 = [int(v) for v in d["bounds_xyxy"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            tid = int(d.get("track_id", -1))
            label = f"id:{tid} {d['class_name']} {d['conf']:.2f}" if tid >= 0 else f"{d['class_name']} {d['conf']:.2f}"
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

        return {
            "shadow_mask": shadow,
            "object_visible": True,
            "mask_vis": vis,
            "bounds_xyxy": bounds_xyxy,
        }
