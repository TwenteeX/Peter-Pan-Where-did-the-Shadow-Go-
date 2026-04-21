"""
SAM polygon → 2D plane → **程式／Agent 在電腦裡平移**（與是否鎖定桌上實體無關）。

- ``capture``: 把當下 mask 輪廓存成 ``polygon_plane_base``。
- 位移來源：``manual_offset_plane``（除錯 IJKL、或 ``translate_by_cm`` / ``set_manual_offset_cm`` 給 Agent）。
- 可選 ``follow_detection``：額外讓輪廓跟偵測框動（少數情境才開）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from peter_pan.protocols import PerceptionOutput
from peter_pan.world_engine.homography_plane import HomographyPlane


@dataclass
class AgentPlaneState:
    """Snapshot of segment polygon in UV + plane (+ optional locked track for follow mode)."""

    locked_track_id: int
    polygon_uv: List[List[float]]
    polygon_plane_base: List[List[float]]
    capture_centroid_uv: Tuple[float, float]
    current_centroid_uv: Tuple[float, float]
    centroid_plane_capture: Tuple[float, float]
    centroid_plane_live: Tuple[float, float]
    manual_offset_plane: Tuple[float, float]
    frame_size_wh: Tuple[int, int] = (0, 0)
    extra: Dict[str, Any] = field(default_factory=dict)


class AgentPlaneBinding:
    """
    擷取後的 mask 輪廓在平面座標裡平移；**預設由 Agent／程式控制位移**，不綁實體。

    ``current_polygon_plane()`` 即送 TD / OSC 的多邊形（含手動／公分偏移）。
    """

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self._state: Optional[AgentPlaneState] = None
        self._config = config or {}
        we = self._config.get("world_engine", {}) or {}
        self._hom = HomographyPlane(we)
        ps = we.get("plane_space", {}) or {}
        self._nudge_step = float(ps.get("nudge_step", 0.02))
        self._follow_detection = bool(ps.get("follow_detection", False))
        self._physical_extent_cm = self._parse_physical_extent_cm(ps)

    def set_config(self, config: dict[str, Any]) -> None:
        """Reload homography if config changes (e.g. hot reload)."""
        self._config = config
        we = self._config.get("world_engine", {}) or {}
        self._hom = HomographyPlane(we)
        ps = we.get("plane_space", {}) or {}
        self._nudge_step = float(ps.get("nudge_step", 0.02))
        self._follow_detection = bool(ps.get("follow_detection", False))
        self._physical_extent_cm = self._parse_physical_extent_cm(ps)

    @staticmethod
    def _parse_physical_extent_cm(ps: dict[str, Any]) -> Tuple[float, float]:
        """平面上一單位跨度對應的桌面寬／深（公分），供 translate_by_cm 換算。"""
        raw = ps.get("physical_extent_cm")
        if isinstance(raw, (list, tuple)) and len(raw) >= 2:
            w = float(raw[0])
            h = float(raw[1])
            if w > 0 and h > 0:
                return (w, h)
        return (60.0, 40.0)

    @property
    def physical_extent_cm(self) -> Tuple[float, float]:
        return self._physical_extent_cm

    @property
    def follow_detection(self) -> bool:
        return self._follow_detection

    @property
    def nudge_step(self) -> float:
        return self._nudge_step

    @property
    def active(self) -> bool:
        return self._state is not None

    @property
    def state(self) -> Optional[AgentPlaneState]:
        return self._state

    def clear(self) -> None:
        self._state = None

    def nudge_manual_plane(self, dX: float, dY: float) -> None:
        """在平面座標裡累加平移（與 plane 單位一致；除錯 IJKL 用）。"""
        if self._state is None:
            return
        mx, my = self._state.manual_offset_plane
        self._state.manual_offset_plane = (mx + dX, my + dY)

    def translate_by_cm(self, dx_cm: float, dy_cm: float) -> None:
        """Agent／程式：沿平面累加位移（公分）。與 YOLO、實體鎖定無關。"""
        pw, ph = self._physical_extent_cm
        self.nudge_manual_plane(dx_cm / pw, dy_cm / ph)

    def set_manual_offset_cm(self, offset_x_cm: float, offset_y_cm: float) -> None:
        """Agent／程式：絕對偏移（公分），從擷取時形狀原點算起。"""
        if self._state is None:
            return
        pw, ph = self._physical_extent_cm
        self._state.manual_offset_plane = (offset_x_cm / pw, offset_y_cm / ph)

    def manual_offset_cm(self) -> Tuple[float, float]:
        """目前平面偏移還原成約略公分（依 physical_extent_cm）。"""
        if self._state is None:
            return (0.0, 0.0)
        pw, ph = self._physical_extent_cm
        mx, my = self._state.manual_offset_plane
        return (mx * pw, my * ph)

    def capture(
        self,
        out: PerceptionOutput,
        frame_shape: Tuple[int, ...],
    ) -> Tuple[bool, str]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if w <= 0 or h <= 0:
            return False, "無效的畫面尺寸"

        sm = out.shadow_mask
        if sm is None or not sm.polygon_xy or len(sm.polygon_xy) < 3:
            return (
                False,
                "沒有 polygon（請確認有偵測；detector_sam2 建議先按 SAM2 取得輪廓）",
            )

        poly = sm.polygon_xy
        poly_uv: List[List[float]] = []
        for pt in poly:
            if len(pt) < 2:
                continue
            poly_uv.append([float(pt[0]) / w, float(pt[1]) / h])
        if len(poly_uv) < 3:
            return False, "polygon 頂點不足"

        xs = [p[0] * w for p in poly_uv]
        ys = [p[1] * h for p in poly_uv]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cuv = (cx / w, cy / h)

        frame_wh = (w, h)
        polygon_plane_base = self._hom.polygon_pixels_to_plane(poly, frame_wh)
        if len(polygon_plane_base) < 3:
            return False, "平面轉換失敗"
        px = [p[0] for p in polygon_plane_base]
        py = [p[1] for p in polygon_plane_base]
        c_plane = (sum(px) / len(px), sum(py) / len(py))

        tid = self._pick_track_id(sm, (cx, cy)) if self._follow_detection else -1

        # One reference centroid in plane so the captured polygon is not pre-shifted.
        self._state = AgentPlaneState(
            locked_track_id=tid,
            polygon_uv=poly_uv,
            polygon_plane_base=polygon_plane_base,
            capture_centroid_uv=cuv,
            current_centroid_uv=cuv,
            centroid_plane_capture=c_plane,
            centroid_plane_live=c_plane,
            manual_offset_plane=(0.0, 0.0),
            frame_size_wh=(w, h),
            extra={"captured_object_visible": out.object_visible},
        )
        follow_hint = (
            f"；偵測跟隨=開（track_id={tid}）"
            if self._follow_detection and tid >= 0
            else "；位移由 Agent：translate_by_cm / set_manual_offset_cm（與實體鎖定無關）"
        )
        return True, (
            f"已儲存 {len(poly_uv)} 頂點；平面質心=({c_plane[0]:.3f},{c_plane[1]:.3f}){follow_hint}"
        )

    def _pick_track_id(self, sm: Any, centroid_xy: Tuple[float, float]) -> int:
        dets = (sm.extra or {}).get("detections") or []
        if not dets:
            return -1
        cx, cy = centroid_xy
        best_tid = -1
        best_area = -1.0
        for d in dets:
            t = int(d.get("track_id", -1))
            b = d.get("bounds_xyxy")
            if not b or len(b) != 4:
                continue
            x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
            if x1 <= cx <= x2 and y1 <= cy <= y2 and t >= 0:
                ar = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if ar > best_area:
                    best_area = ar
                    best_tid = t
        if best_tid >= 0:
            return best_tid
        for d in sorted(dets, key=lambda x: -float(x.get("conf", 0))):
            t = int(d.get("track_id", -1))
            if t >= 0:
                return t
        return -1

    def update_from_perception(
        self,
        out: PerceptionOutput,
        frame_shape: Tuple[int, ...],
    ) -> None:
        if self._state is None or not self._follow_detection:
            return
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if w <= 0 or h <= 0:
            return
        frame_wh = (w, h)

        sm = out.shadow_mask
        dets = ((sm.extra if sm else {}) or {}).get("detections") or []
        tid = self._state.locked_track_id

        if tid >= 0:
            for d in dets:
                if int(d.get("track_id", -1)) != tid:
                    continue
                b = d.get("bounds_xyxy")
                if b and len(b) == 4:
                    x1, y1, x2, y2 = map(float, b)
                    self._state.current_centroid_uv = (
                        (x1 + x2) * 0.5 / w,
                        (y1 + y2) * 0.5 / h,
                    )
                break
        else:
            best = None
            best_ar = 0.0
            for d in dets:
                b = d.get("bounds_xyxy")
                if not b or len(b) != 4:
                    continue
                x1, y1, x2, y2 = map(float, b)
                ar = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if ar > best_ar:
                    best_ar = ar
                    best = (x1, y1, x2, y2)
            if best:
                x1, y1, x2, y2 = best
                self._state.current_centroid_uv = (
                    (x1 + x2) * 0.5 / w,
                    (y1 + y2) * 0.5 / h,
                )

        cu, cv = self._state.current_centroid_uv
        cx_pix = cu * w
        cy_pix = cv * h
        self._state.centroid_plane_live = self._hom.pixel_to_plane(
            cx_pix, cy_pix, frame_wh
        )

    def current_polygon_uv(self) -> List[List[float]]:
        st = self._state
        if st is None:
            return []
        if not self._follow_detection:
            return [list(p) for p in st.polygon_uv]
        du = st.current_centroid_uv[0] - st.capture_centroid_uv[0]
        dv = st.current_centroid_uv[1] - st.capture_centroid_uv[1]
        return [[p[0] + du, p[1] + dv] for p in st.polygon_uv]

    def current_polygon_plane(self) -> List[List[float]]:
        """Closed polygon in plane: optional track rigid shift + manual nudge (TD / OSC source)."""
        st = self._state
        if st is None:
            return []
        if self._follow_detection:
            dX = st.centroid_plane_live[0] - st.centroid_plane_capture[0]
            dY = st.centroid_plane_live[1] - st.centroid_plane_capture[1]
        else:
            dX, dY = 0.0, 0.0
        mx, my = st.manual_offset_plane
        return [
            [p[0] + dX + mx, p[1] + dY + my]
            for p in st.polygon_plane_base
        ]

    def current_polygon_xy_pixels(
        self, frame_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int]]:
        h, w = int(frame_shape[0]), int(frame_shape[1])
        pts: List[Tuple[int, int]] = []
        for u, v in self.current_polygon_uv():
            x = int(round(max(0, min(1, u)) * (w - 1)))
            y = int(round(max(0, min(1, v)) * (h - 1)))
            pts.append((x, y))
        return pts

    def to_plane_dict(self) -> Dict[str, Any]:
        st = self._state
        if st is None:
            return {"active": False}
        ox_cm, oy_cm = self.manual_offset_cm()
        return {
            "active": True,
            "follow_detection": self._follow_detection,
            "physical_extent_cm": list(self._physical_extent_cm),
            "manual_offset_cm": [ox_cm, oy_cm],
            "locked_track_id": st.locked_track_id,
            "capture_centroid_uv": list(st.capture_centroid_uv),
            "current_centroid_uv": list(st.current_centroid_uv),
            "centroid_plane_live": list(st.centroid_plane_live),
            "manual_offset_plane": list(st.manual_offset_plane),
            "polygon_uv": self.current_polygon_uv(),
            "polygon_plane": self.current_polygon_plane(),
            "frame_size_wh": list(st.frame_size_wh),
        }
