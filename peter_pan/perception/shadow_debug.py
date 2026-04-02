"""
Debug visualization (NFR-A3): raw, ROI, mask, contour overlay, JSON overlay.
"""

from __future__ import annotations

import json
from typing import Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore

from peter_pan.geometry import normalize_table_roi


def draw_shadow_debug(
    frame_bgr: Any,
    detection: dict[str, Any],
    *,
    reasoning: Optional[dict[str, Any]] = None,
    environment: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Build a 2x2 or strip of panels: original | ROI | mask | contours.
    Returns {"panels": {name: image}, "composite": image}.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")

    dbg = detection.get("_debug", {})
    mask = dbg.get("mask")
    contours = dbg.get("contours_image", [])

    roi = detection.get("table_roi", [[0, 0], [640, 0], [640, 480], [0, 480]])
    rx0, ry0, rx1, ry1 = normalize_table_roi(roi)
    rx0, ry0, rx1, ry1 = int(rx0), int(ry0), int(rx1), int(ry1)

    h, w = frame_bgr.shape[:2]
    roi_crop = frame_bgr[max(0, ry0) : min(h, ry1), max(0, rx0) : min(w, rx1)].copy()

    overlay = frame_bgr.copy()
    for cnt in contours:
        color = (0, 255, 255)
        cv2.drawContours(overlay, [cnt], -1, color, 2)

    if environment:
        ag = environment.get("agent", {})
        pos = ag.get("position")
        tb = environment.get("table_boundary")
        if pos and len(pos) >= 2:
            pt = _table_to_image_xy(float(pos[0]), float(pos[1]), roi, tb)
            if pt:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 14, (255, 100, 0), -1)
        g = environment.get("goal", {}).get("position")
        if g and len(g) >= 2:
            gp = _table_to_image_xy(float(g[0]), float(g[1]), roi, tb)
            if gp:
                cv2.circle(overlay, (int(gp[0]), int(gp[1])), 10, (0, 255, 100), 2)

    for reg in detection.get("shadow_regions", []):
        poly = reg.get("polygon", [])
        if len(poly) < 2:
            continue
        # Map table coords back to image for overlay (inverse of detector mapping)
        # Simpler: draw from contour in debug — already drawn above
        cx, cy = reg.get("centroid", [0, 0])
        tb2 = environment.get("table_boundary") if environment else None
        pt = _table_to_image_xy(cx, cy, roi, tb2)
        if pt:
            cv2.circle(overlay, (int(pt[0]), int(pt[1])), 8, (255, 0, 255), -1)

    if reasoning and reasoning.get("path"):
        path = reasoning["path"]
        for i in range(len(path) - 1):
            p0 = _table_to_image_xy(path[i][0], path[i][1], roi, environment.get("table_boundary") if environment else None)
            p1 = _table_to_image_xy(path[i + 1][0], path[i + 1][1], roi, environment.get("table_boundary") if environment else None)
            if p0 and p1:
                cv2.line(overlay, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

    if mask is not None:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    else:
        mask_bgr = np.zeros(frame_bgr.shape, dtype=np.uint8) if np is not None else frame_bgr.copy()

    json_str = json.dumps(
        {k: v for k, v in detection.items() if k != "_debug"},
        ensure_ascii=False,
        indent=2,
    )[:2000]
    text_panel = np.zeros((200, w, 3), dtype=np.uint8)
    y0 = 20
    for line in json_str.split("\n")[:12]:
        cv2.putText(text_panel, line[:80], (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        y0 += 18

    composite = np.vstack(
        [
            np.hstack([frame_bgr, overlay]),
            np.hstack([mask_bgr, roi_crop if roi_crop.size else mask_bgr]),
        ]
    )
    return {"panels": {"raw": frame_bgr, "overlay": overlay, "mask": mask_bgr}, "composite": composite, "text": text_panel}


def _table_to_image_xy(
    tx: float,
    ty: float,
    table_roi: list[list[float]],
    table_boundary: Optional[list[list[float]]],
) -> Optional[tuple[float, float]]:
    if not table_boundary:
        return None
    tx0 = min(p[0] for p in table_boundary)
    ty0 = min(p[1] for p in table_boundary)
    tx1 = max(p[0] for p in table_boundary)
    ty1 = max(p[1] for p in table_boundary)
    rx0, ry0, rx1, ry1 = normalize_table_roi(table_roi)
    tw, th = tx1 - tx0, ty1 - ty0
    if tw <= 0 or th <= 0:
        return None
    u = (tx - tx0) / tw
    v = (ty - ty0) / th
    ix = rx0 + u * (rx1 - rx0)
    iy = ry0 + v * (ry1 - ry0)
    return ix, iy
