"""Render agent polygon + environment obstacles in plane coordinates (debug / no projector)."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def render_plane_space_preview(
    polygon_plane: List[List[float]],
    canvas_wh: Tuple[int, int] = (960, 720),
    *,
    margin_ratio: float = 0.12,
    view_mode: str = "full",
    fixed_bounds: Optional[Tuple[float, float, float, float]] = None,
    show_grid: bool = True,
    obstacles: Optional[List[Dict[str, Any]]] = None,
    character_label: Optional[str] = None,
    shadow_action: Optional[str] = None,
    animate_shadow: bool = False,
    fill_shadow: bool = False,
    facing: float = 1.0,
    shadow_sprite: Optional["np.ndarray"] = None,
) -> "np.ndarray":
    """
    Draw polygon + obstacles in plane space.

    ``obstacles``: list of dicts, each with:
      - ``label`` (str): class name
      - ``center_uv`` ([u, v]): center in plane UV [0-1]
      - ``rect_uv`` ([u1, v1, u2, v2]): bounding rect in plane UV (optional)

    ``character_label``: VLM-identified character name shown on canvas.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    cw, ch = int(canvas_wh[0]), int(canvas_wh[1])
    img = np.zeros((ch, cw, 3), dtype=np.uint8)
    img[:] = (32, 32, 36)

    title = "plane_space  [c]recapture  [1-8]action  WASD/IJKL=move  q=quit"
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    if shadow_action:
        cv2.putText(
            img,
            f"action: {shadow_action}",
            (10, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (120, 240, 180),
            1,
            cv2.LINE_AA,
        )

    if character_label:
        cv2.putText(
            img,
            f"character: {character_label}",
            (10, ch - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 220, 255),
            1,
            cv2.LINE_AA,
        )

    if view_mode == "full":
        if fixed_bounds is not None:
            min_x, max_x, min_y, max_y = fixed_bounds
        else:
            min_x, max_x, min_y, max_y = 0.0, 1.0, 0.0, 1.0
        span_x = max(max_x - min_x, 1e-9)
        span_y = max(max_y - min_y, 1e-9)

        def to_px(X: float, Y: float) -> Tuple[int, int]:
            u = (float(X) - min_x) / span_x
            v = (float(Y) - min_y) / span_y
            px = int(round(max(0, min(1, u)) * (cw - 1)))
            py = int(round(max(0, min(1, v)) * (ch - 1)))
            return px, py

        if show_grid:
            for i in range(11):
                t = i / 10.0
                x0 = int(round(t * (cw - 1)))
                y0 = int(round(t * (ch - 1)))
                cv2.line(img, (x0, 0), (x0, ch - 1), (55, 55, 62), 1, cv2.LINE_AA)
                cv2.line(img, (0, y0), (cw - 1, y0), (55, 55, 62), 1, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (cw - 1, ch - 1), (90, 90, 100), 2, cv2.LINE_AA)
            cv2.putText(img, f"({min_x:.2f},{min_y:.2f})", (8, ch - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1, cv2.LINE_AA)
            cv2.putText(img, f"({max_x:.2f},{max_y:.2f})", (cw - 120, ch - 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1, cv2.LINE_AA)

        _draw_obstacles(img, obstacles, to_px)

        draw_poly = _maybe_animate_polygon(
            polygon_plane,
            shadow_action,
            animate_shadow,
            facing,
        )

        if len(draw_poly) < 3:
            cv2.putText(img, "No polygon -- press [c] to SAM + capture",
                        (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 100), 1, cv2.LINE_AA)
            return img

        pts = [to_px(p[0], p[1]) for p in draw_poly]
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        sprite_drawn = _maybe_draw_sprite(img, shadow_sprite, draw_poly, to_px, facing)
        if fill_shadow and not sprite_drawn:
            overlay = img.copy()
            cv2.fillPoly(overlay, [arr], color=(2, 2, 3), lineType=cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.78, img, 0.22, 0, dst=img)
        cv2.polylines(img, [arr], isClosed=True, color=(100, 220, 255), thickness=2, lineType=cv2.LINE_AA)
        xs = [float(p[0]) for p in draw_poly]
        ys = [float(p[1]) for p in draw_poly]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cpx, cpy = to_px(cx, cy)
        cv2.circle(img, (cpx, cpy), 6, (180, 100, 255), -1, cv2.LINE_AA)
        return img

    # --- tight (legacy) ---
    if len(polygon_plane) < 3:
        cv2.putText(img, "No polygon -- capture with [a] after SAM2",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 100), 1, cv2.LINE_AA)
        return img

    draw_poly = _maybe_animate_polygon(
        polygon_plane,
        shadow_action,
        animate_shadow,
        facing,
    )
    xs = [float(p[0]) for p in draw_poly]
    ys = [float(p[1]) for p in draw_poly]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    rw = max_x - min_x
    rh = max_y - min_y
    pad_x = rw * margin_ratio + 1e-6
    pad_y = rh * margin_ratio + 1e-6
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    def to_px_tight(X: float, Y: float) -> Tuple[int, int]:
        u = (float(X) - min_x) / span_x
        v = (float(Y) - min_y) / span_y
        px = int(round(u * (cw - 1)))
        py = int(round(v * (ch - 1)))
        return px, py

    pts = [to_px_tight(p[0], p[1]) for p in draw_poly]
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    sprite_drawn = _maybe_draw_sprite(img, shadow_sprite, draw_poly, to_px_tight, facing)
    if fill_shadow and not sprite_drawn:
        cv2.fillPoly(img, [arr], color=(2, 2, 3), lineType=cv2.LINE_AA)
    cv2.polylines(img, [arr], isClosed=True, color=(100, 220, 255), thickness=2, lineType=cv2.LINE_AA)
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    cpx, cpy = to_px_tight(cx, cy)
    cv2.circle(img, (cpx, cpy), 5, (180, 100, 255), -1, cv2.LINE_AA)
    return img


def _maybe_animate_polygon(
    polygon_plane: List[List[float]],
    shadow_action: Optional[str],
    animate_shadow: bool,
    facing: float,
) -> List[List[float]]:
    if not animate_shadow or len(polygon_plane) < 3:
        return polygon_plane
    from peter_pan.agent_brain.action_animation import animated_polygon_plane

    return animated_polygon_plane(
        polygon_plane,
        shadow_action or "idle",
        time.time(),
        facing=facing,
    )


def _maybe_draw_sprite(
    img: "np.ndarray",
    shadow_sprite: Optional["np.ndarray"],
    polygon_plane: List[List[float]],
    to_px,
    facing: float,
) -> bool:
    if shadow_sprite is None:
        return False
    from peter_pan.generation.sprite_cache import overlay_sprite_on_canvas

    return overlay_sprite_on_canvas(
        img,
        shadow_sprite,
        polygon_plane,
        to_px,
        flip_x=facing < 0,
    )


def _draw_obstacles(
    img: "np.ndarray",
    obstacles: Optional[List[Dict[str, Any]]],
    to_px,
) -> None:
    """Draw YOLO detections mapped to plane coords as green rects + labels."""
    if not obstacles:
        return
    for obs in obstacles:
        label = str(obs.get("label", "?"))
        rect = obs.get("rect_uv")
        center = obs.get("center_uv")
        if rect and len(rect) == 4:
            p1 = to_px(float(rect[0]), float(rect[1]))
            p2 = to_px(float(rect[2]), float(rect[3]))
            cv2.rectangle(img, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)
            tx, ty = p1[0], max(12, p1[1] - 5)
            cv2.putText(img, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)
        elif center and len(center) >= 2:
            cp = to_px(float(center[0]), float(center[1]))
            cv2.drawMarker(img, cp, (0, 200, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
            cv2.putText(img, label, (cp[0] + 8, cp[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)
