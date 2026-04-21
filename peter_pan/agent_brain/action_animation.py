"""Procedural shadow action animation for plane-space previews.

This is the first local renderer layer: it turns one captured polygon into a
small action vocabulary without requiring TouchDesigner or generated assets.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


Point = List[float]


def normalize_action(action: str | None) -> str:
    """Map LLM/freeform actions into the small animation vocabulary."""
    raw = (action or "idle").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "move": "walk",
        "move_to": "walk",
        "walk_right": "walk",
        "walk_left": "walk",
        "run": "walk",
        "approach": "walk",
        "avoid": "flee",
        "escape": "flee",
        "jump": "hop",
        "bounce": "hop",
        "look": "inspect",
        "peek": "inspect",
        "crouch": "hide",
        "duck": "hide",
        "crawl": "hide",
    }
    raw = aliases.get(raw, raw)
    allowed = {"idle", "walk", "hop", "stretch", "hide", "climb", "inspect", "flee", "awaken"}
    return raw if raw in allowed else "idle"


def animated_polygon_plane(
    polygon_plane: Sequence[Sequence[float]],
    action: str | None,
    now_sec: float,
    *,
    facing: float = 1.0,
) -> List[Point]:
    """
    Return an animated copy of ``polygon_plane`` in the same plane coordinates.

    The transforms are intentionally simple and stable: affine squash/stretch,
    bobbing, and shear around the captured polygon centroid.
    """
    if len(polygon_plane) < 3:
        return [list(p) for p in polygon_plane]

    pts = [[float(p[0]), float(p[1])] for p in polygon_plane if len(p) >= 2]
    if len(pts) < 3:
        return pts

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    width = max(max(xs) - min(xs), 1e-6)
    height = max(max(ys) - min(ys), 1e-6)
    action_name = normalize_action(action)
    phase = now_sec * math.tau
    face = -1.0 if facing < 0 else 1.0

    sx = 1.0
    sy = 1.0
    shear = 0.0
    dx = 0.0
    dy = 0.0
    rot = 0.0

    if action_name == "idle":
        sy = 1.0 + 0.025 * math.sin(phase * 0.75)
        sx = 1.0 - 0.015 * math.sin(phase * 0.75)
    elif action_name == "walk":
        step = math.sin(phase * 1.9)
        sx = 1.0 + 0.045 * math.cos(phase * 1.9)
        sy = 1.0 - 0.035 * math.cos(phase * 1.9)
        shear = face * 0.075 * step
        dy = -0.018 * abs(step) * height
    elif action_name == "hop":
        jump = abs(math.sin(phase * 1.15))
        dy = -0.18 * jump * height
        sx = 1.0 + 0.10 * (1.0 - jump)
        sy = 1.0 - 0.12 * (1.0 - jump)
    elif action_name == "stretch":
        s = 0.5 + 0.5 * math.sin(phase * 0.9)
        sx = 0.92 + 0.18 * s
        sy = 0.96 + 0.15 * s
        dy = -0.035 * s * height
    elif action_name == "hide":
        s = 0.5 + 0.5 * math.sin(phase * 1.2)
        sx = 1.10 + 0.06 * s
        sy = 0.48 + 0.08 * s
        dy = 0.22 * height
    elif action_name == "climb":
        step = math.sin(phase * 1.6)
        sy = 1.08 + 0.04 * step
        sx = 0.96 - 0.03 * step
        dy = -0.05 * step * height
        rot = face * 0.10 * math.sin(phase * 0.8)
    elif action_name == "inspect":
        lean = math.sin(phase * 0.9)
        sx = 1.03 + 0.05 * max(0.0, lean)
        shear = face * 0.13 * max(0.0, lean)
        dx = face * 0.035 * max(0.0, lean) * width
    elif action_name == "flee":
        step = math.sin(phase * 2.7)
        sx = 1.10 + 0.05 * abs(step)
        sy = 0.88 - 0.04 * abs(step)
        shear = -face * 0.12 * step
        dy = -0.012 * abs(step) * height
    elif action_name == "awaken":
        s = min(1.0, max(0.0, (math.sin(phase * 0.45) + 1.0) * 0.5))
        sx = 0.65 + 0.35 * s
        sy = 0.35 + 0.65 * s
        dy = 0.18 * (1.0 - s) * height

    cos_r = math.cos(rot)
    sin_r = math.sin(rot)
    out: List[Point] = []
    for x, y in pts:
        lx = x - cx
        ly = y - cy
        ax = lx * sx + shear * ly
        ay = ly * sy
        rx = ax * cos_r - ay * sin_r
        ry = ax * sin_r + ay * cos_r
        out.append([cx + rx + dx, cy + ry + dy])
    return out
