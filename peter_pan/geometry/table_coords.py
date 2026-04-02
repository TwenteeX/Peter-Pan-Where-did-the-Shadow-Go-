"""
Map image/ROI coordinates to unified table coordinate system (PRD Goal A).
Assumes axis-aligned ROI rectangle; table_boundary defines target range.
"""

from __future__ import annotations

from typing import Sequence


def normalize_table_roi(table_roi: list[list[float]]) -> tuple[float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax) in image pixels."""
    xs = [p[0] for p in table_roi]
    ys = [p[1] for p in table_roi]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def image_point_to_table(
    x: float,
    y: float,
    table_roi: list[list[float]],
    table_boundary: list[list[float]],
) -> tuple[float, float]:
    """Map one image point inside ROI to table (x, y)."""
    rx0, ry0, rx1, ry1 = normalize_table_roi(table_roi)
    tx0 = min(p[0] for p in table_boundary)
    ty0 = min(p[1] for p in table_boundary)
    tx1 = max(p[0] for p in table_boundary)
    ty1 = max(p[1] for p in table_boundary)
    tw, th = tx1 - tx0, ty1 - ty0
    if rx1 <= rx0 or ry1 <= ry0:
        return x, y
    u = (x - rx0) / (rx1 - rx0)
    v = (y - ry0) / (ry1 - ry0)
    return tx0 + u * tw, ty0 + v * th


def image_polygon_to_table(
    polygon: list[list[float]],
    table_roi: list[list[float]],
    table_boundary: list[list[float]],
) -> list[list[float]]:
    return [[*image_point_to_table(px, py, table_roi, table_boundary)] for px, py in polygon]


def bbox_from_points(points: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """(x, y, width, height) in same coordinate system as points."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, x1 - x0, y1 - y0
