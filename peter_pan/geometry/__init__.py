"""2D table coordinate helpers for top-down MVP."""

from .table_coords import (
    bbox_from_points,
    image_polygon_to_table,
    image_point_to_table,
    normalize_table_roi,
)

__all__ = [
    "bbox_from_points",
    "image_polygon_to_table",
    "image_point_to_table",
    "normalize_table_roi",
]
