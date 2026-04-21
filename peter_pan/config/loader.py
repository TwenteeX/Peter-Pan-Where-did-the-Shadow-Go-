"""
Load and validate YAML/JSON config. Paths and device IDs are configurable; no hardcoding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

# Optional: use PyYAML if available
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

import json


def get_default_config_path() -> Path:
    """Default config file next to package or in project root."""
    base = Path(__file__).resolve().parent.parent.parent
    for name in ("config.yaml", "config.yml", "config.json"):
        p = base / name
        if p.exists():
            return p
    return base / "config.yaml"


def load_config(path: Optional[os.PathLike | str] = None) -> dict[str, Any]:
    """
    Load config from file. Hardware-agnostic: only data flow and endpoints are defined.
    """
    p = Path(path) if path else get_default_config_path()
    if not p.exists():
        return _default_config_dict()

    raw = p.read_text(encoding="utf-8")
    if p.suffix in (".yaml", ".yml"):
        if not _HAS_YAML:
            raise RuntimeError("YAML config requires PyYAML: pip install pyyaml")
        data = yaml.safe_load(raw) or {}
    else:
        data = json.loads(raw)

    return _merge_with_defaults(data)


def _default_config_dict() -> dict[str, Any]:
    return {
        "project": {
            "name": "Peter Pan, Where did the Shadow Go?",
            "version": "0.1.0",
        },
        "perception": {
            "camera_index": 0,
            "image_width": 1280,
            "image_height": 720,
            "shadow": {
                "method": "background_subtraction",  # "background_subtraction" | "detector_yolo" | "detector_sam2"
                "threshold": 30,
                "min_area": 100.0,
                "detector": {
                    "model": "yolov8n.pt",
                    "min_conf": 0.25,
                    "max_objects": 10,
                },
                "tracker": {
                    "enabled": True,
                    "backend": "bytetrack",  # "bytetrack" or "simple"
                    "bytetrack_cfg": "bytetrack.yaml",
                    "iou_threshold": 0.25,
                    "center_dist_px": 120.0,
                    "max_age_frames": 20,
                },
                "sam2": {
                    "device": "cpu",
                    "model_cfg": "",
                    "checkpoint": "",
                    "run_each_frame": False,
                    "max_objects_per_sam_pass": 10,
                },
            },
            "vlm": {
                "provider": "openai",  # or "huggingface", "local"
                "enabled": False,
                "trigger_mode": "auto",  # "auto" or "manual"
                "interval_sec": 3.0,
                "only_when_object_visible": True,
                "use_largest_contour_crop": True,
                "max_image_side": 1024,
                "max_tokens": 200,
                "max_objects_per_batch": 10,
                "model": "gpt-4o",
                "prompt": "Identify the object in this image and describe its key living attributes in one short phrase (e.g. a small white cat).",
            },
        },
        "world_engine": {
            "use_depth": False,
            "voxel_size": 0.01,
            "plane_space": {
                "follow_detection": False,
                "nudge_step": 0.02,
                "physical_extent_cm": [60.0, 40.0],
                "homography": {
                    "enabled": False,
                    "image_points": [],
                    "plane_points": [],
                },
            },
        },
        "agent_brain": {
            "provider": "openai",
            "model": "gpt-4o-mini",
            "interval_sec": 3.0,
            "max_image_side": 512,
            "max_tokens": 300,
            "system_prompt": (
                "你是一個影子角色，活在一個 {w}x{h} cm 的平面空間裡。\n"
                "你的位置以 (offset_x_cm, offset_y_cm) 表示，原點在擷取時的位置。\n"
                "正 x = 右，正 y = 下。\n"
                "根據照片裡的物體與環境，決定你下一步要怎麼移動。\n"
                "只回覆一個 JSON 物件（不要 markdown）：\n"
                '{{"dx_cm": float, "dy_cm": float, "duration_sec": float, "reason": "簡短說明"}}'
            ),
        },
        "render_bridge": {
            "transport": "osc",
            "osc": {
                "host": "127.0.0.1",
                "port": 7000,
            },
            "websocket": {
                "url": "ws://127.0.0.1:8765",
            },
        },
    }


def _merge_with_defaults(loaded: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge loaded config over defaults so missing keys get defaults."""
    default = _default_config_dict()

    def merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = merge(out[k], v)
            else:
                out[k] = v
        return out

    return merge(default, loaded)
