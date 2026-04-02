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
                "method": "background_subtraction",  # or "sam"
                "threshold": 30,
            },
            "vlm": {
                "provider": "openai",  # or "huggingface", "local"
                "model": "gpt-4o",
                "prompt": "Identify the object in this image and describe its key living attributes in one short phrase (e.g. a small white cat).",
            },
        },
        "world_engine": {
            "use_depth": False,
            "voxel_size": 0.01,
        },
        "agent_brain": {
            "provider": "openai",
            "model": "gpt-4o",
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
        "shadow_detector": {
            "table_roi": [[0.0, 0.0], [640.0, 0.0], [640.0, 480.0], [0.0, 480.0]],
            "table_boundary": [[0.0, 0.0], [640.0, 0.0], [640.0, 720.0], [0.0, 720.0]],
            "color_space": "lab",
            "darkness_threshold": 18,
            "min_area_px": 800,
            "max_regions": 3,
            "morph_open_kernel": 3,
            "morph_close_kernel": 7,
            "poly_epsilon_ratio": 0.002,
            "new_event_centroid_px": 55.0,
            "new_event_area_ratio": 0.35,
        },
        "environment_builder": {
            "objects": [],
            "agent_start": [120.0, 640.0],
            "goal_position": [580.0, 180.0],
        },
        "reasoning_agent": {
            "edge_margin": 36.0,
            "grid_step": 16.0,
            "shadow_bias_weight": 12.0,
            "climbable_margin": 22.0,
            "log_dir": "outputs/reasoning_logs",
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
