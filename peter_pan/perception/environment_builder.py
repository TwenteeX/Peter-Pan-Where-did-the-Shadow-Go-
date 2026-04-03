"""
Build environment.json from shadow detection + table + agent + goal + objects (PRD §7.2).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from peter_pan.config import load_config


@dataclass
class EnvironmentBuilderDefaults:
    agent_id: str = "peter"
    agent_position: list[float] = field(default_factory=lambda: [120.0, 640.0])
    agent_heading: float = 0.0
    agent_state: str = "idle"
    goal_type: str = "reach_platform"
    goal_position: list[float] = field(default_factory=lambda: [580.0, 180.0])
    table_boundary: list[list[float]] = field(
        default_factory=lambda: [[0.0, 0.0], [640.0, 0.0], [640.0, 720.0], [0.0, 720.0]]
    )


def _shadow_regions_for_env(
    shadow_regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach affordances for reasoning agent (PRD §6.2)."""
    out = []
    for r in shadow_regions:
        out.append(
            {
                "id": r.get("id", "shadow"),
                "polygon": r.get("polygon", []),
                "bbox": r.get("bbox", []),
                "centroid": r.get("centroid", []),
                "affordance": ["preferred_zone", "low_glare"],
            }
        )
    return out


def build_environment(
    shadow_detection: dict[str, Any],
    *,
    defaults: Optional[EnvironmentBuilderDefaults] = None,
    objects: Optional[list[dict[str, Any]]] = None,
    agent_position: Optional[list[float]] = None,
    agent_state: Optional[str] = None,
    goal_position: Optional[list[float]] = None,
    agent_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Merge shadow_detection output with agent/goal/objects into environment.json shape.
    """
    d = defaults or EnvironmentBuilderDefaults()
    ts = shadow_detection.get("timestamp", time.time())

    cfg = load_config()
    eb = cfg.get("environment_builder", {})
    sd = cfg.get("shadow_detector", {})
    table_boundary = d.table_boundary
    if eb.get("table_boundary"):
        table_boundary = eb["table_boundary"]
    elif sd.get("table_boundary"):
        table_boundary = sd["table_boundary"]

    env: dict[str, Any] = {
        "timestamp": ts,
        "table_boundary": table_boundary,
        "agent": {
            "id": agent_id or d.agent_id,
            "position": list(agent_position or d.agent_position),
            "heading": d.agent_heading,
            "state": agent_state or d.agent_state,
        },
        "goal": {
            "type": d.goal_type,
            "position": list(goal_position or d.goal_position),
        },
        "objects": list(objects if objects is not None else eb.get("objects", [])),
        "shadow_regions": _shadow_regions_for_env(shadow_detection.get("shadow_regions", [])),
    }
    # Rich context for Claude / narrative reasoning (identity, persona, relationships)
    if eb.get("identity"):
        env["identity"] = eb["identity"]
    if eb.get("scene_relationships") is not None:
        env["scene_relationships"] = eb["scene_relationships"]
    if eb.get("semantic_context"):
        env["semantic_context"] = eb["semantic_context"]
    return env


def save_environment(env: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")


def load_environment(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
