"""
Shared helpers for reasoning backends: logging, JSON extraction from LLM text.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional


def env_hash(env: dict[str, Any]) -> str:
    s = json.dumps(env, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def write_reasoning_log(
    env: dict[str, Any],
    out: dict[str, Any],
    trace: list[str],
    t0: float,
    *,
    log_dir: str,
    backend: str,
    raw_model_response: Optional[str] = None,
) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    name = f"{out['decision_id']}_{int(time.time() * 1000)}.json"
    rec: dict[str, Any] = {
        "timestamp": time.time(),
        "latency_ms": round((time.time() - t0) * 1000, 2),
        "backend": backend,
        "input_environment": env,
        "output": out,
        "rule_trace": trace,
    }
    if raw_model_response is not None:
        rec["raw_model_response"] = raw_model_response
    Path(log_dir, name).write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(log_dir, "reasoning_output_latest.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse first JSON object from model output (handles ```json fences)."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if fence:
        t = fence.group(1).strip()
    # try whole string
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # find outermost braces
    start = t.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model output")
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("Unbalanced JSON in model output")


REQUIRED_OUTPUT_KEYS = (
    "decision_id",
    "high_level_action",
    "target_object",
    "target_point",
    "next_state",
    "path",
    "reason",
)


def normalize_reasoning_output(data: dict[str, Any], decision_id: str) -> dict[str, Any]:
    """Ensure required keys exist; fill meta."""
    out = dict(data)
    if "decision_id" not in out or not out["decision_id"]:
        out["decision_id"] = decision_id
    for k in REQUIRED_OUTPUT_KEYS:
        if k not in out:
            if k == "target_object":
                out[k] = None
            elif k == "target_point":
                out[k] = [0.0, 0.0]
            elif k == "path":
                out[k] = []
            elif k == "reason":
                out[k] = ""
            else:
                out[k] = ""
    meta = out.get("meta")
    if not isinstance(meta, dict):
        out["meta"] = {}
    return out


def save_reasoning_output_file(out: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
