"""
Claude (Anthropic) reasoning: role-play as the source object behind the shadow,
use rich environment (identity, relationships, persona) and output structured JSON
for downstream generation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

from peter_pan.config import load_config

from .reasoning_common import (
    extract_json_object,
    normalize_reasoning_output,
    write_reasoning_log,
)
from .reasoning_agent import RuleBasedReasoningAgent


OUTPUT_SCHEMA_TEXT = """
You MUST respond with a single JSON object only (no markdown outside the JSON), using exactly these keys:
{
  "decision_id": "string (use dec_XXXX format if not provided)",
  "high_level_action": "one of: walk_to_waypoint | avoid_edge | approach_book_edge | climb_book | wait_in_shadow | hesitate | explore | rest",
  "target_object": "string id or null",
  "target_point": [x, y] in table coordinates (numbers),
  "next_state": "short machine-readable state label",
  "path": [ [x,y], ... ] waypoint polyline in table coordinates (can be approximate but must be valid JSON array),
  "reason": "one or two sentences: tie persona, relationships, and geometry together (English or Chinese ok)",
  "meta": {
    "persona_beat": "brief in-character thought (optional)",
    "relationship_focus": "which objects/relations drove this choice (optional)",
    "generation_hints": {
      "animation_tone": "string",
      "posture_hint": "string",
      "pacing": "slow|normal|quick"
    }
  }
}

Respect table boundaries from the environment. Prefer safety near edges. Use shadow_regions and object affordances when choosing path and mood.
"""


def _load_role_prompt(cfg: dict[str, Any]) -> str:
    ra = cfg.get("reasoning_agent", {})
    cl = ra.get("claude", {})
    inline = cl.get("role_prompt", "").strip()
    if inline:
        return inline
    path = cl.get("role_prompt_file")
    if path:
        p = Path(path)
        if not p.is_absolute():
            base = Path(__file__).resolve().parent.parent.parent
            p = base / path
        if p.exists():
            return p.read_text(encoding="utf-8")
    return (
        "You are the disembodied shadow that once belonged to a physical object on the table. "
        "You act and reason in character as that object—anthropomorphized: you have drives, quirks, and social "
        "reactions toward other items (books, obstacles, light and shadow). "
        "Use the structured environment JSON to infer spatial relations and choose the next action that fits your persona."
    )


def _api_key(cl: dict[str, Any]) -> str:
    env_name = cl.get("api_key_env", "ANTHROPIC_API_KEY")
    key = os.environ.get(env_name, "").strip()
    if key:
        return key
    # Optional: allow local dev only via config (discouraged)
    k = cl.get("api_key", "")
    if isinstance(k, str) and k.strip():
        return k.strip()
    raise RuntimeError(
        f"Missing Anthropic API key: set environment variable {env_name} "
        "(recommended) or configure reasoning_agent.claude.api_key for local dev only."
    )


class ClaudeReasoningAgent:
    """
    Calls Anthropic Messages API with system = role prompt + schema rules,
    user = full environment JSON + task instructions.
    """

    def __init__(self, config: Optional[dict] = None):
        self._yaml = config or load_config()
        ra = self._yaml.get("reasoning_agent", {})
        self._cl = ra.get("claude", {})
        self._model = self._cl.get("model", "claude-sonnet-4-20250514")
        self._max_tokens = int(self._cl.get("max_tokens", 2048))
        self._fallback = bool(ra.get("fallback_to_rules", True))
        self._decision_counter = 0
        self._rules = RuleBasedReasoningAgent(config=self._yaml)
        self._role_prompt = _load_role_prompt(self._yaml)

    def _next_id(self) -> str:
        self._decision_counter += 1
        return f"dec_{self._decision_counter:04d}"

    def decision(self, env: dict[str, Any]) -> dict[str, Any]:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass

        t0 = time.time()
        log_dir = self._yaml.get("reasoning_agent", {}).get("log_dir", "outputs/reasoning_logs")
        trace: list[str] = ["claude:messages_api"]

        try:
            _api_key(self._cl)
        except RuntimeError as e:
            trace.append(str(e))
            if self._fallback:
                trace.append("fallback_to_rules")
                out = self._rules.decision(env, log=False)
                write_reasoning_log(env, out, trace, t0, log_dir=log_dir, backend="rules_fallback_no_key")
                return out
            raise

        try:
            import anthropic
        except ImportError:
            trace.append("anthropic package missing; pip install anthropic")
            if self._fallback:
                trace.append("fallback_to_rules")
                out = self._rules.decision(env, log=False)
                write_reasoning_log(env, out, trace, t0, log_dir=log_dir, backend="rules_fallback")
                return out
            raise RuntimeError("pip install anthropic") from None

        client = anthropic.Anthropic(api_key=_api_key(self._cl))
        system = f"{self._role_prompt.strip()}\n\n{OUTPUT_SCHEMA_TEXT}"
        user_payload = {
            "instruction": (
                "Given the full scene state below, decide the single next action for the shadow-agent. "
                "Incorporate identity.persona, identity.source_object_label, scene_relationships, objects, "
                "shadow_regions, goal, and agent position. Output JSON only."
            ),
            "environment": env,
        }
        import json as _json

        user_text = _json.dumps(user_payload, ensure_ascii=False, indent=2)

        try:
            msg = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_text}],
            )
        except Exception as e:
            trace.append(f"api_error:{e!s}")
            if self._fallback:
                trace.append("fallback_to_rules")
                out = self._rules.decision(env, log=False)
                write_reasoning_log(env, out, trace, t0, log_dir=log_dir, backend="rules_fallback")
                return out
            raise

        raw = ""
        for block in msg.content:
            if block.type == "text":
                raw += block.text

        try:
            parsed = extract_json_object(raw)
            did = str(parsed.get("decision_id") or self._next_id())
            out = normalize_reasoning_output(parsed, did)
            if "meta" not in out or not isinstance(out["meta"], dict):
                out["meta"] = {}
            out["meta"]["backend"] = "claude"
            out["meta"]["model"] = self._model
            trace.append("parse_ok")
            write_reasoning_log(
                env,
                out,
                trace,
                t0,
                log_dir=log_dir,
                backend="claude",
                raw_model_response=raw,
            )
            return out
        except Exception as e:
            trace.append(f"json_parse_error:{e!s}")
            if self._fallback:
                trace.append("fallback_to_rules")
                out = self._rules.decision(env, log=False)
                out.setdefault("meta", {})
                if isinstance(out["meta"], dict):
                    out["meta"]["claude_parse_error"] = str(e)
                    out["meta"]["raw_snippet"] = raw[:2000]
                write_reasoning_log(
                    env,
                    out,
                    trace,
                    t0,
                    log_dir=log_dir,
                    backend="rules_fallback_after_claude",
                    raw_model_response=raw,
                )
                return out
            raise

    def save_reasoning_output(self, out: dict[str, Any], path: str | Path) -> None:
        from .reasoning_common import save_reasoning_output_file

        save_reasoning_output_file(out, path)
