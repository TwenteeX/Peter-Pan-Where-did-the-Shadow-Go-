"""Select reasoning backend from config (rules vs Claude)."""

from __future__ import annotations

from typing import Any, Optional, Union

from peter_pan.config import load_config

from .claude_reasoning_agent import ClaudeReasoningAgent
from .reasoning_agent import RuleBasedReasoningAgent


def create_reasoning_agent(config: Optional[dict[str, Any]] = None) -> Union[ClaudeReasoningAgent, RuleBasedReasoningAgent]:
    cfg = config or load_config()
    ra = cfg.get("reasoning_agent", {})
    backend = str(ra.get("backend", "rules")).lower().strip()
    if backend in ("claude", "anthropic"):
        return ClaudeReasoningAgent(config=cfg)
    return RuleBasedReasoningAgent(config=cfg)
