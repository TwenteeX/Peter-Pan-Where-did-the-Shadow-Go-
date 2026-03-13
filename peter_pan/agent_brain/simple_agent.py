"""
Simple Agent Brain for MVP: on "object_visible -> False" (Severed) output "awaken", else "idle".
No LLM required for Phase 4 bridge test.
"""

from __future__ import annotations

from typing import Optional

from peter_pan.protocols import ActionIntent, PerceptionOutput, WorldState
from .interface import AgentBrainInterface


class SimpleAgent(AgentBrainInterface):
    """Maps perception events to actions: severed -> awaken, else idle."""

    def plan_next(
        self,
        perception: PerceptionOutput,
        world: WorldState,
        current_position_xyz: Optional[list[float]] = None,
    ) -> ActionIntent:
        if not perception.object_visible:
            return ActionIntent(action="awaken", params={"source": "severed"})
        # Optional: use shadow_mask centroid as target for "walk" later
        return ActionIntent(action="idle", params={})
