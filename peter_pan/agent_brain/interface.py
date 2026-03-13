"""
Agent Brain interface. Implementations plan actions from state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from peter_pan.protocols import ActionIntent, PerceptionOutput, WorldState


class AgentBrainInterface(ABC):
    """
    Input: semantic label, current shadow position, world state (obstacles/surfaces).
    Output: ActionIntent (action name, target_coord, params).
    """

    @abstractmethod
    def plan_next(
        self,
        perception: PerceptionOutput,
        world: WorldState,
        current_position_xyz: Optional[list[float]] = None,
    ) -> ActionIntent:
        """Compute next action from current perception and world."""
        ...
