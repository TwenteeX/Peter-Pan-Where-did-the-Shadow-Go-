"""
Agent Brain — give the shadow 'life'; plan next physical behavior from state.
Input: Semantic label, shadow position, obstacle coords.
Output: Structured action intents (e.g. climb, move_to).
"""

from .interface import AgentBrainInterface
from .simple_agent import SimpleAgent

__all__ = ["AgentBrainInterface", "SimpleAgent"]
