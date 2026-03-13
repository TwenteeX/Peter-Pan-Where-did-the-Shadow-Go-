"""
Render Bridge interface. Implementations send commands to the rendering engine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from peter_pan.protocols import RenderCommand


class RenderBridgeInterface(ABC):
    """Send RenderCommand to engine (OSC, WebSocket, etc.)."""

    @abstractmethod
    def send(self, command: RenderCommand) -> None:
        """Send one command to the renderer."""
        ...

    @abstractmethod
    def connect(self) -> None:
        """Establish connection if needed."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        ...
