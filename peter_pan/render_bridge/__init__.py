"""
Render Bridge — send commands to rendering engine (UE/Unity/TouchDesigner).
Uses OSC or WebSocket; no binding to specific engine.
"""

from .interface import RenderBridgeInterface
from .osc_client import OSCBridge
from .projector_window import MonitorRect, describe_monitors, list_monitors, setup_projector_window

__all__ = [
    "MonitorRect",
    "RenderBridgeInterface",
    "OSCBridge",
    "describe_monitors",
    "list_monitors",
    "setup_projector_window",
]
