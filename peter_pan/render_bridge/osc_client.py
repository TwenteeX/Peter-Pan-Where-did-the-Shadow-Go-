"""
OSC client for sending commands to UE/Unity/TouchDesigner.
MVP Phase 4: Python sends e.g. /peter_pan/event awaken when object is removed.
"""

from __future__ import annotations

from typing import Optional

try:
    from pythonosc import udp_client
    _HAS_OSC = True
except ImportError:
    _HAS_OSC = False

from peter_pan.protocols import RenderCommand
from peter_pan.config import load_config
from .interface import RenderBridgeInterface


class OSCBridge(RenderBridgeInterface):
    """
    Send RenderCommand as OSC messages. Default address pattern: /peter_pan/<command_type>/<name>.
    Engine side must listen on same host/port and map to animations.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        address_prefix: str = "/peter_pan",
        config: Optional[dict] = None,
    ):
        if not _HAS_OSC:
            raise RuntimeError("pythonosc is required: pip install python-osc")
        cfg = config or load_config()
        osc_cfg = cfg.get("render_bridge", {}).get("osc", {})
        self._host = host or osc_cfg.get("host", "127.0.0.1")
        self._port = port or osc_cfg.get("port", 7000)
        self._prefix = address_prefix.rstrip("/")
        self._client: Optional[udp_client.SimpleUDPClient] = None

    def connect(self) -> None:
        self._client = udp_client.SimpleUDPClient(self._host, self._port)

    def disconnect(self) -> None:
        self._client = None

    def send(self, command: RenderCommand) -> None:
        if self._client is None:
            self.connect()
        addr = f"{self._prefix}/{command.command_type}/{command.name}"
        args = command.to_osc_args()
        self._client.send_message(addr, args)
