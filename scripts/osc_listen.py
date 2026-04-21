"""
Listen for OSC on config render_bridge.osc (default 127.0.0.1:7000).
Use alongside: python -m scripts.run_mvp_bridge

Usage:
  python -m scripts.osc_listen
"""

from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer
except ImportError:
    raise SystemExit("Install python-osc: pip install python-osc") from None

from peter_pan.config import load_config


def main() -> None:
    cfg = load_config()
    osc_cfg = cfg.get("render_bridge", {}).get("osc", {})
    host = osc_cfg.get("host", "127.0.0.1")
    port = int(osc_cfg.get("port", 7000))

    def on_any(address: str, *args: object) -> None:
        print(f"[OSC] {address}  args={list(args)}")

    dispatcher = Dispatcher()
    dispatcher.set_default_handler(on_any)

    server = BlockingOSCUDPServer((host, port), dispatcher)
    print(f"Listening on udp://{host}:{port}  (Ctrl+C to stop)")
    print("Run in another terminal: python -m scripts.run_mvp_bridge")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
