"""
MVP Phase 4: Run perception loop; when object is removed (object_visible -> False),
send OSC command to trigger "awaken" in the render engine.

Usage:
  python -m scripts.run_mvp_bridge

Ensure render engine (UE/Unity/TouchDesigner) is listening on config render_bridge.osc host/port.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from peter_pan.perception.camera_shadow import CameraShadowPerception
from peter_pan.perception.interface import PerceptionHubInterface
from peter_pan.world_engine import StubWorldEngine
from peter_pan.agent_brain import SimpleAgent
from peter_pan.render_bridge import OSCBridge
from peter_pan.protocols import PerceptionOutput, RenderCommand


def main() -> None:
    perception: PerceptionHubInterface = CameraShadowPerception()
    world = StubWorldEngine()
    agent = SimpleAgent()
    bridge = OSCBridge()
    bridge.connect()

    last_visible: bool | None = None

    def on_perception(out: PerceptionOutput) -> None:
        nonlocal last_visible
        intent = agent.plan_next(out, world.get_state())
        # Only send when transition: visible -> not visible (Severed)
        if last_visible is True and out.object_visible is False:
            cmd = RenderCommand(
                command_type="event",
                name=intent.action,
                extra=intent.to_dict(),
            )
            bridge.send(cmd)
            print(f"[Bridge] Sent OSC: event/{intent.action}")
        last_visible = out.object_visible

    print("MVP Bridge: perception -> agent -> OSC. Remove object to trigger 'awaken'.")
    print("Press Ctrl+C to stop.")
    perception.start_stream()
    try:
        while True:
            out = perception.capture_frame()
            if out is not None:
                on_perception(out)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        perception.stop_stream()
        bridge.disconnect()


if __name__ == "__main__":
    main()
