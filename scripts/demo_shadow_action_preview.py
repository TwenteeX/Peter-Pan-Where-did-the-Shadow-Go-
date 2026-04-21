"""
Offline preview for the shadow action vocabulary.

No camera, SAM2, OpenAI, or TouchDesigner required. This is the fastest way to
test the local action renderer before wiring it to a captured polygon.

Usage:
  python -m scripts.demo_shadow_action_preview

Keys:
  1 idle, 2 walk, 3 hop, 4 stretch, 5 hide, 6 climb, 7 inspect, 8 flee
  WASD move, r reset, q quit
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2

from peter_pan.agent_brain.plane_preview_canvas import render_plane_space_preview


def _sample_shadow_polygon() -> list[list[float]]:
    """A small creature-like silhouette in plane UV coordinates."""
    return [
        [0.40, 0.52],
        [0.43, 0.45],
        [0.49, 0.42],
        [0.56, 0.44],
        [0.61, 0.49],
        [0.64, 0.56],
        [0.61, 0.62],
        [0.55, 0.65],
        [0.48, 0.64],
        [0.42, 0.60],
    ]


def main() -> None:
    actions = ["idle", "walk", "hop", "stretch", "hide", "climb", "inspect", "flee"]
    action = "idle"
    facing = 1.0
    ox = 0.0
    oy = 0.0
    step = 0.015

    print(
        "Shadow action preview\n"
        "1 idle, 2 walk, 3 hop, 4 stretch, 5 hide, 6 climb, 7 inspect, 8 flee\n"
        "WASD move, r reset, q quit"
    )

    try:
        while True:
            poly = [[x + ox, y + oy] for x, y in _sample_shadow_polygon()]
            img = render_plane_space_preview(
                poly,
                canvas_wh=(960, 720),
                shadow_action=action,
                animate_shadow=True,
                fill_shadow=True,
                facing=facing,
            )
            cv2.imshow("shadow_action_preview", img)
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in [ord(str(n)) for n in range(1, 9)]:
                action = actions[int(chr(key)) - 1]
                print(f"[manual] action={action}")
            elif key in (ord("w"), ord("W")):
                oy -= step
            elif key in (ord("s"), ord("S")):
                oy += step
            elif key in (ord("a"), ord("A")):
                ox -= step
                facing = -1.0
            elif key in (ord("d"), ord("D")):
                ox += step
                facing = 1.0
            elif key in (ord("r"), ord("R")):
                ox = 0.0
                oy = 0.0
            time.sleep(0.001)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
