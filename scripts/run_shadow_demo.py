"""
Sprint MVP closed loop: Camera → ShadowDetector → environment.json → RuleBasedReasoningAgent
→ reasoning_output.json + debug visualization.

Keys:
  b — capture clean background (no shadow / arm)
  q — quit
  s — save environment.json and reasoning_output.json to outputs/
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# Project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

try:
    import cv2
except ImportError:
    print("pip install opencv-python")
    raise


def _open_video_capture(index: int):
    """Windows: CAP_DSHOW avoids all-black frames with the default MSMF backend."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


from peter_pan.perception.shadow_detector import ShadowDetector
from peter_pan.perception.environment_builder import (
    build_environment,
    save_environment,
    EnvironmentBuilderDefaults,
)
from peter_pan.perception.shadow_debug import draw_shadow_debug
from peter_pan.agent_brain.reasoning_factory import create_reasoning_agent
from peter_pan.config import load_config


def _step_toward(
    pos: list[float],
    target: list[float],
    speed: float,
) -> list[float]:
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    d = math.hypot(dx, dy)
    if d < 1e-6 or d <= speed:
        return [float(target[0]), float(target[1])]
    s = speed / d
    return [pos[0] + dx * s, pos[1] + dy * s]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--no-animate-agent",
        action="store_true",
        help="Keep agent position fixed (only reasoning/path updates).",
    )
    ap.add_argument("--agent-speed", type=float, default=10.0, help="Table units per frame.")
    args = ap.parse_args()

    cfg = load_config()
    perc = cfg.get("perception", {})
    cap_idx = int(perc.get("camera_index", 0))
    cap = _open_video_capture(cap_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cap_idx}")
    # Request size (many drivers ignore this; still helps on some USB cams)
    iw = int(perc.get("image_width", 0) or 0)
    ih = int(perc.get("image_height", 0) or 0)
    if iw > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, iw)
    if ih > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ih)
    # Warm up: first frames are often black or stale on Windows
    for _ in range(15):
        cap.read()

    det = ShadowDetector(config=cfg)
    agent = create_reasoning_agent(cfg)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    defaults = EnvironmentBuilderDefaults()
    eb = cfg.get("environment_builder", {})
    if eb.get("table_boundary"):
        defaults.table_boundary = eb["table_boundary"]
    objects = list(eb.get("objects", []))

    # Simulated agent position (or from config)
    sim_pos = list(eb.get("agent_start", defaults.agent_position))
    goal_pos = list(eb.get("goal_position", defaults.goal_position))

    print(__doc__)
    print("Press 'b' first to capture background with empty tabletop.")

    last_reasoning = None
    fps_t = time.time()
    nframes = 0
    warned_dark = False

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        nframes += 1
        if nframes == 5 and frame is not None and frame.size:
            m = float(frame.mean())
            if m < 2.0 and not warned_dark:
                warned_dark = True
                print(
                    "[WARN] Camera frame is almost black (mean pixel ~{:.1f}). "
                    "On Windows try: perception.camera_index: 1 (different camera), "
                    "or ensure no lens cap / use another app to verify the webcam.".format(m)
                )

        try:
            raw = det.detect(frame)
        except RuntimeError as e:
            hint = np.zeros(frame.shape, dtype=np.uint8)
            cv2.putText(hint, str(e), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(hint, "Press 'b' to capture background", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Peter Pan — Shadow MVP", hint)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                det.set_background(frame)
                print("Background captured.")
            continue

        env = build_environment(
            raw,
            defaults=defaults,
            objects=objects,
            agent_position=sim_pos,
            goal_position=goal_pos,
        )
        last_reasoning = agent.decision(env)

        if not args.no_animate_agent and last_reasoning.get("path"):
            path = last_reasoning["path"]
            if len(path) >= 1:
                target_wp = path[1] if len(path) > 1 else path[0]
                sim_pos = _step_toward(sim_pos, target_wp, args.agent_speed)

        vis = draw_shadow_debug(frame, raw, reasoning=last_reasoning, environment=env)
        comp = vis["composite"]
        if comp.shape[0] > 900:
            comp = cv2.resize(comp, None, fx=0.65, fy=0.65)

        # HUD
        fps_t2 = time.time()
        if nframes % 30 == 0:
            fps = 30 / max(1e-6, (fps_t2 - fps_t))
            fps_t = fps_t2
            print(f"~{fps:.1f} FPS | action={last_reasoning.get('high_level_action')}")

        cv2.imshow("Peter Pan — Shadow MVP", comp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("b"):
            det.set_background(frame)
            print("Background captured.")
        if key == ord("s"):
            save_environment(env, out_dir / "environment.json")
            agent.save_reasoning_output(last_reasoning, out_dir / "reasoning_output.json")
            det.save_json(raw, out_dir / "shadow_detection.json")
            print(f"Saved to {out_dir}/")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
