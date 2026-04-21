"""
Direct UV wall projection MVP.

This skips Unity/TouchDesigner for the first projection pass:

  camera/perception -> AgentPlaneBinding UV world -> OpenCV projector window

Controls in the OpenCV window:
  g / c  run SAM once and capture the polygon as the wall character
  v      run VLM on YOLO tracked objects, or fallback to the largest box
  p      pipeline test: SAM capture, then VLM on the captured frame
  a / d  move left / right
  space  jump
  1-8    action animation: idle, walk, hop, stretch, hide, climb, inspect, flee
  r      reset position
  x      clear capture
  b      reset background frame (for background_subtraction mode)
  q      quit

Example:
  python -m scripts.run_uv_wall_projection --preview
  python -m scripts.run_uv_wall_projection --preview --fullscreen --window-x 1920
  python -m scripts.run_uv_wall_projection --preview --no-background
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2
import numpy as np

from peter_pan.agent_brain.action_animation import animated_polygon_plane
from peter_pan.agent_brain.plane_agent import AgentPlaneBinding
from peter_pan.perception.camera_shadow import CameraShadowPerception
from peter_pan.render_bridge.projector_window import describe_monitors, setup_projector_window


WINDOW_PROJECTION = "uv_wall_projection"
WINDOW_CAMERA = "camera_preview"
WINDOW_MASK = "mask_preview"
ACTION_KEYS = {
    ord("1"): "idle",
    ord("2"): "walk",
    ord("3"): "hop",
    ord("4"): "stretch",
    ord("5"): "hide",
    ord("6"): "climb",
    ord("7"): "inspect",
    ord("8"): "flee",
}


@dataclass
class WallPhysics:
    """Tiny UV-space physics layer for the captured shadow polygon."""

    gravity_uv: float = 1.20
    jump_velocity_uv: float = 0.36
    move_step_uv: float = 0.025
    ground_y: float = 0.88
    offset_x: float = 0.0
    offset_y: float = 0.0
    velocity_y: float = 0.0
    grounded: bool = False
    facing: float = 1.0

    def reset(self, ap: AgentPlaneBinding) -> None:
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.velocity_y = 0.0
        self.grounded = False
        self._apply_to_agent(ap)

    def clear_motion(self) -> None:
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.velocity_y = 0.0
        self.grounded = False

    def move_left(self, ap: AgentPlaneBinding) -> None:
        self.offset_x -= self.move_step_uv
        self.facing = -1.0
        self._clamp_x(ap)
        self._apply_to_agent(ap)

    def move_right(self, ap: AgentPlaneBinding) -> None:
        self.offset_x += self.move_step_uv
        self.facing = 1.0
        self._clamp_x(ap)
        self._apply_to_agent(ap)

    def jump(self) -> bool:
        if not self.grounded:
            return False
        self.velocity_y = -abs(self.jump_velocity_uv)
        self.grounded = False
        return True

    def step(self, ap: AgentPlaneBinding, dt: float) -> None:
        if not ap.active or ap.state is None:
            return
        dt = max(0.0, min(float(dt), 0.08))
        self.velocity_y += self.gravity_uv * dt
        self.offset_y += self.velocity_y * dt

        bounds = _polygon_bounds(ap.state.polygon_plane_base)
        if bounds is not None:
            _min_x, max_x, _min_y, max_y = bounds
            bottom_y = max_y + self.offset_y
            if bottom_y >= self.ground_y:
                self.offset_y = self.ground_y - max_y
                self.velocity_y = 0.0
                self.grounded = True
            else:
                self.grounded = False
            self._clamp_x(ap, bounds=bounds)

        self._apply_to_agent(ap)

    def _clamp_x(
        self,
        ap: AgentPlaneBinding,
        *,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        if not ap.active or ap.state is None:
            return
        bounds = bounds or _polygon_bounds(ap.state.polygon_plane_base)
        if bounds is None:
            return
        min_x, max_x, _min_y, _max_y = bounds
        if min_x + self.offset_x < 0.0:
            self.offset_x = -min_x
        if max_x + self.offset_x > 1.0:
            self.offset_x = 1.0 - max_x

    def _apply_to_agent(self, ap: AgentPlaneBinding) -> None:
        if not ap.active:
            return
        extent_w, extent_h = ap.physical_extent_cm
        ap.set_manual_offset_cm(
            self.offset_x * extent_w,
            self.offset_y * extent_h,
        )


def _polygon_bounds(
    polygon_plane: Sequence[Sequence[float]],
) -> Optional[Tuple[float, float, float, float]]:
    pts = [(float(p[0]), float(p[1])) for p in polygon_plane if len(p) >= 2]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def _project_uv_to_px(
    x: float,
    y: float,
    canvas_wh: Tuple[int, int],
    inset_px: int,
) -> Tuple[int, int]:
    w, h = canvas_wh
    inset = max(0, int(inset_px))
    min_x = inset
    min_y = inset
    max_x = max(min_x + 1, w - 1 - inset)
    max_y = max(min_y + 1, h - 1 - inset)
    px = int(round(min_x + max(0.0, min(1.0, x)) * (max_x - min_x)))
    py = int(round(min_y + max(0.0, min(1.0, y)) * (max_y - min_y)))
    return px, py


def render_projection_canvas(
    polygon_plane: Sequence[Sequence[float]],
    *,
    canvas_wh: Tuple[int, int],
    ground_y: float,
    action: str,
    facing: float,
    fill_value: int,
    bg_value: int,
    inset_px: int,
    show_debug: bool,
    grounded: bool,
) -> np.ndarray:
    """Render a clean projector canvas from UV-space polygon coordinates."""
    w, h = canvas_wh
    canvas = np.full((h, w, 3), int(bg_value), dtype=np.uint8)

    if show_debug:
        p0 = _project_uv_to_px(0.0, 0.0, canvas_wh, inset_px)
        p1 = _project_uv_to_px(1.0, 1.0, canvas_wh, inset_px)
        ground_a = _project_uv_to_px(0.0, ground_y, canvas_wh, inset_px)
        ground_b = _project_uv_to_px(1.0, ground_y, canvas_wh, inset_px)
        cv2.rectangle(canvas, p0, p1, (170, 170, 170), 2, cv2.LINE_AA)
        cv2.line(canvas, ground_a, ground_b, (80, 180, 80), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "UV wall  g/c=capture  A/D=move  space=jump  r=reset  q=quit",
            (24, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (90, 90, 90),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"action={action} grounded={grounded}",
            (24, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )

    draw_poly = [list(p) for p in polygon_plane if len(p) >= 2]
    if len(draw_poly) >= 3:
        draw_poly = animated_polygon_plane(
            draw_poly,
            action,
            time.time(),
            facing=facing,
        )
        pts = [
            _project_uv_to_px(float(p[0]), float(p[1]), canvas_wh, inset_px)
            for p in draw_poly
        ]
        arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [arr], (fill_value, fill_value, fill_value), cv2.LINE_AA)
        if show_debug:
            cv2.polylines(canvas, [arr], True, (35, 35, 35), 2, cv2.LINE_AA)
    elif show_debug:
        cv2.putText(
            canvas,
            "No captured polygon. Press g/c with an object visible.",
            (24, 118),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 80, 80),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _truncate_overlay(text: str, max_len: int = 56) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _detections_from_output(out: Any) -> list[dict[str, Any]]:
    sm = out.shadow_mask
    return ((sm.extra if sm else {}) or {}).get("detections") or []


def _draw_detections(
    raw_frame: np.ndarray,
    out: Any,
    perception: Optional[CameraShadowPerception] = None,
) -> None:
    sm = out.shadow_mask
    detections = _detections_from_output(out)
    for det in detections:
        b = det.get("bounds_xyxy")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in b]
        cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        tid = int(det.get("track_id", -1))
        cls = str(det.get("class_name", "?"))
        conf = float(det.get("conf", 0.0))
        label = f"id:{tid} {cls} {conf:.2f}" if tid >= 0 else f"{cls} {conf:.2f}"
        cv2.putText(
            raw_frame,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        if perception is None or tid < 0:
            continue
        semantic = perception.get_vlm_semantic_for_track(tid)
        if semantic is None:
            continue
        if semantic.description:
            vlm_label = "VLM: " + _truncate_overlay(semantic.description, 48)
            color = (0, 255, 255)
        elif semantic.extra.get("vlm_error"):
            vlm_label = "VLM err: " + _truncate_overlay(str(semantic.raw_response), 42)
            color = (0, 128, 255)
        else:
            continue
        cv2.putText(
            raw_frame,
            vlm_label,
            (x1, max(16, y1 - 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def _count_vlm_targets(detections: list[dict[str, Any]], max_n: int) -> int:
    count = 0
    for det in detections:
        if int(det.get("track_id", -1)) < 0:
            continue
        b = det.get("bounds_xyxy")
        if not b or len(b) != 4:
            continue
        count += 1
        if count >= max_n:
            break
    return count


def _trigger_vlm_multi_or_fallback(
    perception: CameraShadowPerception,
    frame: np.ndarray,
    out: Any,
) -> tuple[bool, str]:
    """Run VLM on tracked detections; fallback to one crop/full frame."""
    vlm_cfg = perception.config.get("perception", {}).get("vlm", {})
    max_n = int(vlm_cfg.get("max_objects_per_batch", 10))
    detections = _detections_from_output(out)
    target_count = _count_vlm_targets(detections, max_n)
    launched = perception.request_vlm_for_all_tracked_objects(frame, detections)
    if not launched:
        return False, "VLM skipped: busy, disabled, or provider is not OpenAI"
    if target_count > 0:
        return True, f"VLM launched for {target_count} tracked object(s)"
    return True, "VLM launched fallback for largest/latest crop"


def _vlm_status_line(perception: CameraShadowPerception) -> str:
    inflight = bool(getattr(perception, "_vlm_inflight", False))
    labels = perception.snapshot_vlm_semantics_by_track()
    if inflight:
        return "VLM: busy..."
    if not labels:
        semantic = getattr(perception, "_last_semantic", None)
        if semantic is not None and getattr(semantic, "description", ""):
            return "VLM: " + _truncate_overlay(semantic.description, 64)
        return "VLM: idle"
    pieces = []
    for tid in sorted(labels.keys())[:4]:
        label = labels[tid]
        if label.description:
            pieces.append(f"{tid}:{_truncate_overlay(label.description, 24)}")
        elif label.extra.get("vlm_error"):
            pieces.append(f"{tid}:err")
    return "VLM: " + (", ".join(pieces) if pieces else "done")


def _setup_window(
    name: str,
    *,
    width: int,
    height: int,
    fullscreen: bool,
    window_x: int,
    window_y: int,
    monitor: Optional[int],
) -> None:
    setup_projector_window(
        name,
        width=width,
        height=height,
        fullscreen=fullscreen,
        window_x=window_x,
        window_y=window_y,
        monitor=monitor,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project the current UV world directly to a wall.")
    parser.add_argument("--width", type=int, default=1920, help="Projector canvas width.")
    parser.add_argument("--height", type=int, default=1080, help="Projector canvas height.")
    parser.add_argument("--fullscreen", action="store_true", help="Open projector window fullscreen.")
    parser.add_argument("--monitor", type=int, default=None, help="Projector monitor index. Use --list-monitors to inspect.")
    parser.add_argument("--list-monitors", action="store_true", help="Print monitor rectangles and exit.")
    parser.add_argument("--window-x", type=int, default=0, help="Projector window x position; use 1920 for many second-monitor setups.")
    parser.add_argument("--window-y", type=int, default=0, help="Projector window y position.")
    parser.add_argument("--inset-px", type=int, default=0, help="Inset the UV projection rectangle inside the canvas.")
    parser.add_argument("--ground-y", type=float, default=0.88, help="Ground line in UV space; larger y is lower.")
    parser.add_argument(
        "--gravity",
        type=float,
        default=1.20,
        help="Gravity in UV units / sec^2. Higher values make jumps fall sooner.",
    )
    parser.add_argument(
        "--jump-velocity",
        type=float,
        default=0.36,
        help="Initial jump velocity in UV units / sec. Lower values make jumps smaller.",
    )
    parser.add_argument("--move-step", type=float, default=0.025, help="A/D movement step in UV units.")
    parser.add_argument(
        "--no-background",
        action="store_true",
        help=(
            "Project a black canvas (no light) and a bright silhouette. "
            "This is the closest projector equivalent of transparent/no-background output."
        ),
    )
    parser.add_argument("--background", type=int, default=235, help="Projector background grayscale value.")
    parser.add_argument(
        "--shadow",
        type=int,
        default=None,
        help="Shadow/silhouette grayscale value. Default: 0, or 255 with --no-background.",
    )
    parser.add_argument("--hide-debug", action="store_true", help="Hide border, ground, and text on projector output.")
    parser.add_argument("--preview", action="store_true", help="Show camera and mask preview windows.")
    parser.add_argument("--no-mask-preview", action="store_true", help="With --preview, hide the mask window.")
    parser.add_argument(
        "--vlm-after-capture",
        action="store_true",
        help="Automatically run VLM after each successful SAM capture.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_monitors:
        print(describe_monitors())
        return
    canvas_wh = (max(1, int(args.width)), max(1, int(args.height)))
    background_value = 0 if args.no_background else int(args.background)
    shadow_value = (
        int(args.shadow)
        if args.shadow is not None
        else (255 if args.no_background else 0)
    )
    show_debug = (not args.hide_debug) and (not args.no_background)

    perception = CameraShadowPerception()
    ap = AgentPlaneBinding(perception.config)
    physics = WallPhysics(
        gravity_uv=float(args.gravity),
        jump_velocity_uv=float(args.jump_velocity),
        move_step_uv=float(args.move_step),
        ground_y=float(args.ground_y),
    )

    action = "idle"
    pending_capture = False
    pending_vlm_after_capture = False
    last_status = ""
    last_status_time = 0.0
    last_t = time.time()

    print(
        "UV Wall Projection MVP\n"
        "Controls: g/c capture, v VLM, p pipeline, a/d move, space jump, "
        "1-8 action, r reset, x clear, b background, q quit\n"
        f"canvas={canvas_wh[0]}x{canvas_wh[1]} ground_y={physics.ground_y:.2f} "
        f"fullscreen={args.fullscreen} no_background={args.no_background} "
        f"monitor={args.monitor} background={background_value} shadow={shadow_value}"
    )

    _setup_window(
        WINDOW_PROJECTION,
        width=canvas_wh[0],
        height=canvas_wh[1],
        fullscreen=args.fullscreen,
        window_x=args.window_x,
        window_y=args.window_y,
        monitor=args.monitor,
    )

    perception.start_stream()
    try:
        while True:
            if perception._cap is None or not perception._cap.isOpened():
                break

            ok, frame = perception._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            out, _diff, mask = perception.process_frame_bgr(frame)

            if pending_capture:
                pending_capture = False
                ok_capture, msg = ap.capture(out, frame.shape)
                if ok_capture:
                    physics.clear_motion()
                    physics.step(ap, 0.0)
                    print(f"[capture] {msg}")
                    last_status = "SAM capture OK"
                    last_status_time = time.time()
                    if pending_vlm_after_capture or args.vlm_after_capture:
                        pending_vlm_after_capture = False
                        launched, detail = _trigger_vlm_multi_or_fallback(
                            perception,
                            frame,
                            out,
                        )
                        print(f"[vlm] {detail}")
                        last_status = detail if launched else detail
                        last_status_time = time.time()
                else:
                    print(f"[capture] failed: {msg}")
                    last_status = "SAM capture failed: " + msg
                    last_status_time = time.time()
                    pending_vlm_after_capture = False

            now = time.time()
            dt = now - last_t
            last_t = now
            physics.step(ap, dt)

            canvas = render_projection_canvas(
                ap.current_polygon_plane(),
                canvas_wh=canvas_wh,
                ground_y=physics.ground_y,
                action=action,
                facing=physics.facing,
                fill_value=max(0, min(255, shadow_value)),
                bg_value=max(0, min(255, background_value)),
                inset_px=int(args.inset_px),
                show_debug=show_debug,
                grounded=physics.grounded,
            )
            cv2.imshow(WINDOW_PROJECTION, canvas)

            if args.preview:
                raw = frame.copy()
                _draw_detections(raw, out, perception)
                status = "CAPTURED" if ap.active else "press g/c to SAM + capture"
                cv2.putText(
                    raw,
                    status,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if ap.active else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                vlm_line = _vlm_status_line(perception)
                cv2.putText(
                    raw,
                    vlm_line,
                    (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if last_status and time.time() - last_status_time < 5.0:
                    cv2.putText(
                        raw,
                        _truncate_overlay(last_status, 80),
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 220, 80),
                        1,
                        cv2.LINE_AA,
                    )
                cv2.imshow(WINDOW_CAMERA, raw)
                if mask is not None and not args.no_mask_preview:
                    cv2.imshow(WINDOW_MASK, mask)

            key = cv2.waitKey(1) & 0xFF
            if key in (0, 255):
                continue
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("g"), ord("G"), ord("c"), ord("C")):
                perception.request_sam2_once()
                pending_capture = True
                pending_vlm_after_capture = False
                print("[capture] SAM scheduled; capturing next processed frame.")
            elif key in (ord("p"), ord("P")):
                perception.request_sam2_once()
                pending_capture = True
                pending_vlm_after_capture = True
                print("[pipeline] SAM scheduled; capture then VLM.")
            elif key in (ord("v"), ord("V")):
                launched, detail = _trigger_vlm_multi_or_fallback(
                    perception,
                    frame,
                    out,
                )
                print(f"[vlm] {detail}")
                last_status = detail
                last_status_time = time.time()
            elif key in (ord("a"), ord("A")):
                physics.move_left(ap)
                action = "walk"
            elif key in (ord("d"), ord("D")):
                physics.move_right(ap)
                action = "walk"
            elif key == ord(" "):
                if physics.jump():
                    action = "hop"
            elif key in (ord("r"), ord("R")):
                physics.reset(ap)
                action = "idle"
                print("[physics] reset position")
            elif key in (ord("x"), ord("X")):
                ap.clear()
                physics.clear_motion()
                action = "idle"
                print("[capture] cleared")
            elif key in (ord("b"), ord("B")):
                perception.set_background(frame)
                print("[perception] background reset")
            elif key in ACTION_KEYS:
                action = ACTION_KEYS[key]
                print(f"[manual] action={action}")
    except KeyboardInterrupt:
        pass
    finally:
        perception.stop_stream()
        cv2.destroyAllWindows()
        print("Stopped.")


if __name__ == "__main__":
    main()
