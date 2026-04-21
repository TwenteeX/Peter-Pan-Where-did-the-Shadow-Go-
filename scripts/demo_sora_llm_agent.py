"""
Sora Video Agent demo: SAM capture -> VLM identity -> Sora/synthetic video
loop -> threshold-mask frames -> GPT motion -> plane_space preview.

This is a side-by-side experiment and does not replace the image sprite-sheet
path in scripts/demo_llm_agent.py.

Usage:
  python -m scripts.demo_sora_llm_agent --preview

Fast local smoke test without calling Sora:
  python -m scripts.demo_sora_llm_agent --preview --no-vlm --no-llm

Full Sora path:
  $env:OPENAI_API_KEY="..."
  python -m scripts.demo_sora_llm_agent --preview --generate-sora
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2
import numpy as np

from peter_pan.agent_brain.action_animation import normalize_action
from peter_pan.agent_brain.llm_agent import LLMAgent, detections_to_plane
from peter_pan.agent_brain.plane_agent import AgentPlaneBinding
from peter_pan.agent_brain.plane_preview_canvas import render_plane_space_preview
from peter_pan.generation.sora_video import (
    ThresholdMaskResult,
    build_shadow_loop_prompt,
    create_synthetic_shadow_video,
    extract_threshold_mask_frames,
    generate_sora_video,
)
from peter_pan.generation.sprite_cache import overlay_sprite_on_canvas
from peter_pan.perception.camera_shadow import CameraShadowPerception
from peter_pan.perception.vlm_openai import describe_frame_openai
from peter_pan.render_bridge.projector_window import describe_monitors, setup_projector_window
from peter_pan.world_engine.homography_plane import HomographyPlane
from scripts.demo_llm_agent import (
    _SmoothMover,
    _alpha_blit,
    _detection_crop_canvas,
    _draw_detection_overlay,
    _truncate_overlay,
)


WINDOW_PROJECTOR = "sora_agent_projection"


@dataclass
class VideoFrameLibrary:
    """Simple BGRA frame loop loaded from threshold-mask output frames."""

    frame_dir: Path
    fps: float = 12.0

    def __post_init__(self) -> None:
        self.frame_dir = Path(self.frame_dir)
        self._frames: list[np.ndarray] = []
        self._load_frames()

    @property
    def frame_count(self) -> int:
        return len(self._frames)

    def get_frame(self, now_sec: float) -> Optional[np.ndarray]:
        if not self._frames:
            return None
        idx = int(now_sec * self.fps) % len(self._frames)
        return self._frames[idx].copy()

    def _load_frames(self) -> None:
        for path in sorted(self.frame_dir.glob("*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
                img = np.concatenate([img, alpha], axis=2)
            self._frames.append(img)


def _do_vlm_sora_and_mask(
    frame_bgr,
    config,
    bounds,
    agent: LLMAgent,
    video_state: dict,
    args: argparse.Namespace,
) -> None:
    try:
        video_state["ready"] = False
        video_state["error"] = None
        video_state["library"] = None

        if args.no_vlm:
            desc = args.character
            video_state["status"] = "character_manual"
            agent.set_character(desc)
            print(f"[character] 使用手動角色描述：{desc}")
        else:
            video_state["status"] = "identifying"
            label = describe_frame_openai(frame_bgr, config, bounds_xyxy=bounds)
            desc = label.description
            agent.set_character(desc)
            print(f"[vlm] 角色辨識完成：「{desc}」→ LLM system prompt 已更新")

        prompt = _resolve_sora_prompt(args, desc)
        run_dir = _next_run_dir(args.output_dir, desc)
        run_dir.mkdir(parents=True, exist_ok=True)
        prompt_path = run_dir / "sora_prompt.txt"
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        video_state["prompt_path"] = prompt_path
        video_state["run_dir"] = run_dir

        if args.video_path is not None:
            video_state["status"] = "using_video"
            video_path = args.video_path
            print(f"[sora-video] 使用既有影片：{video_path}")
        elif args.generate_sora:
            video_state["status"] = "generating_sora"
            print(f"[sora-video] prompt saved: {prompt_path}")
            video_path = generate_sora_video(
                prompt=prompt,
                output_dir=run_dir,
                model=args.sora_model,
                seconds=args.sora_seconds,
                size=args.sora_size,
                input_reference=args.input_reference,
            )
        else:
            video_state["status"] = "synthetic_video"
            video_path = create_synthetic_shadow_video(run_dir / "synthetic_source.mp4")
            print(f"[sora-video] local synthetic video: {video_path}")

        video_state["video_path"] = video_path
        video_state["status"] = "threshold_masking"
        result = extract_threshold_mask_frames(
            video_path,
            run_dir / video_path.stem,
            threshold=args.threshold,
            foreground=args.foreground,
            frame_stride=args.frame_stride,
            max_frames=args.max_frames,
            blur=args.blur,
            morph_kernel=args.morph_kernel,
            soften_alpha=args.soften_alpha,
            keep_largest=args.keep_largest,
        )
        library = VideoFrameLibrary(result.silhouette_dir, fps=args.playback_fps)
        if library.frame_count <= 0:
            raise RuntimeError("No masked video frames were loaded.")

        video_state["result"] = result
        video_state["library"] = library
        video_state["ready"] = True
        video_state["status"] = "ready"
        print(
            "[sora-video] frame library ready: "
            f"{library.frame_count} frames from {result.silhouette_dir}"
        )
        if result.contact_sheet_path:
            print(f"[sora-video] contact sheet: {result.contact_sheet_path}")
    except Exception as exc:  # noqa: BLE001
        video_state["status"] = "error"
        video_state["error"] = str(exc)
        print(f"[sora-video] build failed: {exc}")


def _resolve_sora_prompt(args: argparse.Namespace, character_desc: str) -> str:
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8").strip()
    if args.prompt:
        return str(args.prompt).strip()
    return build_shadow_loop_prompt(
        character=character_desc or args.character,
        action=args.sora_action,
        travel_across_frame=bool(args.travel_across_frame),
    )


def _next_run_dir(output_root: Path, character_desc: str) -> Path:
    slug = _safe_name(character_desc)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path(output_root) / slug / stamp


def _safe_name(text: str) -> str:
    import re

    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:64] or "shadow_agent"


def _video_preview_canvas(
    frame_bgra,
    state: dict,
    action: str,
) -> np.ndarray:
    canvas = np.full((360, 480, 3), 235, dtype=np.uint8)
    if frame_bgra is None:
        cv2.putText(
            canvas,
            "sora frame not ready",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
    else:
        h, w = frame_bgra.shape[:2]
        scale = min(320 / max(1, h), 420 / max(1, w), 5.0)
        resized = cv2.resize(
            frame_bgra,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        x = (canvas.shape[1] - resized.shape[1]) // 2
        y = (canvas.shape[0] - resized.shape[0]) // 2
        _alpha_blit(canvas, resized, x, y)

    label = f"sora-video sprite: {action}"
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    status = str(state.get("status") or "waiting")
    cv2.putText(canvas, f"status: {status}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    result: ThresholdMaskResult | None = state.get("result")
    if result is not None:
        cv2.putText(
            canvas,
            str(result.output_dir),
            (12, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _project_uv_to_px(
    x: float,
    y: float,
    canvas_wh: tuple[int, int],
    inset_px: int,
) -> tuple[int, int]:
    w, h = canvas_wh
    inset = max(0, int(inset_px))
    min_x = inset
    min_y = inset
    max_x = max(min_x + 1, w - 1 - inset)
    max_y = max(min_y + 1, h - 1 - inset)
    px = int(round(min_x + max(0.0, min(1.0, x)) * (max_x - min_x)))
    py = int(round(min_y + max(0.0, min(1.0, y)) * (max_y - min_y)))
    return px, py


def _polygon_bounds(poly: list[list[float]]) -> Optional[tuple[float, float, float, float]]:
    pts = [(float(p[0]), float(p[1])) for p in poly if len(p) >= 2]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), max(xs), min(ys), max(ys)


def _grounded_projector_polygon(
    poly: list[list[float]],
    *,
    ground_y: float,
    pin_ground: bool,
) -> list[list[float]]:
    if not pin_ground:
        return poly
    bounds = _polygon_bounds(poly)
    if bounds is None:
        return poly
    _min_x, _max_x, _min_y, max_y = bounds
    dy = float(ground_y) - max_y
    return [[float(p[0]), float(p[1]) + dy] for p in poly if len(p) >= 2]


def _tint_sprite_alpha(sprite_bgra: Optional[np.ndarray], value: int) -> Optional[np.ndarray]:
    if sprite_bgra is None:
        return None
    out = sprite_bgra.copy()
    if out.ndim != 3 or out.shape[2] < 4:
        return out
    fill = max(0, min(255, int(value)))
    out[:, :, :3] = fill
    return out


def _render_projector_canvas(
    polygon_plane: list[list[float]],
    *,
    sprite_bgra: Optional[np.ndarray],
    canvas_wh: tuple[int, int],
    ground_y: float,
    pin_ground: bool,
    background_value: int,
    shadow_value: int,
    inset_px: int,
    facing: float,
) -> np.ndarray:
    w, h = canvas_wh
    bg = max(0, min(255, int(background_value)))
    shadow = max(0, min(255, int(shadow_value)))
    canvas = np.full((h, w, 3), bg, dtype=np.uint8)

    draw_poly = _grounded_projector_polygon(
        [list(p) for p in polygon_plane if len(p) >= 2],
        ground_y=ground_y,
        pin_ground=pin_ground,
    )
    if len(draw_poly) < 3:
        return canvas

    def to_px(x: float, y: float) -> tuple[int, int]:
        return _project_uv_to_px(x, y, canvas_wh, inset_px)

    tinted = _tint_sprite_alpha(sprite_bgra, shadow)
    if tinted is not None and overlay_sprite_on_canvas(
        canvas,
        tinted,
        draw_poly,
        to_px,
        scale=1.35,
        flip_x=facing < 0,
    ):
        return canvas

    pts = [to_px(float(p[0]), float(p[1])) for p in draw_poly]
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [arr], (shadow, shadow, shadow), cv2.LINE_AA)
    return canvas


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM Agent demo using Sora video threshold-mask frames")
    parser.add_argument("--preview", action="store_true", help="OpenCV windows (raw/mask/plane_space)")
    parser.add_argument("--projector", action="store_true", help="Open clean projector output window.")
    parser.add_argument("--list-monitors", action="store_true", help="Print monitor rectangles and exit.")
    parser.add_argument("--monitor", type=int, default=None, help="Projector monitor index; often 1 for a second display.")
    parser.add_argument("--fullscreen", action="store_true", help="Open projector window fullscreen.")
    parser.add_argument("--width", type=int, default=1920, help="Projector canvas width.")
    parser.add_argument("--height", type=int, default=1080, help="Projector canvas height.")
    parser.add_argument("--window-x", type=int, default=0, help="Projector window x position fallback.")
    parser.add_argument("--window-y", type=int, default=0, help="Projector window y position fallback.")
    parser.add_argument("--inset-px", type=int, default=0, help="Inset the projector UV area.")
    parser.add_argument("--ground-y", type=float, default=1.0, help="Projector ground line in UV space; 1.0 is the bottom edge.")
    parser.add_argument("--free-y", action="store_true", help="Do not pin the projected character bottom to --ground-y.")
    parser.add_argument("--no-background", action="store_true", help="Project black/no-light background and bright silhouette.")
    parser.add_argument("--background", type=int, default=235, help="Projector background grayscale when not using --no-background.")
    parser.add_argument("--shadow", type=int, default=None, help="Projected sprite/silhouette grayscale. Default: 0, or 255 with --no-background.")
    parser.add_argument("--generate-sora", action="store_true", help="Call Sora Video API. Off = local synthetic video.")
    parser.add_argument("--video-path", type=Path, default=None, help="Use an existing video instead of generating one.")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM identification and use --character.")
    parser.add_argument("--no-llm", action="store_true", help="Do not send GPT motion requests; useful for local Sora-frame preview.")
    parser.add_argument("--character", default="cat", help="Manual character description for --no-vlm or fallback prompt.")
    parser.add_argument("--sora-action", default="walking", help="Action in the generated Sora loop.")
    parser.add_argument("--travel-across-frame", action="store_true", help="Ask Sora to move across the frame instead of walking in place.")
    parser.add_argument("--prompt", default=None, help="Override the generated Sora prompt.")
    parser.add_argument("--prompt-file", type=Path, default=None, help="Read Sora prompt from a UTF-8 text file.")
    parser.add_argument("--sora-model", default="sora-2", choices=["sora-2", "sora-2-pro"])
    parser.add_argument("--sora-seconds", default="4", choices=["4", "8", "12"])
    parser.add_argument(
        "--sora-size",
        default="1280x720",
        choices=["1280x720", "720x1280", "1792x1024", "1024x1792"],
    )
    parser.add_argument("--input-reference", type=Path, default=None, help="Optional reference image for Sora.")
    parser.add_argument("--output-dir", type=Path, default=Path("assets/generated_sora_agent"))
    parser.add_argument("--threshold", type=int, default=170)
    parser.add_argument("--foreground", default="dark", choices=["dark", "light"])
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=48)
    parser.add_argument("--playback-fps", type=float, default=12.0)
    parser.add_argument("--blur", type=int, default=3)
    parser.add_argument("--morph-kernel", type=int, default=5)
    parser.add_argument("--soften-alpha", type=int, default=5)
    parser.add_argument("--keep-largest", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.list_monitors:
        print(describe_monitors())
        return

    perception = CameraShadowPerception()
    cfg = perception.config
    ap = AgentPlaneBinding(cfg)
    agent = LLMAgent(cfg)
    mover = _SmoothMover()

    we = cfg.get("world_engine", {}) or {}
    hom = HomographyPlane(we)
    ps = we.get("plane_space", {}) or {}
    extent_cm = ps.get("physical_extent_cm", [60.0, 40.0])
    extent_w = float(extent_cm[0]) if len(extent_cm) >= 1 else 60.0
    extent_h = float(extent_cm[1]) if len(extent_cm) >= 2 else 40.0
    canvas_wh = (max(1, int(args.width)), max(1, int(args.height)))
    background_value = 0 if args.no_background else int(args.background)
    shadow_value = int(args.shadow) if args.shadow is not None else (255 if args.no_background else 0)

    perception.start_stream()
    if args.projector:
        setup_projector_window(
            WINDOW_PROJECTOR,
            width=canvas_wh[0],
            height=canvas_wh[1],
            fullscreen=args.fullscreen,
            window_x=args.window_x,
            window_y=args.window_y,
            monitor=args.monitor,
        )

    captured = False
    pending_capture = False
    agent_enabled = False
    llm_call_count = 0
    character_label: str | None = None
    current_action = "idle"
    facing = 1.0
    last_llm_error: str | None = None
    video_state: dict = {
        "library": None,
        "status": "waiting",
        "error": None,
        "ready": False,
    }

    print(
        f"Sora Video Agent demo - model={agent._model} interval={agent.interval}s "
        f"extent_cm=({extent_w},{extent_h})\n"
        "流程：[g] SAM capture -> VLM/manual character -> Sora/synthetic video -> "
        "threshold mask -> plane_space sprite -> GPT motion.\n"
        "按鍵：[g]=建立/啟動  [c]=重新建立  IJKL=手動微調  "
        "[1-8]=手動 action  [q]=結束\n"
        "提示：不加 --generate-sora 會用本地 synthetic video 測管線。\n"
        f"projector={args.projector} canvas={canvas_wh[0]}x{canvas_wh[1]} "
        f"monitor={args.monitor} fullscreen={args.fullscreen} "
        f"ground_y={args.ground_y:.2f} no_background={args.no_background}"
    )
    if not args.preview and not args.projector:
        print("提示：這個版本需要 --preview 或 --projector 才能接收 [g]/[c] 按鍵。")
    if args.no_llm:
        print("已停用 LLM motion：只測 Sora/threshold frames 貼到 plane_space。")

    try:
        while True:
            if perception._cap is None or not perception._cap.isOpened():
                break
            ok, frame = perception._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            out, _diff, mask = perception.process_frame_bgr(frame)
            ap.update_from_perception(out, frame.shape)
            sm = out.shadow_mask

            if pending_capture:
                if sm and sm.polygon_xy and len(sm.polygon_xy) >= 3:
                    ok_cap, msg = ap.capture(out, frame.shape)
                    print(f"[capture] {msg}")
                    if ok_cap:
                        captured = True
                        agent_enabled = False
                        pending_capture = False
                        current_action = "idle"
                        mover = _SmoothMover()

                        bounds = tuple(sm.bounds_xyxy) if sm.bounds_xyxy and len(sm.bounds_xyxy) == 4 else None
                        video_state.update(
                            {
                                "library": None,
                                "status": "queued",
                                "error": None,
                                "ready": False,
                            }
                        )
                        t = threading.Thread(
                            target=_do_vlm_sora_and_mask,
                            args=(frame.copy(), cfg, bounds, agent, video_state, args),
                            daemon=True,
                        )
                        t.start()

            dets_raw = []
            dets_extra = {}
            if sm and sm.extra:
                dets_extra = sm.extra or {}
                dets_raw = dets_extra.get("detections") or []

            frame_wh = (int(frame.shape[1]), int(frame.shape[0]))
            dets_plane = detections_to_plane(dets_raw, hom, frame_wh, (extent_w, extent_h))

            if captured and video_state.get("ready") and not agent_enabled:
                current_action = "walk"
                mover = _SmoothMover()
                agent_enabled = not args.no_llm
                if agent_enabled:
                    print("[agent] Sora frames ready，開始啟用 LLM 動作/位移。")
                else:
                    print("[agent] Sora frames ready；LLM disabled，保留手動/預覽模式。")

            if agent_enabled and captured and not args.no_llm:
                dispatched = agent.maybe_request(
                    ap.to_plane_dict(),
                    dets_plane,
                    frame,
                )
                if dispatched:
                    llm_call_count += 1
                    print(f"[llm] 已送出第 {llm_call_count} 次請求...")

                intent = agent.pop_latest_intent()
                if intent is not None and intent.target_coord:
                    dx, dy = intent.target_coord[0], intent.target_coord[1]
                    dur = intent.duration_sec or 1.5
                    reason = intent.params.get("reason", "")
                    current_action = normalize_action(intent.action)
                    if current_action == "idle" and abs(dx) > 0.05:
                        current_action = "walk"
                    if abs(dx) > 0.05:
                        facing = 1.0 if dx >= 0 else -1.0
                    print(
                        f"[llm] 收到：action={current_action} dx={dx:+.1f} dy={dy:+.1f} cm "
                        f"duration={dur:.1f}s reason={reason!r}"
                    )
                    mover.set_target(dx, dy, dur, ap.manual_offset_cm())

                if agent.last_error:
                    if agent.last_error != last_llm_error:
                        print(f"[llm] error: {agent.last_error}")
                        last_llm_error = agent.last_error
                else:
                    last_llm_error = None

                ox, oy = mover.current()
                ap.set_manual_offset_cm(ox, oy)

            character_label = agent.character_desc
            video_lib: VideoFrameLibrary | None = video_state.get("library")
            video_frame = video_lib.get_frame(time.time()) if video_lib is not None else None
            poly = ap.current_polygon_plane()

            if args.projector:
                cv2.imshow(
                    WINDOW_PROJECTOR,
                    _render_projector_canvas(
                        poly,
                        sprite_bgra=video_frame,
                        canvas_wh=canvas_wh,
                        ground_y=float(args.ground_y),
                        pin_ground=not args.free_y,
                        background_value=background_value,
                        shadow_value=shadow_value,
                        inset_px=int(args.inset_px),
                        facing=facing,
                    ),
                )

            if args.preview:
                raw = frame.copy()
                _draw_detection_overlay(raw, dets_raw, dets_extra)
                status = (
                    f"Sora Video Agent calls={llm_call_count} "
                    + ("inflight..." if agent.inflight else "idle")
                )
                if character_label:
                    status += f" role={character_label[:30]}"
                if video_state.get("status") not in (None, "waiting"):
                    status += f" video={video_state.get('status')}"
                if pending_capture:
                    status = "BUILDING: waiting for SAM polygon..."
                elif not captured:
                    status = "Press [g] to build Sora video agent"
                elif video_state.get("status") == "error":
                    status = "BUILD ERROR: " + _truncate_overlay(str(video_state.get("error") or ""), 64)
                elif not video_state.get("ready"):
                    status = "BUILDING: static segment; waiting for Sora/threshold frames..."
                elif args.no_llm:
                    status = "Sora frames ready; LLM disabled"

                cv2.putText(
                    raw,
                    status,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if captured else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("raw", raw)
                cv2.imshow("yolo_crop", _detection_crop_canvas(frame, dets_raw))
                if _diff is not None:
                    cv2.imshow("diff", _diff)
                if mask is not None:
                    cv2.imshow("mask", mask)

                obstacles_for_preview = []
                for d in dets_plane:
                    obs: dict = {"label": str(d.get("class_name", "?"))}
                    if d.get("rect_uv"):
                        obs["rect_uv"] = d["rect_uv"]
                    elif d.get("center_uv"):
                        obs["center_uv"] = d["center_uv"]
                    obstacles_for_preview.append(obs)

                cv2.imshow(
                    "plane_space",
                    render_plane_space_preview(
                        poly,
                        canvas_wh=(960, 720),
                        obstacles=obstacles_for_preview,
                        character_label=character_label,
                        shadow_action=current_action,
                        animate_shadow=False,
                        fill_shadow=True,
                        facing=facing,
                        shadow_sprite=video_frame,
                    ),
                )
                cv2.imshow("sora_video_sprite", _video_preview_canvas(video_frame, video_state, current_action))

            if args.preview or args.projector:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("g"), ord("G"), ord("c"), ord("C")):
                    perception.request_sam2_once()
                    pending_capture = True
                    agent_enabled = False
                    captured = False
                    ap.clear()
                    agent = LLMAgent(cfg)
                    mover = _SmoothMover()
                    current_action = "idle"
                    video_state.update(
                        {
                            "library": None,
                            "status": "sam",
                            "error": None,
                            "ready": False,
                        }
                    )
                    print("[agent] 已觸發 SAM；capture 後會建立 Sora/video threshold frame library。")
                elif key == ord("i"):
                    ap.nudge_manual_plane(0, -ap.nudge_step)
                elif key == ord("k"):
                    ap.nudge_manual_plane(0, ap.nudge_step)
                elif key == ord("j"):
                    ap.nudge_manual_plane(-ap.nudge_step, 0)
                    facing = -1.0
                elif key == ord("l"):
                    ap.nudge_manual_plane(ap.nudge_step, 0)
                    facing = 1.0
                elif key in [ord(str(n)) for n in range(1, 9)]:
                    actions = ["idle", "walk", "hop", "stretch", "hide", "climb", "inspect", "flee"]
                    current_action = actions[int(chr(key)) - 1]
                    print(f"[manual] action={current_action}")

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if args.preview or args.projector:
            cv2.destroyAllWindows()
        perception.stop_stream()
        print(f"結束。共呼叫 LLM {llm_call_count} 次。")


if __name__ == "__main__":
    main()
