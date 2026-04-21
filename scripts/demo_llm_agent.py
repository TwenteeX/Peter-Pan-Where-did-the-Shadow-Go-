"""
LLM Agent demo: press [g] → SAM capture → VLM identify → optional sprite
generation → GPT decides action/motion → plane_space preview.

新功能：
  [g] = 一鍵建立 agent：SAM＋capture＋VLM＋sprite cache＋啟動 LLM
  [c] = 重新 capture / rebuild agent
  每幀把 YOLO detection 映射到 plane_space 顯示為綠框

需 OPENAI_API_KEY + config method: detector_sam2 + 畫面有物體。

Usage:
  python -m scripts.demo_llm_agent --preview
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2

from peter_pan.agent_brain.llm_agent import LLMAgent, detections_to_plane
from peter_pan.agent_brain.plane_agent import AgentPlaneBinding
from peter_pan.agent_brain.plane_preview_canvas import render_plane_space_preview
from peter_pan.agent_brain.action_animation import normalize_action
from peter_pan.generation import (
    SpriteActionLibrary,
    SpriteGenerationRequest,
    generate_sprite_cache_openai,
)
from peter_pan.perception.camera_shadow import CameraShadowPerception
from peter_pan.perception.vlm_openai import describe_frame_openai
from peter_pan.world_engine.homography_plane import HomographyPlane

SPRITE_ACTIONS = ("idle", "walk", "hop", "stretch")


class _SmoothMover:
    """Lerp from current offset to target offset over duration_sec."""

    def __init__(self) -> None:
        self.start_cm = (0.0, 0.0)
        self.target_cm = (0.0, 0.0)
        self.duration = 1.0
        self.t0 = 0.0
        self.active = False

    def set_target(self, dx_cm: float, dy_cm: float, duration_sec: float, current_cm: tuple[float, float]) -> None:
        self.start_cm = current_cm
        self.target_cm = (current_cm[0] + dx_cm, current_cm[1] + dy_cm)
        self.duration = max(0.15, duration_sec)
        self.t0 = time.time()
        self.active = True

    def current(self) -> tuple[float, float]:
        if not self.active:
            return self.target_cm
        elapsed = time.time() - self.t0
        t = min(1.0, elapsed / self.duration)
        t = t * t * (3.0 - 2.0 * t)  # smoothstep
        x = self.start_cm[0] + (self.target_cm[0] - self.start_cm[0]) * t
        y = self.start_cm[1] + (self.target_cm[1] - self.start_cm[1]) * t
        if t >= 1.0:
            self.active = False
        return (x, y)


def _do_vlm_identify_and_generate(
    frame_bgr,
    config,
    bounds,
    agent,
    sprite_state: dict,
    *,
    generate_sprites: bool,
):
    """Background thread: VLM identity, then optionally generate sprite sheets."""
    try:
        sprite_state["status"] = "identifying"
        sprite_state["idle_ready"] = False
        sprite_state["actions_ready"] = False
        label = describe_frame_openai(frame_bgr, config, bounds_xyxy=bounds)
        desc = label.description
        agent.set_character(desc)
        print(f"[vlm] 角色辨識完成：「{desc}」→ LLM system prompt 已更新")
        if not generate_sprites:
            sprite_state["status"] = "disabled"
            sprite_state["idle_ready"] = True
            sprite_state["actions_ready"] = True
            return

        sprite_state["status"] = "generating_idle"
        idle_req = SpriteGenerationRequest(
            character_description=desc,
            actions=("idle",),
            frames_per_action=8,
        )
        idle_lib = generate_sprite_cache_openai(idle_req)
        sprite_state["library"] = idle_lib
        sprite_state["idle_ready"] = True
        sprite_state["status"] = "idle_ready"
        sprite_state["error"] = None
        print(
            "[sprite] idle ready: "
            f"{idle_lib.root_dir} actions={', '.join(idle_lib.available_actions)}"
        )

        remaining_actions = tuple(a for a in SPRITE_ACTIONS if a != "idle")
        if remaining_actions:
            sprite_state["status"] = "generating_actions"
            try:
                action_req = SpriteGenerationRequest(
                    character_description=desc,
                    actions=remaining_actions,
                    frames_per_action=8,
                )
                lib = generate_sprite_cache_openai(action_req)
            except Exception as exc:  # keep the idle sprite usable
                sprite_state["status"] = "idle_ready_error"
                sprite_state["error"] = str(exc)
                print(f"[sprite] idle 已可用，但其他 action 生成失敗：{exc}")
                return
        else:
            lib = idle_lib

        sprite_state["library"] = lib
        sprite_state["status"] = "ready"
        sprite_state["actions_ready"] = True
        sprite_state["error"] = None
        print(
            "[sprite] cache ready: "
            f"{lib.root_dir} actions={', '.join(lib.available_actions)}"
        )
    except Exception as exc:
        sprite_state["status"] = "error"
        sprite_state["error"] = str(exc)
        print(f"[agent-build] VLM / sprite 失敗：{exc}")


def _sprite_frame_for_action(
    library: SpriteActionLibrary | None,
    action: str,
    now_sec: float,
):
    if library is None:
        return None
    frame = library.get_frame(action, now_sec)
    if frame is not None:
        return frame
    for fallback in ("idle", "walk"):
        if action == fallback:
            continue
        frame = library.get_frame(fallback, now_sec)
        if frame is not None:
            return frame
    return None


def _sprite_preview_canvas(sprite_bgra, action: str, sprite_lib: SpriteActionLibrary | None):
    """White-background sprite preview so black generated silhouettes are obvious."""
    import numpy as np

    canvas = np.full((360, 480, 3), 235, dtype=np.uint8)
    if sprite_bgra is None:
        cv2.putText(
            canvas,
            "sprite not ready",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 30, 30),
            2,
            cv2.LINE_AA,
        )
        return canvas
    h, w = sprite_bgra.shape[:2]
    scale = min(360 / max(1, h), 420 / max(1, w), 5.0)
    resized = cv2.resize(
        sprite_bgra,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (canvas.shape[1] - resized.shape[1]) // 2
    y = (canvas.shape[0] - resized.shape[0]) // 2
    _alpha_blit(canvas, resized, x, y)
    label = f"sprite: {action}"
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    if sprite_lib is not None:
        cv2.putText(
            canvas,
            str(sprite_lib.root_dir),
            (12, canvas.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _truncate_overlay(text: str, max_len: int = 56) -> str:
    text = (text or "").replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def _draw_detection_overlay(raw, detections: list[dict], extra: dict) -> None:
    """Draw YOLO boxes plus detector/SAM status directly on raw preview."""
    for d in detections:
        b = d.get("bounds_xyxy")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        cv2.rectangle(raw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = str(d.get("class_name", "?"))
        conf = float(d.get("conf", 0.0))
        tid = int(d.get("track_id", -1))
        label = f"id:{tid} {cls} {conf:.2f}" if tid >= 0 else f"{cls} {conf:.2f}"
        cv2.putText(
            raw,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    method = str(extra.get("method", "?"))
    det_count = extra.get("detected_count", len(detections))
    lines = [f"{method}  yolo={det_count}"]
    if extra.get("sam2_ran"):
        lines.append(
            f"SAM ran masks={extra.get('sam2_masks_applied', '?')} "
            f"score={float(extra.get('sam2_score', 0.0)):.2f}"
        )
    elif extra.get("sam2_error"):
        lines.append("SAM error: " + _truncate_overlay(str(extra.get("sam2_error")), 68))
    elif extra.get("error"):
        lines.append("detector error: " + _truncate_overlay(str(extra.get("error")), 68))
    elif extra.get("sam2_note"):
        lines.append(_truncate_overlay(str(extra.get("sam2_note")), 68))
    elif extra.get("sam2_skipped"):
        lines.append("SAM skipped: press [g] or [c] to run once")
    elif extra.get("reason"):
        lines.append("reason: " + str(extra.get("reason")))

    for i, line in enumerate(lines):
        cv2.putText(
            raw,
            line,
            (10, 56 + i * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _detection_crop_canvas(frame_bgr, detections: list[dict]):
    """Preview the largest YOLO crop that SAM/VLM is likely to use."""
    import numpy as np

    canvas = np.full((360, 480, 3), 30, dtype=np.uint8)
    valid = []
    for d in detections:
        b = d.get("bounds_xyxy")
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in b]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > 0:
            valid.append((area, d, (x1, y1, x2, y2)))
    if not valid:
        cv2.putText(canvas, "no YOLO crop", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2)
        return canvas

    _area, d, (x1, y1, x2, y2) = max(valid, key=lambda item: item[0])
    h, w = frame_bgr.shape[:2]
    pad = 12
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return canvas
    ch, cw = crop.shape[:2]
    scale = min(canvas.shape[1] / max(1, cw), canvas.shape[0] / max(1, ch))
    resized = cv2.resize(crop, (max(1, int(cw * scale)), max(1, int(ch * scale))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    ox = (canvas.shape[1] - rw) // 2
    oy = (canvas.shape[0] - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = resized
    cls = str(d.get("class_name", "?"))
    conf = float(d.get("conf", 0.0))
    cv2.putText(canvas, f"{cls} {conf:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return canvas


def _alpha_blit(canvas_bgr, sprite_bgra, x: int, y: int) -> None:
    h, w = canvas_bgr.shape[:2]
    sh, sw = sprite_bgra.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + sw)
    y2 = min(h, y + sh)
    if x1 >= x2 or y1 >= y2:
        return
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    src = sprite_bgra[sy1:sy2, sx1:sx2]
    if src.shape[2] < 4:
        canvas_bgr[y1:y2, x1:x2] = src[:, :, :3]
        return
    alpha = src[:, :, 3:4].astype("float32") / 255.0
    dst = canvas_bgr[y1:y2, x1:x2].astype("float32")
    rgb = src[:, :, :3].astype("float32")
    canvas_bgr[y1:y2, x1:x2] = (rgb * alpha + dst * (1.0 - alpha)).astype("uint8")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Agent demo: GPT drives plane shadow")
    parser.add_argument("--preview", action="store_true", help="OpenCV windows (raw/mask/plane_space)")
    parser.add_argument(
        "--no-generate-sprites",
        action="store_true",
        help="Skip OpenAI image generation; use procedural polygon animation only.",
    )
    args = parser.parse_args()

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

    perception.start_stream()

    captured = False
    pending_capture = False
    agent_enabled = False
    frame_idx = 0
    llm_call_count = 0
    character_label: str | None = None
    current_action = "idle"
    facing = 1.0
    sprite_state: dict = {
        "library": None,
        "status": "waiting",
        "error": None,
        "idle_ready": False,
        "actions_ready": False,
    }
    last_llm_error: str | None = None

    print(
        f"LLM Agent demo — model={agent._model}  interval={agent.interval}s  "
        f"extent_cm=({extent_w},{extent_h})\n"
        "流程：按 [g] → SAM capture → VLM 辨識 → sprite cache → GPT action/motion。\n"
        "按鍵：[g]=建立/啟動 agent  [c]=重新建立  IJKL=手動微調  "
        "[1-8]=手動 action  [q]=結束\n"
        "actions: 1 idle, 2 walk, 3 hop, 4 stretch, 5 hide, 6 climb, 7 inspect, 8 flee\n"
        "需 OPENAI_API_KEY。Ctrl+C 或 q 結束。"
    )
    if not args.preview:
        print("提示：這個版本需要 --preview 才能接收 [g]/[c] 按鍵。")
    if args.no_generate_sprites:
        print("已停用 sprite generation：會使用本地 polygon action animation。")

    try:
        while True:
            if perception._cap is None or not perception._cap.isOpened():
                break
            ok, frame = perception._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            frame_idx += 1

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
                        sprite_state["library"] = None
                        sprite_state["status"] = "queued"
                        sprite_state["error"] = None
                        sprite_state["idle_ready"] = False
                        sprite_state["actions_ready"] = False
                        t = threading.Thread(
                            target=_do_vlm_identify_and_generate,
                            args=(frame.copy(), cfg, bounds, agent, sprite_state),
                            kwargs={"generate_sprites": not args.no_generate_sprites},
                            daemon=True,
                        )
                        t.start()
                else:
                    pass

            dets_raw = []
            dets_extra = {}
            if sm and sm.extra:
                dets_extra = sm.extra or {}
                dets_raw = dets_extra.get("detections") or []

            frame_wh = (int(frame.shape[1]), int(frame.shape[0]))
            dets_plane = detections_to_plane(dets_raw, hom, frame_wh, (extent_w, extent_h))

            if captured and not agent_enabled and (
                sprite_state.get("idle_ready") or sprite_state.get("library") is not None
            ):
                agent_enabled = True
                current_action = "idle"
                mover = _SmoothMover()
                print("[agent] first sprite ready，開始啟用 LLM 動作/位移；其他 action 會繼續背景生成。")

            if agent_enabled and captured:
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
                    if abs(dx) > 0.05:
                        facing = 1.0 if dx >= 0 else -1.0
                    print(
                        f"[llm] 收到：action={current_action}  dx={dx:+.1f} dy={dy:+.1f} cm  "
                        f"duration={dur:.1f}s  reason={reason!r}"
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

            if args.preview:
                raw = frame.copy()
                _draw_detection_overlay(raw, dets_raw, dets_extra)

                status = (
                    f"LLM Agent  calls={llm_call_count}  "
                    + ("inflight..." if agent.inflight else "idle")
                )
                if character_label:
                    status += f"  role={character_label[:30]}"
                if sprite_state.get("status") not in (None, "waiting", "disabled"):
                    status += f"  sprite={sprite_state.get('status')}"
                if pending_capture:
                    status = "BUILDING: waiting for SAM polygon..."
                elif not captured:
                    status = "Press [g] to build agent (SAM + VLM + sprites)"
                elif sprite_state.get("status") == "error":
                    status = "BUILD ERROR: " + _truncate_overlay(str(sprite_state.get("error") or ""), 64)
                elif not sprite_state.get("idle_ready"):
                    status = "BUILDING: static segment; waiting for idle sprite..."
                elif sprite_state.get("status") == "idle_ready_error":
                    status = "IDLE sprite placed; action sprite error (see console)"
                elif not agent_enabled:
                    status = "IDLE sprite placed; finishing action sprites..."
                cv2.putText(
                    raw, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0) if captured else (0, 165, 255),
                    2, cv2.LINE_AA,
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

                poly = ap.current_polygon_plane()
                sprite_lib = sprite_state.get("library")
                now_sec = time.time()
                sprite_frame = _sprite_frame_for_action(
                    sprite_lib,
                    current_action,
                    now_sec,
                )
                animate_shadow = bool(
                    agent_enabled
                    and sprite_frame is None
                    and (args.no_generate_sprites or sprite_lib is None)
                )
                cv2.imshow(
                    "plane_space",
                    render_plane_space_preview(
                        poly,
                        canvas_wh=(960, 720),
                        obstacles=obstacles_for_preview,
                        character_label=character_label,
                        shadow_action=current_action,
                        animate_shadow=animate_shadow,
                        fill_shadow=True,
                        facing=facing,
                        shadow_sprite=sprite_frame,
                    ),
                )
                if sprite_lib is not None:
                    cv2.imshow(
                        "sprite_preview",
                        _sprite_preview_canvas(sprite_frame, current_action, sprite_lib),
                    )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                elif key in (ord("g"), ord("G"), ord("c"), ord("C")):
                    perception.request_sam2_once()
                    pending_capture = True
                    agent_enabled = False
                    captured = False
                    ap.clear()
                    agent = LLMAgent(cfg)
                    mover = _SmoothMover()
                    current_action = "idle"
                    sprite_state["library"] = None
                    sprite_state["status"] = "sam"
                    sprite_state["error"] = None
                    sprite_state["idle_ready"] = False
                    sprite_state["actions_ready"] = False
                    print("[agent] 已觸發 SAM，等待 polygon 後建立 agent；idle sprite 會先生成並放到 plane_space。")
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
        if args.preview:
            cv2.destroyAllWindows()
        perception.stop_stream()
        print(f"結束。共呼叫 LLM {llm_call_count} 次。")


if __name__ == "__main__":
    main()
