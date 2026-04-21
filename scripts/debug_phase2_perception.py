"""
Phase 2: tune CameraShadowPerception — console stats + optional mask preview.

Start with an EMPTY scene (no object on table), run once to lock background via start_stream,
or use the control panel button / key [b] to re-grab background.

Usage:
  python -m scripts.debug_phase2_perception
  python -m scripts.debug_phase2_perception --preview
  python -m scripts.debug_phase2_perception --preview --no-tk   # OpenCV only
  python -m scripts.debug_phase2_perception --preview --no-plane-space
  python -m scripts.debug_phase2_perception --control-panel    # console + buttons, no OpenCV windows
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2
import numpy as np

from peter_pan.agent_brain.plane_agent import AgentPlaneBinding
from peter_pan.agent_brain.plane_preview_canvas import render_plane_space_preview
from peter_pan.perception.camera_shadow import CameraShadowPerception

_agent_plane: AgentPlaneBinding | None = None


def _get_agent_plane(perception: CameraShadowPerception) -> AgentPlaneBinding:
    global _agent_plane
    if _agent_plane is None:
        _agent_plane = AgentPlaneBinding(perception.config)
    return _agent_plane


def _handle_plane_nudge_keys(key: int, perception: CameraShadowPerception) -> None:
    """IJKL = 在平面座標平移輪廓（與 S=SAM2 不衝突）。"""
    ap = _get_agent_plane(perception)
    if not ap.active:
        return
    step = ap.nudge_step
    if key == ord("i"):
        ap.nudge_manual_plane(0.0, -step)
    elif key == ord("k"):
        ap.nudge_manual_plane(0.0, step)
    elif key == ord("j"):
        ap.nudge_manual_plane(-step, 0.0)
    elif key == ord("l"):
        ap.nudge_manual_plane(step, 0.0)


def _handle_preview_hotkey(
    key: int,
    perception: CameraShadowPerception,
    frame,
    out,
    ap: AgentPlaneBinding,
    *,
    log_prefix: str = "[preview]",
) -> bool:
    """Single key from OpenCV waitKey or Tkinter. Returns True = quit."""
    if key == ord("q"):
        return True
    if key == ord("b"):
        perception.set_background(frame)
        print(f"{log_prefix} 已更新背景（基準畫面）。")
    if key == ord("v"):
        ok, detail = _trigger_vlm_multi_or_fallback(perception, frame, out)
        print(f"{log_prefix} VLM 多物體：{detail}" if ok else f"{log_prefix} {detail}")
    if key == ord("s"):
        perception.request_sam2_once()
        print(f"{log_prefix} 已排程下一幀執行 SAM2（需 method: detector_sam2）。")
    if key == ord("a"):
        ok, msg = ap.capture(out, frame.shape)
        print(f"{log_prefix} Agent 平面：{msg}" if ok else f"{log_prefix} Agent 平面失敗：{msg}")
    if key == ord("x"):
        ap.clear()
        print(f"{log_prefix} 已清除 Agent 平面鎖定。")
    _handle_plane_nudge_keys(key, perception)
    return False


def _truncate_overlay(s: str, max_len: int = 52) -> str:
    s = (s or "").replace("\n", " ").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _vlm_max_batch(perception: CameraShadowPerception) -> int:
    return int(
        perception.config.get("perception", {})
        .get("vlm", {})
        .get("max_objects_per_batch", 10)
    )


def _count_tracked_for_vlm(detections: list, max_n: int) -> int:
    n = 0
    for det in detections:
        if int(det.get("track_id", -1)) < 0:
            continue
        b = det.get("bounds_xyxy")
        if not b or len(b) != 4:
            continue
        n += 1
        if n >= max_n:
            break
    return n


def _trigger_vlm_multi_or_fallback(
    perception: CameraShadowPerception, frame, out
) -> tuple[bool, str]:
    """按一次：對每個有效 track_id 各送一個 VLM；若無 track 則退回單次辨識。"""
    sm = out.shadow_mask
    dets = (sm.extra if sm else {}) or {}
    detections = dets.get("detections") or []
    max_n = _vlm_max_batch(perception)
    n_tracked = _count_tracked_for_vlm(detections, max_n)
    launched = perception.request_vlm_for_all_tracked_objects(frame, detections)
    if not launched:
        return False, "已跳過（忙碌、未啟用或非 OpenAI provider）"
    if n_tracked > 0:
        return True, f"已送 {n_tracked} 個 API 請求（每個 track_id 一張裁切，最多 {max_n}）"
    return True, "無有效 track_id，已退回單次辨識（整張或最後框）"


def _draw_agent_plane_overlay(
    raw_frame,
    frame_shape: tuple[int, ...],
    perception: CameraShadowPerception,
) -> None:
    """Draw translated agent polygon in image pixels (magenta)."""
    ap = _get_agent_plane(perception)
    if not ap.active:
        return
    pts = ap.current_polygon_xy_pixels(frame_shape)
    if len(pts) < 3:
        return
    arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(
        raw_frame,
        [arr],
        isClosed=True,
        color=(255, 0, 255),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    st = ap.state
    if st is not None:
        u, v = st.current_centroid_uv
        h, w = int(frame_shape[0]), int(frame_shape[1])
        cx, cy = int(u * (w - 1)), int(v * (h - 1))
        cv2.circle(raw_frame, (cx, cy), 6, (255, 0, 255), -1, cv2.LINE_AA)


def _draw_detections(raw_frame, out, perception: CameraShadowPerception | None = None) -> None:
    """Overlay YOLO detections on raw frame when available."""
    if out.shadow_mask is None:
        return
    detections = (out.shadow_mask.extra or {}).get("detections") or []
    for d in detections:
        bounds = d.get("bounds_xyxy")
        if not bounds or len(bounds) != 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in bounds]
        cv2.rectangle(raw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cls = str(d.get("class_name", "?"))
        conf = float(d.get("conf", 0.0))
        tid = d.get("track_id", -1)
        text = f"id:{tid} {cls} {conf:.2f}" if int(tid) >= 0 else f"{cls} {conf:.2f}"
        y_id = max(28, y1 - 6)
        y_vlm = max(14, y1 - 22)
        if perception is not None and int(tid) >= 0:
            sem = perception.get_vlm_semantic_for_track(int(tid))
            if sem is not None:
                if sem.description:
                    line_vlm = f"VLM: {_truncate_overlay(sem.description)}"
                    color = (0, 255, 255)
                elif sem.extra.get("vlm_error"):
                    line_vlm = f"VLM err: {_truncate_overlay(sem.raw_response, 40)}"
                    color = (0, 128, 255)
                else:
                    line_vlm = ""
                    color = (0, 255, 255)
                if line_vlm:
                    cv2.putText(
                        raw_frame,
                        line_vlm,
                        (x1, y_vlm),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
        cv2.putText(
            raw_frame,
            text,
            (x1, y_id),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )


def run_sync_loop(perception: CameraShadowPerception, args: argparse.Namespace) -> None:
    """Blocking loop: optional OpenCV preview, keyboard b / q only."""
    print(
        "Phase 2 debug: object_visible=True means large foreground vs background.\n"
        "（含 Tkinter 面板時）快捷鍵在「控制視窗」或「OpenCV 視窗」有焦點時皆有效；"
        "僅 OpenCV 時請點一下 raw/mask 視窗再按鍵。\n"
        "[b][v][s][a][x][i/j/k/l][q]；plane_space 為 2D 平面預覽。Ctrl+C stops."
    )
    last_log = 0.0
    try:
        while True:
            if perception._cap is None or not perception._cap.isOpened():
                break
            ok, frame = perception._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            out, diff, mask = perception.process_frame_bgr(frame)

            ap = _get_agent_plane(perception)
            ap.update_from_perception(out, frame.shape)

            now = time.time()
            if now - last_log >= args.interval:
                last_log = now
                _log_line(out, perception)

            if args.preview:
                raw = frame.copy()
                _draw_detections(raw, out, perception)
                _draw_agent_plane_overlay(raw, frame.shape, perception)
                cv2.imshow("raw", raw)
                if diff is not None:
                    cv2.imshow("diff", diff)
                if mask is not None:
                    cv2.imshow("mask", mask)
                if args.plane_space:
                    poly = ap.current_polygon_plane()
                    cv2.imshow(
                        "plane_space",
                        render_plane_space_preview(poly, canvas_wh=(960, 720)),
                    )
                key = cv2.waitKey(1) & 0xFF
                if key not in (0, 255) and _handle_preview_hotkey(
                    key, perception, frame, out, ap
                ):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        perception.stop_stream()


def run_tk_loop(perception: CameraShadowPerception, args: argparse.Namespace) -> None:
    """Tk mainloop + after(): camera + optional OpenCV + control panel buttons."""
    import tkinter as tk
    from tkinter import ttk

    last_frame: list = [None]
    reset_requested = [False]
    vlm_requested = [False]
    sam_requested = [False]
    agent_capture_requested = [False]
    agent_clear_requested = [False]
    # 一鍵：本幀 request SAM，process 後立刻擷取平面（同一幀完成）
    pending_capture_after_sam = [False]
    running = [True]

    root = tk.Tk()
    root.title("Peter Pan — Phase 2 除錯")
    root.resizable(True, False)

    tk_key_queue: deque[str] = deque()

    def on_tk_key(_event: tk.Event) -> None:
        sym = _event.keysym.lower()
        if sym in ("q", "b", "v", "s", "a", "x", "i", "j", "k", "l"):
            tk_key_queue.append(sym)

    root.bind_all("<Key>", on_tk_key)

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill=tk.X)

    hint = (
        "「一鍵 SAM＋平面」＝先跑 SAM 再擷取輪廓到 plane_space；raw 視窗會一直畫 YOLO 框。\n"
        "快捷鍵：b 背景、v VLM、s SAM、a 擷取、x 清除、ijkl 平面平移、q 結束。\n"
        "「重設背景」＝空場景時把目前畫面當基準（背景差分用）。"
    )
    ttk.Label(frm, text=hint, wraplength=420).pack(anchor=tk.W, pady=(0, 8))

    def on_reset_background() -> None:
        reset_requested[0] = True

    def on_quit() -> None:
        running[0] = False
        root.destroy()

    def on_vlm_once() -> None:
        vlm_requested[0] = True

    def on_sam_once() -> None:
        sam_requested[0] = True

    def on_sam_and_plane_once() -> None:
        perception.request_sam2_once()
        pending_capture_after_sam[0] = True
        print("[control] 一鍵 SAM＋平面：已排程本幀 SAM，完成後擷取輪廓。")

    def on_agent_capture() -> None:
        agent_capture_requested[0] = True

    def on_agent_clear() -> None:
        agent_clear_requested[0] = True

    ttk.Button(
        frm,
        text="重設背景（基準畫面）",
        command=on_reset_background,
    ).pack(fill=tk.X, pady=2)
    ttk.Button(
        frm,
        text="VLM 多物體辨識（每個 track_id 各 1 次 API）",
        command=on_vlm_once,
    ).pack(fill=tk.X, pady=2)
    ttk.Button(
        frm,
        text="SAM2 精細分割（單次，需 perception.shadow.method: detector_sam2）",
        command=on_sam_once,
    ).pack(fill=tk.X, pady=2)
    ttk.Button(
        frm,
        text="一鍵：SAM ＋ 擷取到 plane_space（raw 仍顯示 YOLO）",
        command=on_sam_and_plane_once,
    ).pack(fill=tk.X, pady=2)
    ttk.Button(
        frm,
        text="僅擷取平面（已有 SAM 輪廓／快取時）",
        command=on_agent_capture,
    ).pack(fill=tk.X, pady=2)
    ttk.Button(
        frm,
        text="清除 Agent 平面鎖定",
        command=on_agent_clear,
    ).pack(fill=tk.X, pady=2)
    ttk.Button(frm, text="結束", command=on_quit).pack(fill=tk.X, pady=2)

    root.protocol("WM_DELETE_WINDOW", on_quit)

    last_log = [0.0]

    def tick() -> None:
        if not running[0]:
            return
        try:
            if not root.winfo_exists():
                return
        except tk.TclError:
            return

        if perception._cap is None or not perception._cap.isOpened():
            running[0] = False
            root.destroy()
            return

        ok, frame = perception._cap.read()
        if not ok or frame is None:
            if running[0]:
                root.after(50, tick)
            return

        last_frame[0] = frame

        if sam_requested[0]:
            sam_requested[0] = False
            perception.request_sam2_once()
            print("[control] 本幀將執行 SAM2（需 method: detector_sam2）。")

        out, diff, mask = perception.process_frame_bgr(frame)

        ap = _get_agent_plane(perception)
        ap.update_from_perception(out, frame.shape)

        if agent_clear_requested[0]:
            agent_clear_requested[0] = False
            ap.clear()
            print("[control] 已清除 Agent 平面鎖定。")

        if pending_capture_after_sam[0]:
            pending_capture_after_sam[0] = False
            ok, msg = ap.capture(out, frame.shape)
            print(
                f"[control] 一鍵 SAM＋平面：{msg}"
                if ok
                else f"[control] 一鍵 SAM＋平面失敗：{msg}"
            )

        if agent_capture_requested[0]:
            agent_capture_requested[0] = False
            ok, msg = ap.capture(out, frame.shape)
            print(f"[control] Agent 平面：{msg}" if ok else f"[control] Agent 平面失敗：{msg}")

        now = time.time()
        if now - last_log[0] >= args.interval:
            last_log[0] = now
            _log_line(out, perception)

        if reset_requested[0]:
            perception.set_background(frame)
            reset_requested[0] = False
            print("[control] 已重設背景（基準畫面）。")

        if vlm_requested[0]:
            vlm_requested[0] = False
            ok, detail = _trigger_vlm_multi_or_fallback(perception, frame, out)
            print(
                f"[control] VLM 多物體：{detail}" if ok else f"[control] {detail}"
            )

        if args.preview:
            raw = frame.copy()
            _draw_detections(raw, out, perception)
            _draw_agent_plane_overlay(raw, frame.shape, perception)
            cv2.imshow("raw", raw)
            if diff is not None:
                cv2.imshow("diff", diff)
            if mask is not None:
                cv2.imshow("mask", mask)
            if args.plane_space:
                poly = ap.current_polygon_plane()
                cv2.imshow(
                    "plane_space",
                    render_plane_space_preview(poly, canvas_wh=(960, 720)),
                )
            key = cv2.waitKey(1) & 0xFF
            if key not in (0, 255) and _handle_preview_hotkey(
                key, perception, frame, out, ap
            ):
                on_quit()
                return
            while tk_key_queue:
                sym = tk_key_queue.popleft()
                if _handle_preview_hotkey(
                    ord(sym), perception, frame, out, ap, log_prefix="[tk]"
                ):
                    on_quit()
                    return

        if running[0]:
            root.after(15, tick)

    print(
        "Phase 2 debug: 控制視窗與 OpenCV 皆可用快捷鍵（見視窗內說明）；"
        "另開 plane_space 顯示 2D 平面。"
    )
    root.after(0, tick)
    try:
        root.mainloop()
    finally:
        running[0] = False
        if args.preview:
            cv2.destroyAllWindows()
        perception.stop_stream()


def _log_line(out, perception: CameraShadowPerception | None = None) -> None:
    sm = out.shadow_mask
    extra = (sm.extra if sm else {}) or {}
    line = (
        f"object_visible={out.object_visible!s:5}  "
        f"contours={extra.get('contour_count', '—')}"
    )
    if extra.get("detected_count") is not None:
        line += f"  detected={extra.get('detected_count')}"
        dets = extra.get("detections") or []
        if dets:
            names = [str(d.get("class_name", "?")) for d in dets[:3]]
            tids = [str(d.get("track_id", "-")) for d in dets[:3]]
            line += f"  classes={names}  track_ids={tids}"
    if extra.get("sam2_error"):
        line += f"  sam2_error={extra['sam2_error']!r}"
    elif extra.get("sam2_note"):
        line += f"  sam2_note={extra['sam2_note']!r}"
    elif extra.get("reason"):
        line += f"  reason={extra['reason']!r}"
    if extra.get("sam2_ran"):
        line += "  sam2_ran=True"
    if extra.get("sam2_masks_applied") is not None:
        line += f"  sam2_masks={extra.get('sam2_masks_applied')}"
    if perception is not None:
        ap = _get_agent_plane(perception)
        if ap.active and ap.state is not None:
            u, v = ap.state.current_centroid_uv
            line += f"  agent_uv=({u:.3f},{v:.3f})"
    if out.semantic and out.semantic.description:
        line += f"  semantic={out.semantic.description!r}"
    elif out.semantic and out.semantic.extra.get("vlm_error"):
        line += f"  vlm_error={out.semantic.raw_response!r}"
    if perception is not None:
        snap = perception.snapshot_vlm_semantics_by_track()
        if snap:
            parts = []
            for tid in sorted(snap.keys())[:6]:
                lab = snap[tid]
                if lab.description:
                    parts.append(f"{tid}:{_truncate_overlay(lab.description, 28)!r}")
                elif lab.extra.get("vlm_error"):
                    parts.append(f"{tid}:err")
            if parts:
                line += f"  vlm_by_id=[{', '.join(parts)}]"
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug Phase 2 perception loop")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="OpenCV windows: raw | diff | mask",
    )
    parser.add_argument(
        "--control-panel",
        action="store_true",
        help="Tk panel: reset background + quit (without --preview: console only)",
    )
    parser.add_argument(
        "--no-tk",
        action="store_true",
        help="With --preview: no control window, only keyboard [b] [q]",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="Seconds between console logs (default 0.2)",
    )
    parser.add_argument(
        "--no-plane-space",
        action="store_true",
        help="With --preview: do not open the plane_space debug window",
    )
    args = parser.parse_args()
    args.plane_space = not args.no_plane_space

    perception = CameraShadowPerception()
    perception.start_stream()

    use_tk = (args.preview and not args.no_tk) or args.control_panel

    if use_tk:
        run_tk_loop(perception, args)
    else:
        run_sync_loop(perception, args)


if __name__ == "__main__":
    main()
