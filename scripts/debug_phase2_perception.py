"""
Phase 2: tune CameraShadowPerception — console stats + optional mask preview.

Start with an EMPTY scene (no object on table), run once to lock background via start_stream,
or use the control panel button / key [b] to re-grab background.

Usage:
  python -m scripts.debug_phase2_perception
  python -m scripts.debug_phase2_perception --preview
  python -m scripts.debug_phase2_perception --preview --no-tk   # OpenCV only, keys [b][q]
  python -m scripts.debug_phase2_perception --control-panel    # console + buttons, no OpenCV windows
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2

from peter_pan.perception.camera_shadow import CameraShadowPerception


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
        "Keys (with --preview): [b] new background, [v] VLM 多物體（每 track_id 一次）, [q] quit. Ctrl+C stops."
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

            now = time.time()
            if now - last_log >= args.interval:
                last_log = now
                _log_line(out, perception)

            if args.preview:
                raw = frame.copy()
                _draw_detections(raw, out, perception)
                cv2.imshow("raw", raw)
                if diff is not None:
                    cv2.imshow("diff", diff)
                if mask is not None:
                    cv2.imshow("mask", mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("b"):
                    perception.set_background(frame)
                    print("[preview] Background updated from current frame.")
                if key == ord("v"):
                    ok, detail = _trigger_vlm_multi_or_fallback(
                        perception, frame, out
                    )
                    print(f"[preview] VLM 多物體：{detail}" if ok else f"[preview] {detail}")
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
    running = [True]

    root = tk.Tk()
    root.title("Peter Pan — Phase 2 除錯")
    root.resizable(True, False)

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill=tk.X)

    hint = (
        "「重設背景」＝把目前畫面當成新的空場景基準（背景差分用）。\n"
        "請在桌面清空時按，或放好固定場景後再按。"
    )
    ttk.Label(frm, text=hint, wraplength=360).pack(anchor=tk.W, pady=(0, 8))

    def on_reset_background() -> None:
        reset_requested[0] = True

    def on_quit() -> None:
        running[0] = False
        root.destroy()

    def on_vlm_once() -> None:
        vlm_requested[0] = True

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
        out, diff, mask = perception.process_frame_bgr(frame)

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
            cv2.imshow("raw", raw)
            if diff is not None:
                cv2.imshow("diff", diff)
            if mask is not None:
                cv2.imshow("mask", mask)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                on_quit()
                return
            if key == ord("b"):
                perception.set_background(frame)
                print("[preview] Background updated from current frame.")
            if key == ord("v"):
                ok, detail = _trigger_vlm_multi_or_fallback(
                    perception, frame, out
                )
                print(
                    f"[preview] VLM 多物體：{detail}" if ok else f"[preview] {detail}"
                )

        if running[0]:
            root.after(15, tick)

    print(
        "Phase 2 debug: object_visible=True means large foreground vs background.\n"
        "控制視窗：按「重設背景」更新基準畫面、按「VLM 多物體辨識」對每個 track_id 各送 1 次；"
        "預覽視窗可用 [b]/[v]/[q]。"
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
    args = parser.parse_args()

    perception = CameraShadowPerception()
    perception.start_stream()

    use_tk = (args.preview and not args.no_tk) or args.control_panel

    if use_tk:
        run_tk_loop(perception, args)
    else:
        run_sync_loop(perception, args)


if __name__ == "__main__":
    main()
