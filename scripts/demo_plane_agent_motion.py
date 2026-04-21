"""
擷取 SAM 輪廓後，在「固定 0–1 平面 + 格線」裡平移；預設用鍵盤 WASD 像遊戲（公分）。

可選 --auto-wave 啟用自動 sin/cos 示範。

Usage:
  python -m scripts.demo_plane_agent_motion --preview
  python -m scripts.demo_plane_agent_motion --preview --auto-wave
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2

from peter_pan.agent_brain.plane_agent import AgentPlaneBinding
from peter_pan.agent_brain.plane_preview_canvas import render_plane_space_preview
from peter_pan.perception.camera_shadow import CameraShadowPerception


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: move captured mask in plane (cm); WASD = game-style",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="必開：顯示 raw/mask/plane_space 並讀取 WASD",
    )
    parser.add_argument(
        "--step-cm",
        type=float,
        default=2.0,
        help="WASD 每按一次平移幾公分，預設 2",
    )
    parser.add_argument(
        "--auto-wave",
        action="store_true",
        help="擷取後自動 sin/cos 繞圈（與手動 WASD 擇一體驗）",
    )
    parser.add_argument(
        "--amp-x",
        type=float,
        default=8.0,
        help="--auto-wave 水平振幅（公分）",
    )
    parser.add_argument(
        "--amp-y",
        type=float,
        default=5.0,
        help="--auto-wave 垂直振幅（公分）",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="--auto-wave 角速度係數",
    )
    parser.add_argument(
        "--no-auto-sam",
        action="store_true",
        help="不自動觸發 SAM（需自行改程式或另開 phase2）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="定時印 manual_offset_cm",
    )
    args = parser.parse_args()

    perception = CameraShadowPerception()
    ap = AgentPlaneBinding(perception.config)
    perception.start_stream()

    sam_fired = False
    captured = False
    t0 = time.time()
    frame_idx = 0
    warned_no_polygon = False
    current_action = "idle"
    facing = 1.0

    print(
        f"physical_extent_cm={ap.physical_extent_cm}（translate_by_cm 比例尺）\n"
        "流程：自動 SAM 一次 → 有 polygon 即擷取 → plane_space 裡形狀會在「整張 0–1 格線」上動。\n"
        "預設：用鍵盤移動（需 --preview）。WASD=平移、1-8=切 action、r=重置偏移、q=結束。\n"
        "actions: 1 idle, 2 walk, 3 hop, 4 stretch, 5 hide, 6 climb, 7 inspect, 8 flee\n"
        "（畫面：上=W 下=S 左=A 右=D，單位：公分）"
    )
    if args.auto_wave:
        print("已開啟 --auto-wave：會覆寫手動位置成自動繞圈。")
    if not args.preview:
        print("未加 --preview：無法用鍵盤；建議加上。")

    try:
        while True:
            if perception._cap is None or not perception._cap.isOpened():
                break
            ok, frame = perception._cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            frame_idx += 1
            if not args.no_auto_sam and not sam_fired:
                perception.request_sam2_once()
                sam_fired = True

            out, _diff, mask = perception.process_frame_bgr(frame)
            ap.update_from_perception(out, frame.shape)

            if not captured:
                sm = out.shadow_mask
                if sm and sm.polygon_xy and len(sm.polygon_xy) >= 3:
                    ok_cap, msg = ap.capture(out, frame.shape)
                    print(f"[capture] {msg}")
                    if ok_cap:
                        captured = True
                elif frame_idx > 180 and not warned_no_polygon:
                    warned_no_polygon = True
                    print(
                        "[demo] 仍無 polygon：請確認畫面中有物體、"
                        "config 為 detector_sam2，且未使用 --no-auto-sam。"
                    )

            if captured and args.auto_wave:
                t = (time.time() - t0) * args.speed
                ox_cm = args.amp_x * math.sin(t)
                oy_cm = args.amp_y * math.cos(t * 0.87)
                ap.set_manual_offset_cm(ox_cm, oy_cm)

            if args.preview:
                if mask is not None:
                    cv2.imshow("mask", mask)
                raw = frame.copy()
                st = (
                    "MODE: auto-wave"
                    if (captured and args.auto_wave)
                    else ("MODE: WASD (cm)" if captured else "WAIT: SAM + capture…")
                )
                cv2.putText(
                    raw,
                    st,
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0) if captured else (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("raw", raw)
                poly = ap.current_polygon_plane()
                cv2.imshow(
                    "plane_space",
                    render_plane_space_preview(
                        poly,
                        canvas_wh=(960, 720),
                        shadow_action=current_action,
                        animate_shadow=True,
                        fill_shadow=True,
                        facing=facing,
                    ),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if captured and not args.auto_wave:
                    step = args.step_cm
                    if key in (ord("w"), ord("W")):
                        ap.translate_by_cm(0.0, -step)
                    elif key in (ord("s"), ord("S")):
                        ap.translate_by_cm(0.0, step)
                    elif key in (ord("a"), ord("A")):
                        ap.translate_by_cm(-step, 0.0)
                        facing = -1.0
                    elif key in (ord("d"), ord("D")):
                        ap.translate_by_cm(step, 0.0)
                        facing = 1.0
                    elif key in (ord("r"), ord("R")):
                        ap.set_manual_offset_cm(0.0, 0.0)
                        print("[demo] offset 已重置為 (0,0) cm")
                    elif key in [ord(str(n)) for n in range(1, 9)]:
                        actions = ["idle", "walk", "hop", "stretch", "hide", "climb", "inspect", "flee"]
                        current_action = actions[int(chr(key)) - 1]
                        print(f"[manual] action={current_action}")

            if args.verbose and captured and frame_idx % 30 == 0:
                ox, oy = ap.manual_offset_cm()
                print(f"[agent] manual_offset_cm=({ox:+.2f}, {oy:+.2f})")

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        if args.preview:
            cv2.destroyAllWindows()
        perception.stop_stream()
        print("結束。")


if __name__ == "__main__":
    main()
