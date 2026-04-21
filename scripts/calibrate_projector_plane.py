"""
Camera -> projector-plane homography calibration.

Run with the projector on the target display, click the projected rectangle's
four corners in the camera preview, then press ``s`` to write config.yaml.

Example:
  python -m scripts.calibrate_projector_plane --monitor 1 --fullscreen
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2
import numpy as np
import yaml

from peter_pan.config import load_config
from peter_pan.render_bridge.projector_window import describe_monitors, setup_projector_window


WINDOW_PROJECTOR = "projector_calibration_target"
WINDOW_CAMERA = "camera_calibration_preview"
CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate camera pixels to projector UV plane.")
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=1920, help="Projector target width.")
    parser.add_argument("--height", type=int, default=1080, help="Projector target height.")
    parser.add_argument("--monitor", type=int, default=None, help="Projector monitor index, often 1 for the second screen.")
    parser.add_argument("--list-monitors", action="store_true", help="Print monitor rectangles and exit.")
    parser.add_argument("--fullscreen", action="store_true", help="Open the calibration target fullscreen.")
    parser.add_argument("--window-x", type=int, default=0)
    parser.add_argument("--window-y", type=int, default=0)
    parser.add_argument("--inset-px", type=int, default=60, help="Inset for the projected calibration rectangle.")
    parser.add_argument("--auto", action="store_true", help="Try to auto-detect the bright projected rectangle.")
    return parser.parse_args()


def _make_target(width: int, height: int, inset: int) -> np.ndarray:
    canvas = np.zeros((max(1, height), max(1, width), 3), dtype=np.uint8)
    inset = max(0, min(inset, min(width, height) // 4))
    x1, y1 = inset, inset
    x2, y2 = width - inset - 1, height - inset - 1
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 6, cv2.LINE_AA)
    marker_r = max(10, min(width, height) // 80)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for idx, (x, y) in enumerate(corners, start=1):
        cv2.circle(canvas, (x, y), marker_r, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(idx),
            (x + marker_r + 8, y + marker_r + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return canvas


def _order_points(points: list[tuple[float, float]]) -> list[list[float]]:
    pts = np.array(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[int(np.argmin(sums))]
    ordered[2] = pts[int(np.argmax(sums))]
    ordered[1] = pts[int(np.argmin(diffs))]
    ordered[3] = pts[int(np.argmax(diffs))]
    return [[float(x), float(y)] for x, y in ordered]


def _auto_detect_projector_quad(frame: np.ndarray) -> Optional[list[list[float]]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold = max(180, int(np.percentile(blurred, 99) * 0.75))
    _ok, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8), iterations=2)
    contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < frame.shape[0] * frame.shape[1] * 0.01:
        return None
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.03 * peri, True)
    if len(approx) == 4:
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
    else:
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        pts = [(float(x), float(y)) for x, y in box]
    return _order_points(pts)


def _draw_preview(frame: np.ndarray, points: list[list[float]], auto_points: Optional[list[list[float]]]) -> np.ndarray:
    out = frame.copy()
    if auto_points:
        arr = np.array(auto_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [arr], True, (255, 180, 0), 2, cv2.LINE_AA)
    if points:
        for idx, (x, y) in enumerate(points):
            cv2.circle(out, (int(x), int(y)), 7, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(out, str(idx + 1), (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if len(points) >= 2:
            arr = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [arr], len(points) == 4, (0, 255, 0), 2, cv2.LINE_AA)

    next_idx = min(len(points), 3)
    msg = (
        "AUTO found: press a to accept, s to save"
        if auto_points and len(points) < 4
        else f"Click {CORNER_NAMES[next_idx]} corner"
        if len(points) < 4
        else "Press s to save, r reset, u undo, q quit"
    )
    cv2.putText(out, msg, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Order: 1 top-left, 2 top-right, 3 bottom-right, 4 bottom-left", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def _write_homography(config_path: Path, image_points: list[list[float]]) -> None:
    cfg = load_config(config_path)
    plane_space = cfg.setdefault("world_engine", {}).setdefault("plane_space", {})
    homography = plane_space.setdefault("homography", {})
    homography["enabled"] = True
    homography["image_points"] = [[round(float(x), 2), round(float(y), 2)] for x, y in image_points]
    homography["plane_points"] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]

    backup = config_path.with_suffix(config_path.suffix + ".bak")
    if config_path.exists():
        shutil.copy2(config_path, backup)
    config_path.write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"[calibration] wrote homography to {config_path}")
    print(f"[calibration] backup: {backup}")
    print(f"[calibration] image_points={homography['image_points']}")


def main() -> None:
    args = _parse_args()
    if args.list_monitors:
        print(describe_monitors())
        return

    cfg = load_config(args.config)
    perception_cfg = cfg.get("perception", {})
    camera_index = int(args.camera_index if args.camera_index is not None else perception_cfg.get("camera_index", 0))
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")
    width = int(perception_cfg.get("image_width", 0) or 0)
    height = int(perception_cfg.get("image_height", 0) or 0)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    target = _make_target(max(1, args.width), max(1, args.height), args.inset_px)
    setup_projector_window(
        WINDOW_PROJECTOR,
        width=args.width,
        height=args.height,
        fullscreen=args.fullscreen,
        window_x=args.window_x,
        window_y=args.window_y,
        monitor=args.monitor,
    )
    cv2.imshow(WINDOW_PROJECTOR, target)
    cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)

    points: list[list[float]] = []
    auto_points: Optional[list[list[float]]] = None

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([float(x), float(y)])
            print(f"[calibration] point {len(points)} {CORNER_NAMES[len(points) - 1]}=({x},{y})")

    cv2.setMouseCallback(WINDOW_CAMERA, on_mouse)
    print("Click corners in camera preview: top-left, top-right, bottom-right, bottom-left.")
    print("Keys: a auto/accept, u undo, r reset, s save, q quit.")

    try:
        while True:
            cv2.imshow(WINDOW_PROJECTOR, target)
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            if args.auto or auto_points is None:
                auto_points = _auto_detect_projector_quad(frame)
            cv2.imshow(WINDOW_CAMERA, _draw_preview(frame, points, auto_points))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                points.clear()
            elif key in (ord("u"), ord("U")) and points:
                points.pop()
            elif key in (ord("a"), ord("A")):
                auto_points = _auto_detect_projector_quad(frame)
                if auto_points:
                    points[:] = auto_points
                    print("[calibration] accepted auto-detected corners")
                else:
                    print("[calibration] auto-detect failed; click corners manually")
            elif key in (ord("s"), ord("S")):
                if len(points) != 4:
                    print("[calibration] need exactly 4 points before saving")
                    continue
                _write_homography(args.config, _order_points([(p[0], p[1]) for p in points]))
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
