"""
Probe OpenCV camera indices to find a USB / external camera.

On Windows, tries DirectShow (CAP_DSHOW) first — same as run_shadow_demo.

Usage:
  python -m scripts.list_cameras
  python -m scripts.list_cameras --max 6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import cv2
except ImportError:
    raise SystemExit("pip install opencv-python") from None


def _open_index(index: int):
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10, help="Try indices 0 .. max-1")
    ap.add_argument("--warmup", type=int, default=10, help="Frames to discard per device")
    args = ap.parse_args()

    print("Index | opened | WxH      | mean_brightness | note")
    print("------+--------+----------+-----------------+-----")

    for i in range(args.max):
        cap = _open_index(i)
        if not cap.isOpened():
            print(f"  {i:2d}  |   no   |    —     |       —         | cannot open")
            continue
        for _ in range(args.warmup):
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  {i:2d}  |  yes   |    —     |       —         | read failed")
            continue
        h, w = frame.shape[:2]
        m = float(frame.mean())
        note = ""
        if m < 2.0:
            note = "looks black — try another index or lighting"
        elif m < 15.0:
            note = "very dark"
        else:
            note = "ok"
        print(f"  {i:2d}  |  yes   | {w}x{h} | {m:13.1f} | {note}")

    print()
    print("Set the working index in config.yaml:")
    print("  perception:")
    print("    camera_index: <N>")


if __name__ == "__main__":
    main()
