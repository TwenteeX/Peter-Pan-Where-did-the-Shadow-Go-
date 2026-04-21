"""OpenCV projector-window helpers, including monitor placement."""

from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from typing import List, Optional
from ctypes import wintypes

import cv2


@dataclass(frozen=True)
class MonitorRect:
    """Desktop-space rectangle for one monitor."""

    index: int
    x: int
    y: int
    width: int
    height: int
    is_primary: bool = False

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


def list_monitors() -> List[MonitorRect]:
    """Return monitor rectangles in virtual desktop coordinates."""
    if sys.platform.startswith("win"):
        return _list_windows_monitors()
    return [MonitorRect(index=0, x=0, y=0, width=1920, height=1080, is_primary=True)]


def monitor_by_index(index: Optional[int]) -> Optional[MonitorRect]:
    if index is None:
        return None
    monitors = list_monitors()
    if not monitors:
        return None
    idx = max(0, min(int(index), len(monitors) - 1))
    return monitors[idx]


def setup_projector_window(
    name: str,
    *,
    width: int,
    height: int,
    fullscreen: bool,
    window_x: int = 0,
    window_y: int = 0,
    monitor: Optional[int] = None,
) -> MonitorRect | None:
    """
    Create/place an OpenCV window.

    ``monitor`` is more reliable than guessing ``window_x`` on multi-display
    installs because it uses the OS virtual desktop rectangle directly.
    """
    mon = monitor_by_index(monitor)
    x = mon.x if mon is not None else int(window_x)
    y = mon.y if mon is not None else int(window_y)
    w = mon.width if mon is not None and fullscreen else int(width)
    h = mon.height if mon is not None and fullscreen else int(height)

    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, int(x), int(y))
    cv2.resizeWindow(name, max(1, int(w)), max(1, int(h)))
    cv2.waitKey(80)
    if fullscreen:
        try:
            cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(name, int(x), int(y))
        time.sleep(0.05)
    return mon


def describe_monitors() -> str:
    monitors = list_monitors()
    if not monitors:
        return "No monitors reported by OS."
    return "\n".join(
        (
            f"{m.index}: x={m.x} y={m.y} w={m.width} h={m.height}"
            + (" primary" if m.is_primary else "")
        )
        for m in monitors
    )


def _list_windows_monitors() -> List[MonitorRect]:
    user32 = ctypes.windll.user32

    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.c_ulong),
            ("rcMonitor", RECT),
            ("rcWork", RECT),
            ("dwFlags", ctypes.c_ulong),
        ]

    monitors: List[MonitorRect] = []
    monitor_enum_proc = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    def callback(hmonitor, _hdc, _rect, _data):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
            rect = info.rcMonitor
            monitors.append(
                MonitorRect(
                    index=len(monitors),
                    x=int(rect.left),
                    y=int(rect.top),
                    width=int(rect.right - rect.left),
                    height=int(rect.bottom - rect.top),
                    is_primary=bool(info.dwFlags & 1),
                )
            )
        return 1

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(callback), 0)
    monitors.sort(key=lambda m: (not m.is_primary, m.x, m.y))
    return [
        MonitorRect(i, m.x, m.y, m.width, m.height, m.is_primary)
        for i, m in enumerate(monitors)
    ]
