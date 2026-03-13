"""
Perception Hub interface. Implementations must satisfy this I/O contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from peter_pan.protocols import PerceptionOutput


class PerceptionHubInterface(ABC):
    """
    Input: RGB stream (and optionally depth).
    Output: PerceptionOutput (shadow mask, semantic label, object_visible, optional point cloud).
    """

    @abstractmethod
    def capture_frame(self) -> Optional[PerceptionOutput]:
        """Capture one frame and return perception result. None if capture failed."""
        ...

    @abstractmethod
    def start_stream(self) -> None:
        """Start camera/stream if needed."""
        ...

    @abstractmethod
    def stop_stream(self) -> None:
        """Release camera/stream."""
        ...


def run_perception_loop(
    hub: PerceptionHubInterface,
    on_output,
    *,
    stop_event=None,
):
    """
    Run perception in a loop and call on_output(PerceptionOutput) each time.
    If stop_event is provided (e.g. threading.Event), loop until it is set.
    """
    hub.start_stream()
    try:
        while stop_event is None or not stop_event.is_set():
            out = hub.capture_frame()
            if out is not None:
                on_output(out)
    finally:
        hub.stop_stream()
