"""
LLM-based Agent Brain: periodically ask GPT-4o where the shadow should move (in cm),
run in a background thread so the main loop stays at 30 fps.
"""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore

from peter_pan.protocols import ActionIntent
from peter_pan.agent_brain.action_animation import normalize_action


def _frame_to_b64_jpeg(frame_bgr: np.ndarray, max_side: int = 512) -> str:
    if cv2 is None:
        raise RuntimeError("OpenCV required")
    h, w = frame_bgr.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        frame_bgr = cv2.resize(
            frame_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extract first JSON object from LLM text (may contain markdown fences)."""
    text = text.strip()
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        candidate = _normalize_jsonish_numbers(m.group(0))
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(_normalize_jsonish_numbers(text))
    except json.JSONDecodeError:
        return None


def _normalize_jsonish_numbers(text: str) -> str:
    """Allow common LLM slip: positive numbers written as +7.4 in JSON."""
    return re.sub(r"(:\s*)\+(\d+(?:\.\d+)?)", r"\1\2", text)


def _detections_summary(detections: List[dict]) -> List[dict]:
    """Compact detection list for the LLM prompt (no raw tensors)."""
    out: List[dict] = []
    for d in detections[:8]:
        entry: Dict[str, Any] = {}
        if d.get("class_name"):
            entry["object"] = str(d["class_name"])
        b = d.get("bounds_xyxy")
        if b and len(b) == 4:
            cx = (float(b[0]) + float(b[2])) / 2
            cy = (float(b[1]) + float(b[3])) / 2
            entry["center_px"] = [round(cx), round(cy)]
        if d.get("center_plane_cm"):
            entry["center_cm"] = d["center_plane_cm"]
        tid = d.get("track_id", -1)
        if int(tid) >= 0:
            entry["track_id"] = int(tid)
        out.append(entry)
    return out


def detections_to_plane(
    detections: List[dict],
    homography_plane: Any,
    frame_wh: Tuple[int, int],
    extent_cm: Tuple[float, float],
) -> List[Dict[str, Any]]:
    """
    Map YOLO detections from pixel coords to plane UV and cm.

    Returns enriched copies with ``center_uv``, ``rect_uv``, ``center_plane_cm``
    (for LLM prompt and plane_space rendering).
    """
    w, h = int(frame_wh[0]), int(frame_wh[1])
    ew, eh = float(extent_cm[0]), float(extent_cm[1])
    out: List[Dict[str, Any]] = []
    for d in detections[:12]:
        d2 = dict(d)
        b = d.get("bounds_xyxy")
        if not b or len(b) != 4:
            out.append(d2)
            continue
        x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        u1, v1 = homography_plane.pixel_to_plane(x1, y1, (w, h))
        u2, v2 = homography_plane.pixel_to_plane(x2, y2, (w, h))
        cu, cv = (u1 + u2) / 2, (v1 + v2) / 2
        d2["center_uv"] = [round(cu, 4), round(cv, 4)]
        d2["rect_uv"] = [round(u1, 4), round(v1, 4), round(u2, 4), round(v2, 4)]
        d2["center_plane_cm"] = [round(cu * ew, 1), round(cv * eh, 1)]
        out.append(d2)
    return out


class LLMAgent:
    """
    Every ``interval_sec`` seconds, sends current state + photo to GPT and
    receives ``{"dx_cm", "dy_cm", "duration_sec", "reason"}`` back.

    Call ``maybe_request(plane_dict, detections, frame_bgr)`` every frame;
    it will skip if still within the throttle window or a request is in-flight.

    Read latest result via ``pop_latest_intent() -> Optional[ActionIntent]``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        ab = config.get("agent_brain", {})
        self._model: str = ab.get("model", "gpt-4o-mini")
        self._interval: float = float(ab.get("interval_sec", 3.0))
        self._max_image_side: int = int(ab.get("max_image_side", 512))
        self._max_tokens: int = int(ab.get("max_tokens", 300))

        we = config.get("world_engine", {}) or {}
        ps = we.get("plane_space", {}) or {}
        extent = ps.get("physical_extent_cm", [60.0, 40.0])
        self._extent_w = float(extent[0]) if len(extent) >= 1 else 60.0
        self._extent_h = float(extent[1]) if len(extent) >= 2 else 40.0

        default_sys = (
            "你是一個影子角色，活在一個 {w}x{h} cm 的平面空間裡。\n"
            "你的位置以 (offset_x_cm, offset_y_cm) 表示，原點在擷取時的位置。\n"
            "正 x = 右，正 y = 下。\n"
            "根據照片裡的物體與環境，決定你下一步要怎麼移動與表演。\n"
            "action 必須是 idle, walk, hop, stretch, hide, climb, inspect, flee 其中之一。\n"
            "只回覆一個 JSON 物件（不要 markdown）：\n"
            '{{"action": "walk", "dx_cm": float, "dy_cm": float, "duration_sec": float, "reason": "簡短說明"}}'
        )
        raw_sys = ab.get("system_prompt", default_sys)
        self._system_prompt: str = raw_sys.format(
            w=self._extent_w, h=self._extent_h
        )
        if '"action"' not in self._system_prompt and "action 必須" not in self._system_prompt:
            self._system_prompt += (
                "\n\n另外，請加入 action 欄位。action 必須是 idle, walk, hop, "
                "stretch, hide, climb, inspect, flee 其中之一。\n"
                '回覆格式：{"action": "walk", "dx_cm": float, "dy_cm": float, '
                '"duration_sec": float, "reason": "簡短說明"}'
            )

        self._character_desc: Optional[str] = None
        self._base_system_prompt: str = self._system_prompt

        self._last_request_time: float = 0.0
        self._inflight: bool = False
        self._lock = threading.Lock()
        self._latest_intent: Optional[ActionIntent] = None
        self._latest_raw: Optional[str] = None
        self._error: Optional[str] = None

    @property
    def character_desc(self) -> Optional[str]:
        return self._character_desc

    def set_character(self, description: str) -> None:
        """VLM 辨識後呼叫：把角色描述注入 system prompt。"""
        self._character_desc = description
        self._system_prompt = (
            f"你的角色是「{description}」的影子。\n" + self._base_system_prompt
        )

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def inflight(self) -> bool:
        return self._inflight

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    @property
    def last_raw_response(self) -> Optional[str]:
        return self._latest_raw

    def pop_latest_intent(self) -> Optional[ActionIntent]:
        """Consume and return the latest ActionIntent from LLM (or None if nothing new)."""
        with self._lock:
            intent = self._latest_intent
            self._latest_intent = None
            return intent

    def maybe_request(
        self,
        plane_dict: dict[str, Any],
        detections: List[dict],
        frame_bgr: np.ndarray,
    ) -> bool:
        """Call every frame; returns True if a new request was dispatched."""
        now = time.time()
        if self._inflight:
            return False
        if now - self._last_request_time < self._interval:
            return False
        self._last_request_time = now
        self._inflight = True
        self._error = None

        plane_snap = dict(plane_dict)
        dets_snap = _detections_summary(detections)
        frame_copy = frame_bgr.copy()

        t = threading.Thread(
            target=self._worker,
            args=(plane_snap, dets_snap, frame_copy),
            daemon=True,
        )
        t.start()
        return True

    def _worker(
        self,
        plane_dict: dict,
        detections_summary: List[dict],
        frame_bgr: np.ndarray,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            self._error = "pip install openai"
            self._inflight = False
            return

        try:
            b64 = _frame_to_b64_jpeg(frame_bgr, self._max_image_side)

            offset_cm = plane_dict.get("manual_offset_cm", [0, 0])
            user_text = (
                f"目前影子位置 offset_cm=({offset_cm[0]:+.1f}, {offset_cm[1]:+.1f})，"
                f"平面大小 {self._extent_w}x{self._extent_h} cm。\n"
                f"偵測到的物件：{json.dumps(detections_summary, ensure_ascii=False)}\n"
                "請決定下一步移動。"
            )

            client = OpenAI()
            resp = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}",
                                },
                            },
                        ],
                    },
                ],
                max_tokens=self._max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            self._latest_raw = raw
            parsed = _extract_json(raw)
            if parsed is None:
                self._error = f"JSON parse failed: {raw[:120]}"
                self._inflight = False
                return

            action = normalize_action(str(parsed.get("action", "walk")))
            dx = float(parsed.get("dx_cm", 0.0))
            dy = float(parsed.get("dy_cm", 0.0))
            dur = max(0.3, float(parsed.get("duration_sec", 1.5)))
            reason = str(parsed.get("reason", ""))

            intent = ActionIntent(
                action=action,
                target_coord=[dx, dy],
                duration_sec=dur,
                params={"reason": reason, "raw": raw},
            )
            with self._lock:
                self._latest_intent = intent

        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
        finally:
            self._inflight = False
