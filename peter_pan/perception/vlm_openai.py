"""
Vision-language call for object / scene description (README: VLM + SemanticLabel).
Requires: pip install openai, env OPENAI_API_KEY.
"""

from __future__ import annotations

import base64
from typing import Any, Optional

from peter_pan.protocols import SemanticLabel

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def _frame_to_jpeg_bytes(frame_bgr: "np.ndarray", quality: int = 85) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV required for VLM image encoding")
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def _maybe_crop(
    frame_bgr: "np.ndarray",
    bounds_xyxy: Optional[tuple[float, float, float, float]],
    min_side: int = 32,
) -> "np.ndarray":
    if bounds_xyxy is None or cv2 is None:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (int(bounds_xyxy[0]), int(bounds_xyxy[1]), int(bounds_xyxy[2]), int(bounds_xyxy[3]))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return frame_bgr
    return frame_bgr[y1:y2, x1:x2].copy()


def _maybe_downscale(frame_bgr: "np.ndarray", max_side: int) -> "np.ndarray":
    if cv2 is None or max_side <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return frame_bgr
    scale = max_side / m
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def describe_frame_openai(
    frame_bgr: "np.ndarray",
    config: dict[str, Any],
    *,
    bounds_xyxy: Optional[tuple[float, float, float, float]] = None,
) -> SemanticLabel:
    """
    Send one BGR frame to OpenAI vision; return SemanticLabel for set_semantic_from_vlm().
    """
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("Install openai: pip install openai") from e

    vlm = config.get("perception", {}).get("vlm", {})
    model = vlm.get("model", "gpt-4o")
    prompt = vlm.get(
        "prompt",
        "Identify the main object in this image in one short phrase.",
    )
    use_crop = bool(vlm.get("use_largest_contour_crop", True))
    max_side = int(vlm.get("max_image_side", 1024))

    img = frame_bgr
    if use_crop and bounds_xyxy is not None:
        img = _maybe_crop(frame_bgr, bounds_xyxy)
    img = _maybe_downscale(img, max_side)

    b64 = base64.standard_b64encode(_frame_to_jpeg_bytes(img)).decode("ascii")
    client = OpenAI()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    },
                ],
            }
        ],
        max_tokens=min(int(vlm.get("max_tokens", 200)), 500),
    )

    text = (response.choices[0].message.content or "").strip()
    return SemanticLabel(description=text, raw_response=text, extra={"model": model})
