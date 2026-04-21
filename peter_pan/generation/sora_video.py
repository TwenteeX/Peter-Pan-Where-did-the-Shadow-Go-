"""Sora video generation plus threshold-mask frame extraction."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEFAULT_CHARACTER = "cat"
DEFAULT_ACTION = "walking"


@dataclass
class ThresholdMaskResult:
    video_path: Path
    output_dir: Path
    frame_count: int
    mask_dir: Path
    rgba_dir: Path
    silhouette_dir: Path
    contact_sheet_path: Optional[Path]


def build_shadow_loop_prompt(
    *,
    character: str,
    action: str,
    travel_across_frame: bool,
) -> str:
    character = (character or DEFAULT_CHARACTER).strip()
    action = (action or DEFAULT_ACTION).strip()
    if travel_across_frame:
        motion_line = (
            f"The {character} moves from left to right inside the frame with smooth, natural timing. "
            "Keep the whole body visible for the entire clip with generous padding at both edges."
        )
    else:
        motion_line = (
            f"The {character} performs the {action} cycle in place, facing right. "
            "The feet, legs, tail, and body move naturally, but the character's overall position "
            "stays centered so an external program can move the sprite across the table."
        )

    return (
        f"A clean, minimalist animation of one {character} shadow for an interactive tabletop projection.\n"
        f"The {character} is represented as a solid black silhouette, completely filled: "
        "no texture, no internal details, no gradients, no highlights.\n"
        "Strict flat 2D side view, vector-like appearance, crisp but not jagged edges.\n\n"
        "Background must be either fully transparent if alpha video is supported, "
        "or pure white if transparency is not supported, so threshold background removal is easy.\n"
        "No shadows, no lighting, no floor line, no environment, no additional objects, no text, no labels.\n\n"
        "The animation must be a perfect seamless loop:\n"
        "- the first frame and last frame must match exactly;\n"
        "- the motion must loop continuously with no visible jump;\n"
        "- the timing must be periodic and evenly paced;\n"
        "- the scale and bounding box must stay stable across frames.\n\n"
        f"{motion_line}\n\n"
        "Centered composition with the full body visible in every frame. "
        "Use high resolution and keep the silhouette large enough for clean masking, "
        "but leave transparent or white padding around the full body."
    )


def write_prompt_file(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt + "\n", encoding="utf-8")


def generate_sora_video(
    *,
    prompt: str,
    output_dir: Path,
    model: str,
    seconds: str,
    size: str,
    input_reference: Optional[Path],
) -> Path:
    """Create a Sora video job, wait for completion, and download the mp4."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set; refusing to call Sora.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai first: pip install openai") from exc

    client = OpenAI()
    if not hasattr(client, "videos"):
        raise RuntimeError("This openai package does not expose client.videos; upgrade openai.")

    output_dir.mkdir(parents=True, exist_ok=True)
    create_kwargs = {
        "model": model,
        "prompt": prompt,
        "seconds": seconds,
        "size": size,
    }
    if input_reference is not None:
        create_kwargs["input_reference"] = input_reference

    print("[sora] starting video job...")
    video = client.videos.create(**create_kwargs)
    print(f"[sora] job id={video.id} status={video.status}")

    while video.status in ("queued", "in_progress"):
        time.sleep(2.0)
        video = client.videos.retrieve(video.id)
        progress = getattr(video, "progress", 0) or 0
        print(f"[sora] status={video.status} progress={progress}%")

    if video.status != "completed":
        err = getattr(video, "error", None)
        message = getattr(err, "message", None) if err is not None else None
        raise RuntimeError(message or f"Sora video job ended with status={video.status}")

    out_path = output_dir / f"{video.id}.mp4"
    print(f"[sora] downloading video -> {out_path}")
    content = client.videos.download_content(video.id, variant="video")
    content.write_to_file(str(out_path))
    return out_path


def extract_threshold_mask_frames(
    video_path: Path,
    output_dir: Path,
    *,
    threshold: int,
    foreground: str,
    frame_stride: int,
    max_frames: int,
    blur: int,
    morph_kernel: int,
    soften_alpha: int,
    keep_largest: bool,
) -> ThresholdMaskResult:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    mask_dir = output_dir / "masks"
    rgba_dir = output_dir / "rgba_frames"
    silhouette_dir = output_dir / "black_silhouette_frames"
    for path in (mask_dir, rgba_dir, silhouette_dir):
        path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    source_idx = 0
    contact_frames: list[np.ndarray] = []
    stride = max(1, int(frame_stride))
    limit = max(1, int(max_frames))

    try:
        while saved < limit:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if source_idx % stride != 0:
                source_idx += 1
                continue

            mask, alpha = threshold_shadow_mask(
                frame,
                threshold=threshold,
                foreground=foreground,
                blur=blur,
                morph_kernel=morph_kernel,
                soften_alpha=soften_alpha,
                keep_largest=keep_largest,
            )
            rgba = apply_alpha_to_source(frame, alpha)
            silhouette = black_silhouette(alpha)

            stem = f"{saved:03d}"
            cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
            cv2.imwrite(str(rgba_dir / f"{stem}.png"), rgba)
            cv2.imwrite(str(silhouette_dir / f"{stem}.png"), silhouette)
            if len(contact_frames) < 12:
                contact_frames.append(_contact_preview_tile(frame, mask, silhouette))
            saved += 1
            source_idx += 1
    finally:
        cap.release()

    contact_sheet = None
    if contact_frames:
        contact_sheet = output_dir / "contact_sheet.png"
        cv2.imwrite(str(contact_sheet), make_contact_sheet(contact_frames, cols=3))

    return ThresholdMaskResult(
        video_path=video_path,
        output_dir=output_dir,
        frame_count=saved,
        mask_dir=mask_dir,
        rgba_dir=rgba_dir,
        silhouette_dir=silhouette_dir,
        contact_sheet_path=contact_sheet,
    )


def threshold_shadow_mask(
    frame_bgr: np.ndarray,
    *,
    threshold: int = 170,
    foreground: str = "dark",
    blur: int = 3,
    morph_kernel: int = 5,
    soften_alpha: int = 5,
    keep_largest: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blur = _odd_or_zero(blur)
    if blur > 1:
        gray = cv2.GaussianBlur(gray, (blur, blur), 0)

    mode = cv2.THRESH_BINARY_INV if foreground == "dark" else cv2.THRESH_BINARY
    _ok, mask = cv2.threshold(gray, int(threshold), 255, mode)

    k = _odd_or_zero(morph_kernel)
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if keep_largest:
        mask = largest_component(mask)

    soften = _odd_or_zero(soften_alpha)
    alpha = mask
    if soften > 1:
        alpha = cv2.GaussianBlur(mask, (soften, soften), 0)
    return mask, alpha


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return mask

    best_label = -1
    best_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label

    out = np.zeros_like(mask)
    if best_label >= 0:
        out[labels == best_label] = 255
    return out


def apply_alpha_to_source(frame_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = alpha
    return bgra


def black_silhouette(alpha: np.ndarray) -> np.ndarray:
    out = np.zeros(alpha.shape + (4,), dtype=np.uint8)
    out[:, :, 3] = alpha
    return out


def create_synthetic_shadow_video(
    path: Path,
    *,
    frame_count: int = 36,
    size: tuple[int, int] = (320, 180),
    fps: float = 12.0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create synthetic video: {path}")

    width, height = size
    try:
        for idx in range(frame_count):
            frame = np.full((height, width, 3), 245, dtype=np.uint8)
            t = idx / max(1, frame_count - 1)
            cx = int(width * (0.25 + 0.5 * t))
            cy = int(height * (0.55 + 0.05 * np.sin(t * np.pi * 4)))
            cv2.ellipse(frame, (cx, cy), (24, 42), 0, 0, 360, (4, 4, 4), -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy - 46), 20, (3, 3, 3), -1, cv2.LINE_AA)
            leg = int(16 * np.sin(t * np.pi * 8))
            cv2.line(frame, (cx - 10, cy + 35), (cx - 20 - leg, cy + 65), (3, 3, 3), 8, cv2.LINE_AA)
            cv2.line(frame, (cx + 10, cy + 35), (cx + 20 + leg, cy + 65), (3, 3, 3), 8, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return path


def assert_usable_result(result: ThresholdMaskResult) -> None:
    if result.frame_count <= 0:
        raise RuntimeError("No frames were extracted.")

    first = cv2.imread(str(result.mask_dir / "000.png"), cv2.IMREAD_GRAYSCALE)
    if first is None:
        raise RuntimeError("First mask was not written.")
    coverage = float(np.count_nonzero(first)) / float(first.size)
    if coverage <= 0.001:
        raise RuntimeError("First mask is almost empty; tune --threshold or --foreground.")
    if coverage >= 0.95:
        raise RuntimeError("First mask is almost full; tune --threshold or --foreground.")


def print_result(result: ThresholdMaskResult) -> None:
    print("[mask-test] ok")
    print(f"  source video: {result.video_path}")
    print(f"  frames: {result.frame_count}")
    print(f"  masks: {result.mask_dir}")
    print(f"  transparent source frames: {result.rgba_dir}")
    print(f"  black silhouette frames: {result.silhouette_dir}")
    if result.contact_sheet_path:
        print(f"  contact sheet: {result.contact_sheet_path}")


def make_contact_sheet(frames: list[np.ndarray], *, cols: int = 3) -> np.ndarray:
    thumbs = [_resize_to_tile(frame, (360, 160)) for frame in frames]
    rows = int(np.ceil(len(thumbs) / cols))
    tile_h, tile_w = thumbs[0].shape[:2]
    sheet = np.full((rows * tile_h, cols * tile_w, 3), 235, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        y = row * tile_h
        x = col * tile_w
        sheet[y : y + tile_h, x : x + tile_w] = thumb
    return sheet


def _contact_preview_tile(frame_bgr: np.ndarray, mask: np.ndarray, silhouette_bgra: np.ndarray) -> np.ndarray:
    frame = _resize_to_tile(frame_bgr, (120, 160))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr = _resize_to_tile(mask_bgr, (120, 160))
    rgba_bgr = _alpha_over_white(silhouette_bgra)
    rgba_bgr = _resize_to_tile(rgba_bgr, (120, 160))
    return np.concatenate([frame, mask_bgr, rgba_bgr], axis=1)


def _alpha_over_white(bgra: np.ndarray) -> np.ndarray:
    bgr = bgra[:, :, :3].astype(np.float32)
    alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
    white = np.full_like(bgr, 255)
    return (bgr * alpha + white * (1.0 - alpha)).astype(np.uint8)


def _resize_to_tile(frame_bgr: np.ndarray, tile_wh: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = tile_wh
    h, w = frame_bgr.shape[:2]
    scale = min(tile_w / max(1, w), tile_h / max(1, h))
    resized = cv2.resize(
        frame_bgr,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    out = np.full((tile_h, tile_w, 3), 235, dtype=np.uint8)
    y = (tile_h - resized.shape[0]) // 2
    x = (tile_w - resized.shape[1]) // 2
    out[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return out


def _odd_or_zero(value: int) -> int:
    value = max(0, int(value))
    if value == 0:
        return 0
    return value if value % 2 == 1 else value + 1
