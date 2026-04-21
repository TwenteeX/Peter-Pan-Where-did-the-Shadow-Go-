"""OpenAI-backed sprite frame generation and local playback cache."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


SPRITE_SHEET_COLS = 3
SPRITE_SHEET_ROWS = 2
SPRITE_SHEET_FRAME_COUNT = SPRITE_SHEET_COLS * SPRITE_SHEET_ROWS
DEFAULT_FRAMES_PER_ACTION = 8
SPRITE_SHEET_GENERATION_MODE = "single_sheet_postprocessed_png"
SPRITE_FRAME_BASELINE_Y_RATIO = 0.84
SPRITE_FRAME_TARGET_WIDTH_RATIO = 0.62
SPRITE_FRAME_TARGET_HEIGHT_RATIO = 0.66
ALLOW_PARTIAL_OBJECT_CROP = True


@dataclass
class SpriteGenerationRequest:
    """Parameters for creating a reusable shadow sprite cache."""

    character_description: str
    actions: Tuple[str, ...] = ("idle", "walk", "hop", "stretch")
    frames_per_action: int = DEFAULT_FRAMES_PER_ACTION
    model: str = "gpt-image-1"
    size: str = "1536x1024"
    quality: str = "medium"
    use_reference_image: bool = True


class SpriteActionLibrary:
    """Loads cached sprite frames/sheets and exposes one frame per action/time."""

    def __init__(self, root_dir: str | Path, frames_per_action: int = DEFAULT_FRAMES_PER_ACTION) -> None:
        self.root_dir = Path(root_dir)
        self.frames_per_action = int(frames_per_action)
        self._sheets: Dict[str, np.ndarray] = {}
        self._frames: Dict[str, List[np.ndarray]] = {}
        self._metadata: Dict[str, Any] = {}
        self._load_metadata()
        self.ensure_split_frames()

    @property
    def available_actions(self) -> List[str]:
        actions = set(self._frames.keys()) | set(self._sheets.keys())
        if self.root_dir.exists():
            frame_root = self.root_dir / "frames"
            frame_actions = [p.name for p in frame_root.iterdir() if p.is_dir()] if frame_root.exists() else []
            sheet_actions = [
                p.stem for p in self.root_dir.glob("*.png")
                if not re.search(r"_\d{3}$", p.stem)
            ]
            actions.update(frame_actions)
            actions.update(sheet_actions)
        return sorted(actions)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._metadata)

    def get_frame(self, action: str, now_sec: float, fps: float = 8.0) -> Optional[np.ndarray]:
        """Return a BGRA frame for ``action`` or None when no sprite is cached."""
        if cv2 is None:
            return None
        frames = self._load_frames(action)
        if frames:
            idx = int(now_sec * fps) % len(frames)
            return frames[idx].copy()

        sheet = self._load_sheet(action)
        if sheet is None:
            return None
        h, w = sheet.shape[:2]
        frame_count = max(1, int(self._metadata.get("frames_per_action", self.frames_per_action)))
        cols, rows = _sheet_grid(self._metadata, frame_count)
        split_frames = split_sprite_sheet(sheet, frame_count, grid=(cols, rows))
        if split_frames:
            self._frames[action] = split_frames
            idx = int(now_sec * fps) % len(split_frames)
            return split_frames[idx].copy()
        return None

    def ensure_split_frames(self, *, overwrite: bool = False) -> Dict[str, int]:
        """Split existing ``<action>.png`` sheets into ``frames/<action>/<idx>.png``."""
        if cv2 is None or not self.root_dir.exists():
            return {}
        counts: Dict[str, int] = {}
        sheet_entries = _metadata_sheet_entries(self.root_dir, self._metadata)
        for action, sheets, expected_count in sheet_entries:
            frame_dir = self.root_dir / "frames" / action
            existing = sorted(frame_dir.glob("*.png")) if frame_dir.exists() else []
            if len(existing) >= expected_count and not overwrite:
                counts[action] = len(existing)
                continue
            frame_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                for old in frame_dir.glob("*.png"):
                    old.unlink()
            written = 0
            for sheet_idx, sheet_path in enumerate(sheets):
                sheet = self._read_bgra(sheet_path)
                if sheet is None:
                    continue
                cols, rows = _sheet_grid(self._metadata, expected_count)
                cells_per_sheet = cols * rows
                start_idx = sheet_idx * cells_per_sheet
                sheet_frame_count = min(cells_per_sheet, expected_count - start_idx)
                if sheet_frame_count <= 0:
                    continue
                frames = split_sprite_sheet(
                    sheet,
                    sheet_frame_count,
                    grid=(cols, rows),
                    debug_prefix=self.root_dir / f"{action}_sheet{sheet_idx:03d}",
                )
                for local_idx, frame in enumerate(frames):
                    out_idx = start_idx + local_idx
                    cv2.imwrite(str(frame_dir / f"{out_idx:03d}.png"), frame)
                    written += 1
            counts[action] = written
            self._frames.pop(action, None)
        return counts

    def _load_metadata(self) -> None:
        p = self.root_dir / "metadata.json"
        if not p.exists():
            return
        try:
            self._metadata = json.loads(p.read_text(encoding="utf-8"))
            self.frames_per_action = int(self._metadata.get("frames_per_action", self.frames_per_action))
        except Exception:
            self._metadata = {}

    def _load_sheet(self, action: str) -> Optional[np.ndarray]:
        if cv2 is None:
            return None
        action = _safe_name(action)
        if action in self._sheets:
            return self._sheets[action]
        p = self.root_dir / f"{action}.png"
        if not p.exists():
            return None
        img = self._read_bgra(p)
        if img is None:
            return None
        self._sheets[action] = img
        return img

    def _load_frames(self, action: str) -> List[np.ndarray]:
        if cv2 is None:
            return []
        action = _safe_name(action)
        if action in self._frames:
            return self._frames[action]
        frame_dir = self.root_dir / "frames" / action
        if not frame_dir.exists():
            return []
        frames: List[np.ndarray] = []
        for p in sorted(frame_dir.glob("*.png")):
            if p.stem.endswith("_debug_boxes") or p.stem.endswith("_clean"):
                continue
            img = self._read_bgra(p)
            if img is not None:
                frames.append(img)
        if frames:
            self._frames[action] = frames
        return frames

    @staticmethod
    def _read_bgra(path: Path) -> Optional[np.ndarray]:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        if img.shape[2] == 3:
            alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
            return np.concatenate([img, alpha], axis=2)
        return img


def generate_sprite_cache_openai(
    request: SpriteGenerationRequest,
    *,
    output_root: str | Path = "assets/generated_sprites",
    overwrite: bool = False,
) -> SpriteActionLibrary:
    """
    Generate transparent PNG animation frames with OpenAI Images API and cache them.

    If cached frames already exist, this returns the cache without another API call.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for sprite cache handling")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install openai: pip install openai") from exc

    cache_dir = Path(output_root) / _safe_name(request.character_description)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    reference_path = _ensure_reference_image(cache_dir) if request.use_reference_image else None
    frame_count = max(1, int(request.frames_per_action))
    sheet_count = 1
    sheet_cols, sheet_rows = _generation_grid(frame_count)
    existing_frames = [
        cache_dir / "frames" / _safe_name(a) / f"{idx:03d}.png"
        for a in request.actions
        for idx in range(frame_count)
    ]
    existing_sheets = [
        _sheet_path(cache_dir, _safe_name(a), sheet_idx, sheet_count)
        for a in request.actions
        for sheet_idx in range(sheet_count)
    ]
    existing_mode = _read_generation_mode(metadata_path)
    should_reprocess = existing_mode != SPRITE_SHEET_GENERATION_MODE
    if (
        not overwrite
        and not should_reprocess
        and existing_mode == SPRITE_SHEET_GENERATION_MODE
        and all(p.exists() for p in existing_sheets)
        and all(p.exists() for p in existing_frames)
    ):
        return SpriteActionLibrary(cache_dir, frames_per_action=frame_count)

    client = OpenAI()
    created: List[str] = []
    for action in request.actions:
        action_name = _safe_name(action)
        frame_dir = cache_dir / "frames" / action_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        sheet_paths = [
            _sheet_path(cache_dir, action_name, sheet_idx, sheet_count)
            for sheet_idx in range(sheet_count)
        ]
        if overwrite:
            for old in frame_dir.glob("*.png"):
                old.unlink()
            for old_sheet in cache_dir.glob(f"{action_name}_*.png"):
                old_sheet.unlink()
            for sheet_path in sheet_paths:
                if sheet_path.exists():
                    sheet_path.unlink()

        for sheet_idx, sheet_path in enumerate(sheet_paths):
            start_idx = 0
            sheet_frame_count = frame_count
            if sheet_path.exists() and not overwrite:
                created.append(sheet_path.name)
            else:
                prompt = _sheet_prompt(
                    request.character_description,
                    action_name,
                    frame_count,
                    cols=sheet_cols,
                    rows=sheet_rows,
                    start_idx=start_idx,
                    sheet_frame_count=sheet_frame_count,
                    sheet_idx=sheet_idx,
                    sheet_count=sheet_count,
                    has_reference=reference_path is not None,
                )
                resp = _request_sprite_sheet_image(
                    client,
                    request,
                    prompt,
                    reference_path,
                )
                b64 = getattr(resp.data[0], "b64_json", None)
                if not b64:
                    raise RuntimeError("OpenAI image response did not include b64_json")
                sheet_path.write_bytes(base64.b64decode(b64))
                created.append(sheet_path.name)

            sheet = SpriteActionLibrary._read_bgra(sheet_path)
            if sheet is None:
                raise RuntimeError(f"Could not read generated sprite sheet: {sheet_path}")
            frames = split_sprite_sheet(
                sheet,
                sheet_frame_count,
                grid=(sheet_cols, sheet_rows),
                debug_prefix=cache_dir / action_name,
            )
            for local_idx, frame in enumerate(frames):
                frame_idx = start_idx + local_idx
                out_path = frame_dir / f"{frame_idx:03d}.png"
                if out_path.exists() and not overwrite and not should_reprocess:
                    continue
                cv2.imwrite(str(out_path), frame)
            _save_reference_from_frames(cache_dir / "reference.png", frames, overwrite=False)

        if sheet_count > 1:
            legacy_path = cache_dir / f"{action_name}.png"
            if legacy_path.exists() and legacy_path not in sheet_paths:
                legacy_path.unlink()

    sheet_files = {
        _safe_name(action): [
            _sheet_path(cache_dir, _safe_name(action), sheet_idx, sheet_count).name
            for sheet_idx in range(sheet_count)
        ]
        for action in request.actions
    }
    metadata = {
        "character_description": request.character_description,
        "actions": list(request.actions),
        "frames_per_action": frame_count,
        "sheet_count_per_action": sheet_count,
        "frames_per_sheet": frame_count,
        "sheet_grid": [sheet_cols, sheet_rows],
        "sheet_files": sheet_files,
        "frame_postprocess": {
            "remove_background": True,
            "align_anchor": "bottom_center",
            "baseline_y_ratio": SPRITE_FRAME_BASELINE_Y_RATIO,
            "target_width_ratio": SPRITE_FRAME_TARGET_WIDTH_RATIO,
            "target_height_ratio": SPRITE_FRAME_TARGET_HEIGHT_RATIO,
        },
        "model": request.model,
        "size": request.size,
        "quality": request.quality,
        "use_reference_image": request.use_reference_image,
        "reference_image": reference_path.name if reference_path is not None else None,
        "generation_mode": SPRITE_SHEET_GENERATION_MODE,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "created_files": created,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return SpriteActionLibrary(cache_dir, frames_per_action=frame_count)


def _read_generation_mode(metadata_path: Path) -> str:
    if not metadata_path.exists():
        return ""
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return str(data.get("generation_mode", ""))


def _request_sprite_sheet_image(
    client: Any,
    request: SpriteGenerationRequest,
    prompt: str,
    reference_path: Optional[Path],
):
    if reference_path is not None and reference_path.exists():
        try:
            with reference_path.open("rb") as ref_file:
                return client.images.edit(
                    model=request.model,
                    image=ref_file,
                    prompt=prompt,
                    size=request.size,
                    quality=request.quality,
                    background="transparent",
                    input_fidelity="high",
                    n=1,
                )
        except TypeError:
            with reference_path.open("rb") as ref_file:
                return client.images.edit(
                    model=request.model,
                    image=ref_file,
                    prompt=prompt,
                    size=request.size,
                    quality=request.quality,
                    background="transparent",
                    n=1,
                )
        except Exception:
            pass

    try:
        return client.images.generate(
            model=request.model,
            prompt=prompt,
            size=request.size,
            quality=request.quality,
            background="transparent",
            n=1,
        )
    except TypeError:
        return client.images.generate(
            model=request.model,
            prompt=prompt,
            size=request.size,
            quality=request.quality,
            n=1,
        )


def _ensure_reference_image(cache_dir: Path) -> Optional[Path]:
    reference_path = cache_dir / "reference.png"
    if reference_path.exists():
        return reference_path

    candidates: List[np.ndarray] = []
    frame_root = cache_dir / "frames"
    if frame_root.exists():
        for p in sorted(frame_root.glob("*/*.png")):
            if p.stem.endswith("_debug_boxes") or p.stem.endswith("_clean"):
                continue
            img = SpriteActionLibrary._read_bgra(p) if cv2 is not None else None
            if img is not None:
                candidates.append(img)

    if _save_reference_from_frames(reference_path, candidates, overwrite=True):
        return reference_path
    return None


def _save_reference_from_frames(reference_path: Path, frames: List[np.ndarray], *, overwrite: bool) -> bool:
    if cv2 is None or not frames:
        return False
    if reference_path.exists() and not overwrite:
        return True
    best = _best_reference_frame(frames)
    if best is None:
        return False
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reference_path), best)
    return True


def _best_reference_frame(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    best_frame: Optional[np.ndarray] = None
    best_score = -1.0
    for frame in frames:
        if frame.ndim != 3 or frame.shape[2] < 4:
            continue
        box = _alpha_bbox(frame)
        if box is None:
            continue
        alpha = frame[:, :, 3]
        area = float(np.count_nonzero(alpha > 8))
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        touches_edge = x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1
        score = area * (0.5 if touches_edge else 1.0)
        if score > best_score:
            best_score = score
            best_frame = frame
    return best_frame.copy() if best_frame is not None else None


def _generation_grid(frame_count: int) -> Tuple[int, int]:
    count = max(1, int(frame_count))
    if count <= 1:
        return (1, 1)
    if count <= 2:
        return (2, 1)
    if count <= 4:
        return (2, 2)
    if count <= SPRITE_SHEET_FRAME_COUNT:
        return (SPRITE_SHEET_COLS, SPRITE_SHEET_ROWS)
    if count <= 8:
        return (4, 2)
    if count <= 12:
        return (4, 3)
    cols = int(np.ceil(np.sqrt(count * 1.5)))
    rows = int(np.ceil(count / cols))
    return (max(1, cols), max(1, rows))


def _sheet_count(frame_count: int) -> int:
    count = max(1, int(frame_count))
    return max(1, (count + SPRITE_SHEET_FRAME_COUNT - 1) // SPRITE_SHEET_FRAME_COUNT)


def _sheet_path(root_dir: Path, action_name: str, sheet_idx: int, sheet_count: int) -> Path:
    if sheet_count <= 1:
        return root_dir / f"{action_name}.png"
    return root_dir / f"{action_name}_{sheet_idx:03d}.png"


def _metadata_sheet_entries(root_dir: Path, metadata: Dict[str, Any]) -> List[Tuple[str, List[Path], int]]:
    sheet_files = metadata.get("sheet_files")
    if isinstance(sheet_files, dict):
        entries: List[Tuple[str, List[Path], int]] = []
        expected_count = max(1, int(metadata.get("frames_per_action", SPRITE_SHEET_FRAME_COUNT)))
        for raw_action, raw_files in sheet_files.items():
            action = _safe_name(str(raw_action))
            if not isinstance(raw_files, list):
                continue
            files = [root_dir / str(name) for name in raw_files]
            files = [p for p in files if p.exists()]
            if files:
                entries.append((action, files, expected_count))
        if entries:
            return entries

    frame_count = max(1, int(metadata.get("frames_per_action", SPRITE_SHEET_FRAME_COUNT)))
    return [
        (_safe_name(path.stem), [path], frame_count)
        for path in sorted(root_dir.glob("*.png"))
        if not re.search(r"_\d{3}$", path.stem)
    ]


def split_sprite_sheet(
    sheet_bgra: np.ndarray,
    frame_count: int,
    *,
    grid: Optional[Tuple[int, int]] = None,
    debug_prefix: Optional[Path] = None,
) -> List[np.ndarray]:
    """Split a sprite sheet into BGRA frames, preferring detected pose crops."""
    count = max(1, int(frame_count))
    cols, rows = grid or (count, 1)
    detected_boxes = _detect_pose_boxes(sheet_bgra, count)
    used_detection = len(detected_boxes) > 0
    if used_detection:
        boxes = detected_boxes[:count]
        frames = [
            _frame_from_detected_box(sheet_bgra, box, cols, rows)
            for box in boxes
        ]
    else:
        if debug_prefix is not None:
            _write_split_debug_images(
                debug_prefix,
                sheet_bgra,
                [],
                detected_boxes,
                count,
                cols,
                rows,
                False,
            )
        raise RuntimeError(
            "Object crop failed: detected 0 pose candidates. "
            "Refusing to use fixed-grid crop."
        )

    frames = _align_frames_bottom_center(frames)
    if debug_prefix is not None:
        _write_split_debug_images(
            debug_prefix,
            sheet_bgra,
            frames,
            boxes,
            count,
            cols,
            rows,
            used_detection,
        )
    return frames


def _sheet_grid(metadata: Dict[str, Any], frame_count: int) -> Tuple[int, int]:
    raw = metadata.get("sheet_grid")
    if isinstance(raw, list) and len(raw) == 2:
        try:
            cols = max(1, int(raw[0]))
            rows = max(1, int(raw[1]))
            if cols * rows >= max(1, int(frame_count)):
                return cols, rows
        except Exception:
            pass
    return (max(1, int(frame_count)), 1)


def _cell_from_sheet(
    sheet_bgra: np.ndarray,
    frame_idx: int,
    frame_count: int,
    cols: int,
    rows: int,
) -> np.ndarray:
    h, w = sheet_bgra.shape[:2]
    count = max(1, int(frame_count))
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    idx = max(0, min(count - 1, int(frame_idx)))
    col = idx % cols
    row = idx // cols
    cell_w = max(1, w // cols)
    cell_h = max(1, h // rows)
    x1 = min(w - 1, col * cell_w)
    y1 = min(h - 1, row * cell_h)
    x2 = w if col == cols - 1 else min(w, x1 + cell_w)
    y2 = h if row == rows - 1 else min(h, y1 + cell_h)
    return sheet_bgra[y1:y2, x1:x2].copy()


def _detect_pose_boxes(sheet_bgra: np.ndarray, expected_count: int) -> List[Tuple[int, int, int, int]]:
    if cv2 is None or sheet_bgra.ndim != 3:
        return []
    mask = _alpha_pose_mask(sheet_bgra)
    if mask is None:
        mask = _local_dark_pose_mask(sheet_bgra)
    if mask is None:
        return []
    return _boxes_from_pose_mask(mask, expected_count)


def _alpha_pose_mask(sheet_bgra: np.ndarray) -> Optional[np.ndarray]:
    if sheet_bgra.ndim != 3 or sheet_bgra.shape[2] < 4 or cv2 is None:
        return None
    alpha = sheet_bgra[:, :, 3]
    opaque = alpha > 8
    transparent = alpha < 250
    if np.count_nonzero(transparent) <= alpha.size * 0.05:
        return None
    if np.count_nonzero(opaque) <= alpha.size * 0.002:
        return None
    mask = opaque.astype(np.uint8) * 255
    close_kernel = np.ones((7, 7), dtype=np.uint8)
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


def _local_dark_pose_mask(sheet_bgra: np.ndarray) -> Optional[np.ndarray]:
    if sheet_bgra.ndim != 3 or cv2 is None:
        return None
    bgr = sheet_bgra[:, :, :3]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=35, sigmaY=35)
    local_dark = bg.astype(np.int16) - blurred.astype(np.int16)
    mask = ((local_dark >= 8) & (blurred <= 120)).astype(np.uint8) * 255
    close_kernel = np.ones((13, 13), dtype=np.uint8)
    open_kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


def _boxes_from_pose_mask(mask: np.ndarray, expected_count: int) -> List[Tuple[int, int, int, int]]:
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    h, w = mask.shape[:2]
    image_area = h * w
    min_area = max(80, int(image_area * 0.002))
    max_area = int(image_area * 0.25)
    boxes: List[Tuple[int, int, int, int, int]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        if bw < w * 0.04 or bh < h * 0.05:
            continue
        if bw > w * 0.9 or bh > h * 0.9:
            continue
        boxes.append((x, y, x + bw, y + bh, area))

    if len(boxes) < expected_count:
        contour_boxes = _contour_pose_boxes(mask, expected_count)
        boxes = _merge_box_candidates(boxes, contour_boxes)
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)[:expected_count]
    boxes_xyxy = [(x1, y1, x2, y2) for x1, y1, x2, y2, _area in boxes]
    return _sort_pose_boxes_reading_order(boxes_xyxy)


def _contour_pose_boxes(mask: np.ndarray, expected_count: int) -> List[Tuple[int, int, int, int, int]]:
    if cv2 is None:
        return []
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    image_area = h * w
    min_area = max(80, int(image_area * 0.002))
    boxes: List[Tuple[int, int, int, int, int]] = []
    for contour in contours:
        area = int(cv2.contourArea(contour))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < w * 0.035 or bh < h * 0.04:
            continue
        boxes.append((int(x), int(y), int(x + bw), int(y + bh), area))
    return sorted(boxes, key=lambda b: b[4], reverse=True)[:expected_count]


def _merge_box_candidates(
    first: List[Tuple[int, int, int, int, int]],
    second: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    merged: List[Tuple[int, int, int, int, int]] = []
    for box in first + second:
        if all(_box_iou(box[:4], existing[:4]) < 0.7 for existing in merged):
            merged.append(box)
    return merged


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _sort_pose_boxes_reading_order(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    heights = [max(1, y2 - y1) for _x1, y1, _x2, y2 in boxes]
    row_tol = max(10.0, float(np.median(heights)) * 0.55)
    rows: List[List[Tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda b: (b[1] + b[3]) * 0.5):
        cy = (box[1] + box[3]) * 0.5
        placed = False
        for row in rows:
            row_cy = np.mean([(b[1] + b[3]) * 0.5 for b in row])
            if abs(cy - row_cy) <= row_tol:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    ordered: List[Tuple[int, int, int, int]] = []
    for row in sorted(rows, key=lambda r: np.mean([(b[1] + b[3]) * 0.5 for b in r])):
        ordered.extend(sorted(row, key=lambda b: (b[0] + b[2]) * 0.5))
    return ordered


def _frame_from_detected_box(
    sheet_bgra: np.ndarray,
    box: Tuple[int, int, int, int],
    cols: int,
    rows: int,
) -> np.ndarray:
    h, w = sheet_bgra.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad = max(8, int(max(bw, bh) * 0.16))
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    crop = sheet_bgra[cy1:cy2, cx1:cx2].copy()
    crop = _remove_sprite_background(crop)
    cell_w = max(1, w // max(1, int(cols)))
    cell_h = max(1, h // max(1, int(rows)))
    return _center_crop_on_canvas(crop, cell_w, cell_h)


def _center_crop_on_canvas(frame_bgra: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, 4), dtype=np.uint8)
    box = _alpha_bbox(frame_bgra)
    if box is None:
        return out
    x1, y1, x2, y2 = box
    cropped = frame_bgra[y1:y2, x1:x2].copy()
    ch, cw = cropped.shape[:2]
    max_w = max(1, int(width * 0.9))
    max_h = max(1, int(height * 0.9))
    scale = min(max_w / max(1, cw), max_h / max(1, ch), 1.0)
    if scale < 0.999 and cv2 is not None:
        cropped = cv2.resize(
            cropped,
            (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        ch, cw = cropped.shape[:2]
    x = max(0, (width - cw) // 2)
    y = max(0, int(round(height * SPRITE_FRAME_BASELINE_Y_RATIO)) - ch)
    x2_out = min(width, x + cw)
    y2_out = min(height, y + ch)
    if x >= x2_out or y >= y2_out:
        return out
    out[y:y2_out, x:x2_out] = cropped[: y2_out - y, : x2_out - x]
    return out


def _remove_sprite_background(frame_bgra: np.ndarray) -> np.ndarray:
    """Convert gray/opaque generated backdrops into transparent black silhouettes."""
    if frame_bgra.ndim != 3 or frame_bgra.shape[2] < 4 or cv2 is None:
        return frame_bgra

    bgr = frame_bgra[:, :, :3]
    alpha = frame_bgra[:, :, 3]
    existing_alpha = alpha if np.count_nonzero(alpha < 250) > alpha.size * 0.05 else None

    if existing_alpha is not None:
        mask = (existing_alpha > 8).astype(np.uint8) * 255
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        dark_level = float(np.percentile(blurred, 1.0))
        threshold = int(max(24, min(72, dark_level + 34)))
        mask = (blurred <= threshold).astype(np.uint8) * 255
        mask = _largest_connected_component(mask)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = _largest_connected_component(mask)

    soft_alpha = cv2.GaussianBlur(mask, (5, 5), 0)
    out = np.zeros_like(frame_bgra)
    out[:, :, 3] = soft_alpha
    return out


def _largest_connected_component(mask: np.ndarray) -> np.ndarray:
    if cv2 is None:
        return mask
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return mask
    h, w = mask.shape[:2]
    labels_to_score: List[Tuple[int, int]] = []
    fallback: List[Tuple[int, int]] = []
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches_border = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
        fallback.append((area, label))
        if not touches_border:
            labels_to_score.append((area, label))
    if not labels_to_score:
        labels_to_score = fallback
    largest_label = max(labels_to_score)[1]
    return ((labels == largest_label).astype(np.uint8) * 255)


def _align_frames_bottom_center(frames: List[np.ndarray]) -> List[np.ndarray]:
    frames = [_normalize_silhouette_size(frame) for frame in frames]
    boxes = [_alpha_bbox(frame) for frame in frames]
    valid = [box for box in boxes if box is not None]
    if not frames or not valid:
        return frames

    h, w = frames[0].shape[:2]
    target_x = w * 0.5
    target_y = h * SPRITE_FRAME_BASELINE_Y_RATIO
    aligned: List[np.ndarray] = []
    for frame, box in zip(frames, boxes):
        if box is None:
            aligned.append(frame)
            continue
        x1, y1, x2, y2 = box
        anchor_x = (x1 + x2 - 1) * 0.5
        anchor_y = float(y2 - 1)
        dx = int(round(target_x - anchor_x))
        dy = int(round(target_y - anchor_y))
        aligned.append(_translate_frame(frame, dx, dy))
    return aligned


def _normalize_silhouette_size(frame_bgra: np.ndarray) -> np.ndarray:
    box = _alpha_bbox(frame_bgra)
    if box is None:
        return frame_bgra
    h, w = frame_bgra.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    target_w = max(1.0, w * SPRITE_FRAME_TARGET_WIDTH_RATIO)
    target_h = max(1.0, h * SPRITE_FRAME_TARGET_HEIGHT_RATIO)
    scale = min(target_w / bw, target_h / bh)
    if abs(scale - 1.0) < 0.03:
        return frame_bgra

    scaled_w = max(1, int(round(w * scale)))
    scaled_h = max(1, int(round(h * scale)))
    if cv2 is None:
        return frame_bgra
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled = cv2.resize(frame_bgra, (scaled_w, scaled_h), interpolation=interpolation)

    scaled_box = _alpha_bbox(scaled)
    if scaled_box is None:
        return frame_bgra
    sx1, sy1, sx2, sy2 = scaled_box
    src_anchor_x = (sx1 + sx2 - 1) * 0.5
    src_anchor_y = float(sy2 - 1)
    target_x = w * 0.5
    target_y = h * SPRITE_FRAME_BASELINE_Y_RATIO
    dx = int(round(target_x - src_anchor_x))
    dy = int(round(target_y - src_anchor_y))

    out = np.zeros_like(frame_bgra)
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(scaled_w, w - dx)
    src_y2 = min(scaled_h, h - dy)
    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = dst_x1 + max(0, src_x2 - src_x1)
    dst_y2 = dst_y1 + max(0, src_y2 - src_y1)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return frame_bgra
    out[dst_y1:dst_y2, dst_x1:dst_x2] = scaled[src_y1:src_y2, src_x1:src_x2]
    return out


def _alpha_bbox(frame_bgra: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if frame_bgra.ndim != 3 or frame_bgra.shape[2] < 4:
        return None
    alpha = frame_bgra[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def _translate_frame(frame_bgra: np.ndarray, dx: int, dy: int) -> np.ndarray:
    if dx == 0 and dy == 0:
        return frame_bgra.copy()
    h, w = frame_bgra.shape[:2]
    out = np.zeros_like(frame_bgra)
    src_x1 = max(0, -dx)
    src_y1 = max(0, -dy)
    src_x2 = min(w, w - dx) if dx >= 0 else w
    src_y2 = min(h, h - dy) if dy >= 0 else h
    dst_x1 = max(0, dx)
    dst_y1 = max(0, dy)
    dst_x2 = dst_x1 + max(0, src_x2 - src_x1)
    dst_y2 = dst_y1 + max(0, src_y2 - src_y1)
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return out
    out[dst_y1:dst_y2, dst_x1:dst_x2] = frame_bgra[src_y1:src_y2, src_x1:src_x2]
    return out


def _write_split_debug_images(
    debug_prefix: Path,
    sheet_bgra: np.ndarray,
    frames: List[np.ndarray],
    boxes: List[Tuple[int, int, int, int]],
    frame_count: int,
    cols: int,
    rows: int,
    used_detection: bool,
) -> None:
    if cv2 is None:
        return
    debug_prefix.parent.mkdir(parents=True, exist_ok=True)
    debug = _debug_canvas(sheet_bgra)
    color = (0, 255, 0) if used_detection else (0, 180, 255)
    if boxes:
        for idx, (x1, y1, x2, y2) in enumerate(boxes[:frame_count]):
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                debug,
                str(idx),
                (x1 + 4, max(18, y1 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    else:
        h, w = sheet_bgra.shape[:2]
        cell_w = max(1, w // max(1, cols))
        cell_h = max(1, h // max(1, rows))
        for idx in range(frame_count):
            col = idx % cols
            row = idx // cols
            x1 = col * cell_w
            y1 = row * cell_h
            x2 = w if col == cols - 1 else min(w, x1 + cell_w)
            y2 = h if row == rows - 1 else min(h, y1 + cell_h)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            cv2.putText(debug, str(idx), (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if used_detection and len(boxes) < frame_count:
        label = f"object crop partial {len(boxes)}/{frame_count}"
    elif used_detection:
        label = "object crop"
    else:
        label = "object crop failed"
    cv2.putText(debug, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(debug_prefix.with_name(debug_prefix.name + "_debug_boxes.png")), debug)

    if frames:
        clean = _repack_frames(frames, cols, rows)
        cv2.imwrite(str(debug_prefix.with_name(debug_prefix.name + "_clean.png")), clean)


def _debug_canvas(sheet_bgra: np.ndarray) -> np.ndarray:
    if sheet_bgra.ndim != 3:
        return sheet_bgra.copy()
    bgr = sheet_bgra[:, :, :3].copy()
    if sheet_bgra.shape[2] >= 4:
        alpha = sheet_bgra[:, :, 3:4].astype(np.float32) / 255.0
        bg = np.full_like(bgr, 230)
        bgr = (bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return bgr


def _repack_frames(frames: List[np.ndarray], cols: int, rows: int) -> np.ndarray:
    cell_h, cell_w = frames[0].shape[:2]
    out = np.zeros((cell_h * rows, cell_w * cols, 4), dtype=np.uint8)
    for idx, frame in enumerate(frames[: cols * rows]):
        row = idx // cols
        col = idx % cols
        y1 = row * cell_h
        x1 = col * cell_w
        h = min(cell_h, frame.shape[0])
        w = min(cell_w, frame.shape[1])
        out[y1:y1 + h, x1:x1 + w] = frame[:h, :w]
    return out


def overlay_sprite_on_canvas(
    canvas_bgr: np.ndarray,
    sprite_bgra: np.ndarray,
    polygon_plane: Iterable[Iterable[float]],
    to_px,
    *,
    scale: float = 1.35,
    flip_x: bool = False,
) -> bool:
    """Alpha-composite a sprite over a plane polygon's bounding box on ``canvas_bgr``."""
    if cv2 is None or sprite_bgra is None or sprite_bgra.size == 0:
        return False
    pts = [to_px(float(p[0]), float(p[1])) for p in polygon_plane if len(p) >= 2]
    if len(pts) < 3:
        return False
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max(4, max_x - min_x)
    bh = max(4, max_y - min_y)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5

    sh, sw = sprite_bgra.shape[:2]
    if sh <= 0 or sw <= 0:
        return False
    target_w = max(8, int(bw * scale))
    target_h = max(8, int(target_w * sh / sw))
    if target_h < bh * 0.8:
        target_h = max(8, int(bh * scale))
        target_w = max(8, int(target_h * sw / sh))
    sprite = cv2.resize(sprite_bgra, (target_w, target_h), interpolation=cv2.INTER_AREA)
    if flip_x:
        sprite = cv2.flip(sprite, 1)

    x1 = int(round(cx - target_w * 0.5))
    y1 = int(round(cy - target_h * 0.5))
    return _alpha_blit(canvas_bgr, sprite, x1, y1)


def _sheet_prompt(
    character: str,
    action: str,
    frame_count: int,
    *,
    cols: int = SPRITE_SHEET_COLS,
    rows: int = SPRITE_SHEET_ROWS,
    start_idx: int = 0,
    sheet_frame_count: Optional[int] = None,
    sheet_idx: int = 0,
    sheet_count: int = 1,
    has_reference: bool = False,
) -> str:
    cols = max(1, int(cols))
    rows = max(1, int(rows))
    cells_per_sheet = cols * rows
    sheet_frame_count = max(1, int(sheet_frame_count or min(cells_per_sheet, frame_count)))
    poses = [
        _pose_description(action, frame_idx, frame_count)
        for frame_idx in range(start_idx, min(frame_count, start_idx + sheet_frame_count))
    ]
    pose_lines = "\n".join(
        f"{idx + 1}. Cell {idx + 1}: global frame {start_idx + idx + 1} of {frame_count}, {pose}"
        for idx, pose in enumerate(poses)
    )
    empty_cell_note = ""
    if sheet_frame_count < cells_per_sheet:
        first_empty = sheet_frame_count + 1
        empty_cell_note = (
            f"Cells {first_empty} through {cells_per_sheet} must be completely transparent and empty.\n"
        )
    reference_note = ""
    if has_reference:
        reference_note = (
            "Use the attached reference image as the exact character silhouette identity. "
            "Keep the same body proportions, head shape, ear shape, tail shape, leg thickness, and overall scale style. "
            "Only change the pose for animation; do not redesign the character.\n"
        )
    return (
        "Create one transparent PNG sprite sheet for a character animation cycle.\n"
        f"Canvas layout: exactly {cols} columns by {rows} rows, "
        f"exactly {cells_per_sheet} equal-size cells, ordered left to right across each row, "
        "then top row to bottom row.\n"
        f"The layout must be {cols} by {rows}, not any other arrangement. "
        "Do not use 5 columns, 2 rows, 3 rows with uneven columns, diagonal placement, or freeform spacing. "
        "Every cell must occupy the same invisible rectangular slot in a strict uniform grid.\n"
        f"This is sheet {sheet_idx + 1} of {sheet_count}; it contains "
        f"{sheet_frame_count} occupied animation cells for global frames "
        f"{start_idx + 1} through {start_idx + sheet_frame_count} of {frame_count}.\n"
        f"Character: the shadow of {character}.\n"
        f"Action: {action} animation cycle.\n\n"
        f"{reference_note}"
        "Pose list:\n"
        f"{pose_lines}\n\n"
        f"{empty_cell_note}"
        "Each occupied cell must contain exactly one full-body character silhouette for that frame.\n"
        "Do not create extra poses, duplicate characters, panels, visible borders, grid lines, text, or labels.\n"
        "Transparent background only across the whole canvas. No gray backdrop, no floor, no cast shadow.\n"
        "Style: pure black projected shadow silhouette, slightly soft edge, no facial features, "
        "no color, no texture.\n"
        "In every occupied cell, the silhouette bounding box must be the same size: "
        f"about {int(SPRITE_FRAME_TARGET_WIDTH_RATIO * 100)} percent of the cell width and "
        f"about {int(SPRITE_FRAME_TARGET_HEIGHT_RATIO * 100)} percent of the cell height. "
        "Do not let any silhouette become larger, smaller, cropped, or closer to the cell edge than the others.\n"
        "In every cell, center the character at the same relative anchor point with the same scale "
        "and enough transparent padding around the full body.\n"
        "Align every cell to the same invisible baseline: the lowest foot or body point must touch "
        "the same invisible horizontal line at about 84 percent of the cell height.\n"
        "Keep the torso centered on the same invisible vertical centerline in every cell. "
        "Do not shift the character left, right, up, or down between cells.\n"
        "Keep the character identity consistent across all cells."
    )


def _frame_prompt(character: str, action: str, frame_idx: int, frame_count: int) -> str:
    pose = _pose_description(action, frame_idx, frame_count)
    return (
        "Create exactly one animation frame as a transparent PNG.\n"
        f"Character: the shadow of {character}.\n"
        f"Action: {action} animation cycle.\n"
        f"Frame: {frame_idx + 1} of {frame_count}, {pose}.\n\n"
        "The image must contain exactly one full-body character silhouette.\n"
        "Do not create a sprite sheet. Do not create multiple poses. Do not create panels or a sequence.\n"
        "Transparent background only. No gray backdrop, no floor, no cast shadow, no text, no labels.\n"
        "Style: pure black projected shadow silhouette, slightly soft edge, no facial features, "
        "no color, no texture.\n"
        "Center the character in the image with enough transparent padding around the full body.\n"
        "Keep the character identity consistent with the other frames."
    )


def _pose_description(action: str, frame_idx: int, frame_count: int) -> str:
    t = 0.0 if frame_count <= 1 else frame_idx / (frame_count - 1)
    action = _safe_name(action)
    poses = {
        "idle": [
            "standing calmly, neutral pose",
            "subtle breathing stretch upward",
            "subtle breathing relax downward",
            "standing calmly, neutral pose",
        ],
        "walk": [
            "left foot forward, body leaning slightly forward",
            "passing pose, body centered",
            "right foot forward, body leaning slightly forward",
            "passing pose, body centered",
        ],
        "hop": [
            "crouching before a jump, body compressed",
            "rising upward, body stretched slightly",
            "highest point of the jump, feet lifted",
            "landing squash, body compressed near ground",
        ],
        "stretch": [
            "standing neutral before stretching",
            "arms and body stretching upward",
            "longest upward stretch, body tall",
            "relaxing back toward neutral",
        ],
        "hide": [
            "standing alert",
            "crouching low",
            "very low hiding pose, body compact",
            "staying low and still",
        ],
        "climb": [
            "reaching upward with one arm",
            "pulling body upward",
            "one leg lifted as if climbing",
            "settling higher on the surface",
        ],
        "inspect": [
            "standing curious",
            "leaning forward to look closely",
            "leaning farthest forward",
            "returning to curious stance",
        ],
        "flee": [
            "startled pose",
            "running away with body stretched",
            "fast running stride",
            "small compressed sprint pose",
        ],
    }
    seq = poses.get(action, poses["idle"])
    if frame_count == len(seq):
        return seq[min(frame_idx, len(seq) - 1)]
    pos = min(len(seq) - 1, max(0, round(t * (len(seq) - 1))))
    return seq[pos]


def _safe_name(text: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:64] or "shadow_agent"


def _trim_transparent(img: np.ndarray, pad: int = 8) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] < 4:
        return img
    alpha = img[:, :, 3]
    ys, xs = np.where(alpha > 8)
    if len(xs) == 0 or len(ys) == 0:
        return img
    x1 = max(0, int(xs.min()) - pad)
    x2 = min(img.shape[1], int(xs.max()) + pad + 1)
    y1 = max(0, int(ys.min()) - pad)
    y2 = min(img.shape[0], int(ys.max()) + pad + 1)
    return img[y1:y2, x1:x2].copy()


def _trim_file_in_place(path: Path) -> None:
    if cv2 is None:
        return
    img = SpriteActionLibrary._read_bgra(path)
    if img is None:
        return
    trimmed = _trim_transparent(img, pad=16)
    cv2.imwrite(str(path), trimmed)


def _alpha_blit(canvas_bgr: np.ndarray, sprite_bgra: np.ndarray, x: int, y: int) -> bool:
    h, w = canvas_bgr.shape[:2]
    sh, sw = sprite_bgra.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + sw)
    y2 = min(h, y + sh)
    if x1 >= x2 or y1 >= y2:
        return False
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    src = sprite_bgra[sy1:sy2, sx1:sx2]
    if src.shape[2] < 4:
        canvas_bgr[y1:y2, x1:x2] = src[:, :, :3]
        return True
    alpha = (src[:, :, 3:4].astype(np.float32) / 255.0)
    dst = canvas_bgr[y1:y2, x1:x2].astype(np.float32)
    rgb = src[:, :, :3].astype(np.float32)
    canvas_bgr[y1:y2, x1:x2] = (rgb * alpha + dst * (1.0 - alpha)).astype(np.uint8)
    return True
