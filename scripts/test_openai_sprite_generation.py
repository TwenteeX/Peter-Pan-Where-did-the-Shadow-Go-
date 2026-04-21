"""
End-to-end OpenAI sprite generation test.

This does not need a camera, SAM2, or TouchDesigner. It calls OpenAI image
generation, saves one transparent sprite sheet per action, splits it into cached
frames, then previews the cached frames.

Usage:
  python -m scripts.test_openai_sprite_generation
  python -m scripts.test_openai_sprite_generation --character "a small white cat" --actions idle walk hop stretch
  python -m scripts.test_openai_sprite_generation --no-preview
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import cv2

from peter_pan.generation import SpriteGenerationRequest, generate_sprite_cache_openai


def main() -> None:
    parser = argparse.ArgumentParser(description="Test OpenAI shadow sprite generation")
    parser.add_argument(
        "--character",
        default="a small curious cat",
        help="Character description used in the sprite prompt.",
    )
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["idle", "walk", "hop", "stretch"],
        help="Actions to generate. Each action costs one generated image.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Frames to request per action. 8 uses one 4x2 sheet.",
    )
    parser.add_argument("--model", default="gpt-image-1", help="OpenAI image model.")
    parser.add_argument("--quality", default="medium", choices=["low", "medium", "high", "auto"])
    parser.add_argument("--size", default="1536x1024", help="Image size, e.g. 1536x1024 for 3x2 cells.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if cached files exist.")
    parser.add_argument("--no-preview", action="store_true", help="Skip OpenCV preview window.")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. In PowerShell, run:\n"
            '$env:OPENAI_API_KEY="你的 key"\n'
            "then run this command again."
        )

    req = SpriteGenerationRequest(
        character_description=args.character,
        actions=tuple(args.actions),
        frames_per_action=max(1, int(args.frames)),
        model=args.model,
        size=args.size,
        quality=args.quality,
    )

    print(
        "[test] generating sprite cache\n"
        f"  character={req.character_description!r}\n"
        f"  actions={', '.join(req.actions)}\n"
        f"  model={req.model} quality={req.quality} size={req.size}\n"
        "  note: one sprite sheet is generated per action unless already cached"
    )
    lib = generate_sprite_cache_openai(req, overwrite=args.overwrite)
    print(f"[test] cache ready: {lib.root_dir}")
    print(f"[test] available actions: {', '.join(lib.available_actions)}")

    if args.no_preview:
        return

    print("[test] preview: 1-8 switch action, q quit")
    actions = list(req.actions)
    idx = 0
    while True:
        action = actions[idx % len(actions)]
        frame = lib.get_frame(action, time.time(), fps=8.0)
        if frame is None:
            raise SystemExit(f"Could not load generated sprite frame for action={action}")
        preview = _preview_canvas(frame, action, lib.root_dir)
        cv2.imshow("openai_sprite_generation_test", preview)
        key = cv2.waitKey(16) & 0xFF
        if key in (ord("q"), ord("Q")):
            break
        if key in [ord(str(n)) for n in range(1, 9)]:
            n = int(chr(key)) - 1
            if 0 <= n < len(actions):
                idx = n
        elif key in (ord(" "), ord("\t")):
            idx = (idx + 1) % len(actions)
    cv2.destroyAllWindows()


def _preview_canvas(frame_bgra, action: str, cache_dir: Path):
    canvas = 255 * (frame_bgra[:, :, :3] * 0 + 1)
    canvas = canvas.astype("uint8")
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)
    h, w = frame_bgra.shape[:2]
    scale = min(680 / max(1, w), 500 / max(1, h), 4.0)
    resized = cv2.resize(frame_bgra, (max(1, int(w * scale)), max(1, int(h * scale))))
    ch, cw = 720, 960
    out = 255 * (resized[:, :, :3] * 0 + 1)
    out = out.astype("uint8")
    out = cv2.resize(out, (cw, ch))
    out[:] = (235, 235, 235)
    x = (cw - resized.shape[1]) // 2
    y = (ch - resized.shape[0]) // 2
    _alpha_blit(out, resized, x, y)
    cv2.putText(out, f"action: {action}", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2, cv2.LINE_AA)
    cv2.putText(out, str(cache_dir), (20, ch - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (70, 70, 70), 1, cv2.LINE_AA)
    return out


def _alpha_blit(canvas_bgr, sprite_bgra, x: int, y: int) -> None:
    h, w = canvas_bgr.shape[:2]
    sh, sw = sprite_bgra.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + sw)
    y2 = min(h, y + sh)
    if x1 >= x2 or y1 >= y2:
        return
    sx1 = x1 - x
    sy1 = y1 - y
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    src = sprite_bgra[sy1:sy2, sx1:sx2]
    alpha = src[:, :, 3:4].astype("float32") / 255.0
    dst = canvas_bgr[y1:y2, x1:x2].astype("float32")
    rgb = src[:, :, :3].astype("float32")
    canvas_bgr[y1:y2, x1:x2] = (rgb * alpha + dst * (1.0 - alpha)).astype("uint8")


if __name__ == "__main__":
    main()
