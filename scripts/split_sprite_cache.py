"""
Split generated sprite sheets into individual frame PNGs.

Usage:
  python -m scripts.split_sprite_cache assets/generated_sprites/a_plush_teddy_bear
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from peter_pan.generation.sprite_cache import SpriteActionLibrary


def main() -> None:
    parser = argparse.ArgumentParser(description="Split sprite sheets into frame PNGs")
    parser.add_argument("cache_dir", help="Generated sprite cache directory")
    parser.add_argument("--frames", type=int, default=6, help="Frames per action sheet")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing split frames")
    args = parser.parse_args()

    lib = SpriteActionLibrary(args.cache_dir, frames_per_action=args.frames)
    counts = lib.ensure_split_frames(overwrite=args.overwrite)
    if not counts:
        print("No sheets found to split.")
        return
    print(f"Split cache: {Path(args.cache_dir)}")
    for action, count in sorted(counts.items()):
        print(f"  {action}: {count} frames -> {Path(args.cache_dir) / 'frames' / action}")


if __name__ == "__main__":
    main()
