"""Package captured application frames as an animated demo (optional Pillow)."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a GIF from --capture-dir application frames")
    parser.add_argument("capture", type=Path, help="Directory containing frames.jsonl and PNG frames")
    parser.add_argument("output", type=Path, help="Output GIF path")
    parser.add_argument("--width", type=int, default=960, help="Output width, default 960")
    parser.add_argument("--compact", action="store_true",
                        help="Remove consecutive identical PNGs and limit stationary holds to 2 seconds")
    args = parser.parse_args()
    if args.width <= 0:
        parser.error("--width must be positive")
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is required only for this optional GIF packaging tool.")
        return 3
    try:
        metadata = [json.loads(line) for line in (args.capture / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
        if not metadata:
            raise ValueError("No captured frames")
        original_count = len(metadata)
        if args.compact:
            selected, previous_digest = [], None
            for entry in metadata:
                digest = hashlib.sha256((args.capture / f"frame_{entry['frame']:06d}.png").read_bytes()).digest()
                if digest != previous_digest:
                    selected.append(entry)
                previous_digest = digest
            metadata = selected
        frames, durations = [], []
        for index, entry in enumerate(metadata):
            with Image.open(args.capture / f"frame_{entry['frame']:06d}.png") as source:
                height = round(source.height * args.width / source.width)
                frames.append(source.convert("RGB").resize((args.width, height), Image.Resampling.LANCZOS)
                              .quantize(colors=128, method=Image.Quantize.MEDIANCUT))
            elapsed = metadata[index + 1]["wall_time"] - entry["wall_time"] if index + 1 < len(metadata) else 0.5
            if args.compact:
                elapsed = min(elapsed, 2.0)
            durations.append(max(10, round(elapsed * 1000 / 10) * 10))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False)
        print(f"DEMO_GIF_OK source_frames={original_count} frames={len(frames)} "
              f"duration_ms={sum(durations)} compact={args.compact} output={args.output}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"DEMO_ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
