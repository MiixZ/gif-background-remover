#!/usr/bin/env python3
"""
GIF Background Remover

A generic CLI tool that removes the background from animated GIFs using
pre-trained segmentation models from the `rembg` project.

Supported models:
    - bria-rmbg (default, good for general images)
    - isnet-anime (better for anime-style illustrations)
    - u2net, u2net_human_seg, isnet-general-use, sam, etc.

The tool preserves frame delays and loop settings from the original GIF.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageSequence
from rembg import remove, new_session


SUPPORTED_MODELS = (
    "bria-rmbg",
    "isnet-anime",
    "isnet-general-use",
    "u2net",
    "u2net_human_seg",
    "u2netp",
    "silueta",
    "sam",
)

DEFAULT_MODEL = "bria-rmbg"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Remove the background from animated GIFs."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more GIF files or directories containing GIFs.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory. Defaults to the same directory as the input file.",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=SUPPORTED_MODELS,
        help=f"Segmentation model to use. Default: {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help=(
            "Combine the chosen model with isnet-anime using a lightened mask. "
            "Useful when the default model cuts off parts of the subject."
        ),
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help=(
            "Crop each frame before processing. Format: WxH+X+Y. "
            "The output canvas size still matches the original GIF."
        ),
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="-transparent",
        help="Suffix appended to the output filename. Default: -transparent.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="GIF",
        choices=["GIF", "APNG", "WEBP"],
        help="Output format. Default: GIF.",
    )
    parser.add_argument(
        "--dispose",
        type=str,
        default="previous",
        choices=["previous", "background", "none"],
        help="Frame disposal method. Default: previous.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=256,
        help="Maximum number of colors in the output palette. Default: 256.",
    )
    return parser.parse_args(argv)


def collect_gif_paths(inputs: Iterable[Path]) -> List[Path]:
    """Expand directories and return a list of GIF file paths."""
    paths: List[Path] = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*.gif")))
        elif item.is_file() and item.suffix.lower() == ".gif":
            paths.append(item)
        else:
            print(f"Warning: skipping {item} (not a GIF file or directory)", file=sys.stderr)
    return paths


def parse_crop_box(spec: str | None) -> Tuple[int, int, int, int] | None:
    """Parse a crop string like '420x281+80+0' into (width, height, left, top)."""
    if spec is None:
        return None
    try:
        size, offset = spec.split("+", 1)
        width, height = (int(x) for x in size.split("x"))
        left, top = (int(x) for x in offset.split("+", 1))
        return width, height, left, top
    except ValueError as exc:
        raise ValueError(f"Invalid crop specification: {spec!r}") from exc


def extract_frames(src: Image.Image) -> List[Image.Image]:
    """Return a list of RGBA frames from an animated GIF, coalesced."""
    frames: List[Image.Image] = []
    for frame in ImageSequence.Iterator(src):
        # Coalesce to full frames so disposal methods behave predictably.
        rgba = frame.convert("RGBA")
        frames.append(rgba)
    return frames


def get_frame_delays(src: Image.Image) -> List[int]:
    """Return frame durations in hundredths of a second."""
    delays: List[int] = []
    for frame in ImageSequence.Iterator(src):
        # GIF duration is in milliseconds; convert to centiseconds.
        ms = frame.info.get("duration", 100)
        delays.append(max(1, int(ms / 10)))
    return delays


def get_loop_count(src: Image.Image) -> int:
    """Return the loop count from the original GIF (0 means infinite)."""
    return src.info.get("loop", 0)


def remove_background(image: Image.Image, session) -> Image.Image:
    """Remove the background from a single PIL RGBA image."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    result_bytes = remove(buf.read(), session=session)
    return Image.open(io.BytesIO(result_bytes)).convert("RGBA")


def process_frame(
    frame: Image.Image,
    session_primary,
    session_secondary=None,
) -> Image.Image:
    """Process a single frame, optionally combining two model masks."""
    primary = remove_background(frame, session_primary)

    if session_secondary is None:
        return primary

    secondary = remove_background(frame, session_secondary)

    # Build binary masks from the alpha channels.
    primary_mask = primary.getchannel("A").point(lambda p: 255 if p > 128 else 0)
    secondary_mask = secondary.getchannel("A").point(lambda p: 255 if p > 128 else 0)

    # Lighten composite: keep a pixel if either model thinks it is foreground.
    combined_mask = ImageChops.lighter(primary_mask, secondary_mask)

    # Apply the combined mask to the original frame.
    output = frame.copy()
    output.putalpha(combined_mask)
    return output


def crop_frame(
    frame: Image.Image,
    crop_box: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
) -> Image.Image:
    """Crop a frame and paste it back onto a transparent canvas of original size."""
    width, height, left, top = crop_box
    cropped = frame.crop((left, top, left + width, top + height))
    canvas = Image.new("RGBA", original_size, (0, 0, 0, 0))
    canvas.paste(cropped, (left, top))
    return canvas


def save_output(
    frames: Sequence[Image.Image],
    delays: Sequence[int],
    loop: int,
    dispose: str,
    quality: int,
    output_path: Path,
    output_format: str,
) -> None:
    """Save processed frames as an animated image."""
    if output_format == "GIF":
        # GIF transparency is 1-bit; quantize to palette with alpha.
        first = frames[0]
        palette_frames: List[Image.Image] = []
        for frame in frames:
            # Convert to palette with alpha channel preserved as index 0.
            quantized = frame.quantize(
                colors=quality,
                method=Image.Quantize.MEDIANCUT,
                dither=Image.Dither.NONE,
            )
            palette_frames.append(quantized)

        palette_frames[0].save(
            output_path,
            save_all=True,
            append_images=palette_frames[1:],
            duration=[d * 10 for d in delays],  # milliseconds
            loop=loop,
            disposal=2 if dispose == "previous" else (2 if dispose == "background" else 1),
            optimize=False,
        )
    elif output_format == "APNG":
        frames[0].save(
            output_path,
            save_all=True,
            append_images=list(frames[1:]),
            duration=[d * 10 for d in delays],
            loop=loop,
        )
    elif output_format == "WEBP":
        frames[0].save(
            output_path,
            save_all=True,
            append_images=list(frames[1:]),
            duration=[d * 10 for d in delays],
            loop=loop,
            lossless=True,
        )


def process_gif(
    gif_path: Path,
    output_dir: Path | None,
    model: str,
    combine: bool,
    crop_box: Tuple[int, int, int, int] | None,
    suffix: str,
    output_format: str,
    dispose: str,
    quality: int,
) -> Path:
    """Process a single GIF and write the result to disk."""
    session_primary = new_session(model)
    session_secondary = new_session("isnet-anime") if combine else None

    with Image.open(gif_path) as src:
        original_size = src.size
        frames = extract_frames(src)
        delays = get_frame_delays(src)
        loop = get_loop_count(src)

    processed_frames: List[Image.Image] = []
    for idx, frame in enumerate(frames, start=1):
        print(f"  Frame {idx}/{len(frames)}")

        if crop_box is not None:
            frame = crop_frame(frame, crop_box, original_size)

        processed = process_frame(frame, session_primary, session_secondary)
        processed_frames.append(processed)

    # Determine output path.
    out_dir = output_dir if output_dir is not None else gif_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{gif_path.stem}{suffix}.{output_format.lower()}"
    output_path = out_dir / out_name

    save_output(
        processed_frames,
        delays,
        loop,
        dispose,
        quality,
        output_path,
        output_format,
    )
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    crop_box = parse_crop_box(args.crop)

    gif_paths = collect_gif_paths(args.inputs)
    if not gif_paths:
        print("No GIF files found.", file=sys.stderr)
        return 1

    for gif_path in gif_paths:
        print(f"Processing {gif_path} ...")
        output_path = process_gif(
            gif_path=gif_path,
            output_dir=args.output,
            model=args.model,
            combine=args.combine,
            crop_box=crop_box,
            suffix=args.suffix,
            output_format=args.format,
            dispose=args.dispose,
            quality=args.quality,
        )
        print(f"  -> {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
