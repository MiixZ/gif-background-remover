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
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
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
        "--bg-color",
        type=str,
        default=None,
        help=(
            "Remove a flat background colour as well as the model mask. "
            "Use 'auto' to sample it from the frame corners, or pass a hex "
            "value such as '#FFFFFF'. This also removes enclosed background "
            "regions that the model keeps, for example the gap between legs."
        ),
    )
    parser.add_argument(
        "--bg-fuzz",
        type=float,
        default=4.0,
        help="Colour tolerance for --bg-color, in percent. Default: 4.",
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


def parse_background_color(value: str | None) -> Tuple[int, int, int] | None:
    """Parse a hex colour such as '#FFFFFF'. Returns None for 'auto'/absent."""
    if value is None or value.lower() == "auto":
        return None
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if len(text) != 6:
        raise ValueError(f"Invalid background colour: {value!r}")
    return tuple(int(text[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def detect_background_color(frame: Image.Image) -> Tuple[int, int, int]:
    """Return the most common colour found in the four frame corners."""
    rgb = frame.convert("RGB")
    width, height = rgb.size
    corners = [
        rgb.getpixel((0, 0)),
        rgb.getpixel((width - 1, 0)),
        rgb.getpixel((0, height - 1)),
        rgb.getpixel((width - 1, height - 1)),
    ]
    return Counter(corners).most_common(1)[0][0]  # type: ignore[return-value]


def flat_background_mask(
    frame: Image.Image,
    color: Tuple[int, int, int],
    fuzz_percent: float,
) -> Image.Image:
    """Return a black/white mask of pixels close to a flat background colour."""
    rgb = np.asarray(frame.convert("RGB"), dtype=np.int16)
    reference = np.array(color, dtype=np.int16)
    difference = np.abs(rgb - reference)

    # ImageMagick measures fuzz as a percentage of the quantum range, so the
    # per-channel threshold is the same fraction of 255.
    threshold = max(0.0, fuzz_percent) / 100.0 * 255.0
    matches = np.all(difference <= threshold, axis=2)

    mask_array = np.where(matches, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_array, mode="L")


def process_frame(
    frame: Image.Image,
    session_primary,
    session_secondary=None,
    background_color: Tuple[int, int, int] | None = None,
    background_fuzz: float = 4.0,
    background_color_is_auto: bool = False,
) -> Image.Image:
    """Process a single frame, optionally combining two model masks."""
    primary = remove_background(frame, session_primary)

    if session_secondary is None:
        output = primary
    else:
        secondary = remove_background(frame, session_secondary)

        # Build binary masks from the alpha channels.
        primary_mask = primary.getchannel("A").point(lambda p: 255 if p > 128 else 0)
        secondary_mask = secondary.getchannel("A").point(lambda p: 255 if p > 128 else 0)

        # Lighten composite: keep a pixel if either model thinks it is foreground.
        combined_mask = ImageChops.lighter(primary_mask, secondary_mask)

        # Apply the combined mask to the original frame.
        output = frame.copy()
        output.putalpha(combined_mask)

    if background_color is not None or background_color_is_auto:
        color = (
            detect_background_color(frame)
            if background_color_is_auto
            else background_color
        )
        # Intersect with the flat-colour mask: a pixel must be foreground for
        # the model *and* visibly different from the background colour.
        keep_mask = flat_background_mask(frame, color, background_fuzz)
        transparent = Image.new(output.mode, output.size, (0, 0, 0, 0))
        output = Image.composite(output, transparent, keep_mask)

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


def build_shared_palette(
    frames: Sequence[Image.Image],
    colors: int,
    tile_size: Tuple[int, int] = (96, 96),
) -> Image.Image:
    """Build one palette shared by every frame so colours do not flicker.

    A small montage of downscaled frames is quantized once; the resulting
    palette image is then reused to map every frame. The last index is
    reserved as a placeholder colour for fully transparent pixels.
    """
    usable = max(1, min(colors, 255))
    tiles = [
        frame.convert("RGB").resize(tile_size, Image.Resampling.BILINEAR)
        for frame in frames
    ]
    montage = Image.new("RGB", (tile_size[0] * len(tiles), tile_size[1]))
    for index, tile in enumerate(tiles):
        montage.paste(tile, (index * tile_size[0], 0))

    palette_image = montage.quantize(colors=usable, method=Image.Quantize.MEDIANCUT)

    # Pad the palette to the full usable range, then append a placeholder
    # entry that stands for "transparent". Magenta is used so that no real
    # pixel is likely to map to it by accident.
    palette = (palette_image.getpalette() or [])[: usable * 3]
    palette += [0, 0, 0] * (usable - len(palette) // 3)
    palette += [255, 0, 255]
    palette += [0, 0, 0] * (256 - usable - 1)
    palette_image.putpalette(palette[: 256 * 3])
    return palette_image


def to_transparent_palette(
    frame: Image.Image,
    palette_image: Image.Image,
    transparent_index: int = 255,
    alpha_threshold: int = 128,
) -> Image.Image:
    """Map one RGBA frame onto the shared palette, punching out transparency."""
    paletted = frame.convert("RGB").quantize(
        palette=palette_image,
        dither=Image.Dither.FLOYDSTEINBERG,
    )

    alpha = frame.getchannel("A")
    transparent = alpha.point(lambda value: 255 if value < alpha_threshold else 0, mode="1")
    paletted.paste(transparent_index, transparent)
    paletted.info["transparency"] = transparent_index
    return paletted


def save_gif_with_imagemagick(
    frames: Sequence[Image.Image],
    delays: Sequence[int],
    loop: int,
    dispose: str,
    output_path: Path,
) -> bool:
    """Assemble the GIF with ImageMagick, which handles palette and disposal
    reliably for large transparent frames.

    Returns True on success, False when ImageMagick is not installed.
    """
    if shutil.which("magick") is None:
        return False

    with tempfile.TemporaryDirectory(prefix="gif-bg-remover-") as tmp:
        frame_paths: List[str] = []
        for index, frame in enumerate(frames):
            frame_path = Path(tmp) / f"frame_{index:04d}.png"
            frame.save(frame_path, format="PNG")
            frame_paths.append(str(frame_path))

        command = ["magick", "-loop", str(loop), "-dispose", dispose]
        for path, delay in zip(frame_paths, delays):
            command.extend(["-delay", str(delay), path])
        command.append(str(output_path))

        subprocess.run(command, check=True, capture_output=True)

    return True


def save_gif_with_pillow(
    frames: Sequence[Image.Image],
    delays: Sequence[int],
    loop: int,
    dispose: str,
    quality: int,
    output_path: Path,
) -> None:
    """Fallback GIF writer used when ImageMagick is unavailable."""
    palette_image = build_shared_palette(frames, quality)
    palette_frames = [to_transparent_palette(frame, palette_image) for frame in frames]

    disposal_code = {"previous": 3, "background": 2, "none": 1}[dispose]

    palette_frames[0].save(
        output_path,
        save_all=True,
        append_images=palette_frames[1:],
        duration=[d * 10 for d in delays],  # milliseconds
        loop=loop,
        disposal=disposal_code,
        transparency=255,
        optimize=False,
    )


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
        used_imagemagick = save_gif_with_imagemagick(
            frames, delays, loop, dispose, output_path
        )
        if not used_imagemagick:
            save_gif_with_pillow(frames, delays, loop, dispose, quality, output_path)
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
    background_color: Tuple[int, int, int] | None = None,
    background_fuzz: float = 4.0,
    background_auto: bool = False,
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

        processed = process_frame(
            frame,
            session_primary,
            session_secondary,
            background_color=background_color,
            background_fuzz=background_fuzz,
            background_color_is_auto=background_auto,
        )
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
    background_auto = args.bg_color is not None and args.bg_color.lower() == "auto"
    background_color = parse_background_color(args.bg_color)

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
            background_color=background_color,
            background_fuzz=args.bg_fuzz,
            background_auto=background_auto,
        )
        print(f"  -> {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
