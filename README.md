# GIF Background Remover

A generic command-line tool for removing backgrounds from animated GIFs using pre-trained segmentation models from the [`rembg`](https://github.com/danielgatis/rembg) project.

No model training is performed; the tool orchestrates existing models and preserves the original timing (frame delays and loop count) when rebuilding the output.

## Features

- Process a single GIF or a whole folder of GIFs.
- Choose from several `rembg` models (`bria-rmbg`, `isnet-anime`, `u2net`, etc.).
- Optional model combination for tricky subjects.
- Optional pre-crop for scenes where the background object is confused with the subject.
- Optional flat-colour removal for enclosed background regions the model misses.
- Output to GIF, APNG, or WebP.
- Preserves frame durations and loop settings.

## Installation

```bash
# Clone the repository
git clone https://github.com/MiixZ/gif-background-remover.git
cd gif-background-remover

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

The first time you run the tool with a given model, `rembg` will download the corresponding ONNX weights to `~/.local/share/rembg/models/`. This is usually around 1–2 GB in total if you use several models.

## Basic usage

```bash
# Remove the background from one GIF
python gif_bg_remover.py input.gif

# Process every GIF in a folder
python gif_bg_remover.py ./gifs/

# Specify output directory and model
python gif_bg_remover.py ./gifs/ -o ./output/ -m isnet-anime
```

The result is saved next to the original file with the `-transparent` suffix by default.

## Command-line options

| Option | Description |
| --- | --- |
| `inputs` | One or more GIF files or directories. |
| `-o, --output` | Output directory. Defaults to the input file's directory. |
| `-m, --model` | Segmentation model. Default: `bria-rmbg`. |
| `--combine` | Combine the selected model with `isnet-anime` using a lightened mask. Useful when the default model cuts off parts of the subject. |
| `--crop WxH+X+Y` | Crop each frame before processing, then paste it back onto a transparent canvas of the original size. Format example: `420x281+80+0`. |
| `--bg-color auto\|#RRGGBB` | Also remove a flat background colour (sampled from the corners with `auto`). Removes enclosed background regions the model keeps, such as the gap between legs. |
| `--bg-fuzz PERCENT` | Colour tolerance for `--bg-color`. Default: `4`. |
| `--bg-min-area PERCENT` | Minimum size, as a percentage of the frame, for a matching region to be removed. Keeps small details that share the background colour, such as the white of the eyes. Default: `0.05`. |
| `--suffix` | Output filename suffix. Default: `-transparent`. |
| `--format` | Output format: `GIF`, `APNG`, or `WEBP`. Default: `GIF`. |
| `--dispose` | Frame disposal method: `previous`, `background`, or `none`. Default: `previous`. |
| `--quality` | Maximum palette size for GIF output. Default: `256`. |

## Examples

A real before/after pair, 720×1280, 115 frames, 11.5 seconds, flat white background.

**Original**

<img src="https://github.com/MiixZ/gif-background-remover/releases/download/v1.0.0/miku.gif" width="280" alt="Original GIF with a white background">

**Processed**

<img src="https://github.com/MiixZ/gif-background-remover/releases/download/v1.0.0/miku-transparent.gif" width="280" alt="Same GIF with the background removed">

Produced with:

```bash
python gif_bg_remover.py miku.gif --bg-color auto
```

The `--bg-color auto` step is what removes the enclosed white areas the model keeps, such as the gap between the legs, while `--bg-min-area` (0.05 % by default) protects small white details like the highlights in the eyes.

Both GIFs are attached to the [v1.0.0 release](https://github.com/MiixZ/gif-background-remover/releases/tag/v1.0.0) instead of being committed, so they do not bloat the git history. The Miku artwork belongs to its original author and is included only to demonstrate the tool.

## Model selection tips

- **`bria-rmbg`** is the default and works well for general images, including most anime scenes with plain backgrounds.
- **`isnet-anime`** is tuned for anime-style illustrations and sometimes preserves facial details better than `bria-rmbg`.
- **`u2net`** and **`u2net_human_seg`** are the classic rembg models; the human-seg variant is good for photos of people.

When a single model cuts off part of the subject (for example, around the eyes), try `--combine` to merge its mask with `isnet-anime`.

## Handling hard cases

Some GIFs contain background objects whose colors are very similar to the subject. In those cases the model may include the background as part of the foreground mask.

### Example: background object on the left edge

If the left side of every frame contains a lamp, wall, or other static object that the model keeps including, crop it away before processing:

```bash
python gif_bg_remover.py tricky.gif \
  --crop 420x281+80+0 \
  -m bria-rmbg
```

This crops each frame to a 420×281 rectangle starting at (80, 0), removes the background from that crop, and pastes the result back into a transparent 500×281 canvas. The final GIF keeps the original dimensions, minus the cropped-out distraction.

### Example: model cuts the subject's face

If `bria-rmbg` removes part of the subject's face because it is too close to the background color, combine it with `isnet-anime`:

```bash
python gif_bg_remover.py tricky.gif \
  -m bria-rmbg \
  --combine
```

The tool creates a binary mask from each model and keeps any pixel that either model classifies as foreground.

### Example: enclosed background the model does not see

Segmentation models often miss background regions that are completely surrounded by the subject, such as the gap between the legs, the space inside an arm, or a hole under a skirt. Those regions stay opaque and appear as a coloured patch that flickers between frames.

If the background is a single flat colour, remove it as well as the model mask:

```bash
python gif_bg_remover.py tricky.gif --bg-color auto
```

`auto` samples the colour from the four frame corners. You can also pass an explicit value, and loosen the tolerance when the background is not perfectly uniform:

```bash
python gif_bg_remover.py tricky.gif --bg-color '#FFFFFF' --bg-fuzz 6
```

The final alpha is the model mask **intersected** with "not background colour", so ordinary white details on the subject are only removed when they match the background within the fuzz tolerance. Lower `--bg-fuzz` if the subject loses pale areas.

Small details that share the background colour — eye highlights, sparkles, thin white trim — are protected by `--bg-min-area`: only matching regions covering at least that percentage of the frame are removed (0.05 % by default, roughly 460 pixels on a 720×1280 frame). Raise it if small background specks survive; set it to `0` to remove every matching pixel.

```bash
# Keep details smaller than 0.2% of the frame (e.g. a very detailed subject)
python gif_bg_remover.py tricky.gif --bg-color auto --bg-min-area 0.2
```

## Output formats

- **GIF** is the default and most compatible. Frames are assembled with ImageMagick when it is installed, because it handles palettes and frame disposal reliably for large transparent frames; without ImageMagick the tool falls back to Pillow. GIF transparency is 1-bit, so very soft edges may look slightly pixelated.
- **APNG** preserves full 8-bit alpha and works in most modern browsers.
- **WebP** also preserves full alpha and usually produces smaller files.

## Requirements

- Python 3.10 or newer.
- `onnxruntime`, installed automatically by `rembg[cpu]`.
- ImageMagick (`magick`) is optional and recommended for GIF output.

## Model licences

This project does not train, bundle, or redistribute any model weights. The segmentation models are downloaded by `rembg` into `~/.local/share/rembg/models/` on first use, from the project's own release page.

| Model used by rembg | Upstream | Licence |
| --- | --- | --- |
| `bria-rmbg` (RMBG 2.0), the default | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | CC BY-NC 4.0 — **non-commercial use only**; commercial use requires a separate agreement with BRIA AI |
| `isnet-anime`, used by `--combine` | [SkyTNT/anime-seg](https://huggingface.co/skytnt/anime-seg) | Apache-2.0 |
| any other `--model` value | see the [rembg](https://github.com/danielgatis/rembg) documentation | varies per model |

Because the default model is non-commercial, check the licence of the model you choose before using this tool or its output commercially.

## Notes

- The tool coalesces animated GIFs into full frames before processing, so every disposal method is applied to complete frames.
- Processing speed depends on the model and image size; a typical 500×280 GIF takes a few seconds per frame on a CPU.
- This tool is intended for personal use. Respect copyright when processing material you do not own.

## License

MIT
