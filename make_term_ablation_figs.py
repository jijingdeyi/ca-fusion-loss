"""Term-ablation collages with a red box and a zoomed inset row."""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import ROOT, TEST_PATH

OUTDIR = ROOT / "results" / "term_ablation"
IR_DIR = Path(TEST_PATH) / "ir"
VI_DIR = Path(TEST_PATH) / "vi"

PANELS = [
    ("IR", None, "ir"),
    ("Visible", None, "vi"),
    ("Base", "abl-h0-w0", "fused"),
    ("Halo-only", "abl-h1-w0", "fused"),
    ("Washout-only", "abl-h0-w0p5", "fused"),
    ("Full", "grid-h1-w0p5", "fused"),
]

# (left, top, right, bottom) on the original 640x480 image
SCENES = [
    {
        "name": "00111D.png",
        "out": "00111D_term_ablation.png",
        "box": (48, 120, 208, 304),
    },
    {
        "name": "00931N.png",
        "out": "00931N_term_ablation.png",
        "box": (256, 144, 424, 292),
    },
]

PANEL_W = 320
GAP = 6
BAR = 34
BOX_COLOR = (220, 30, 30)
BOX_WIDTH = 3
ZOOM_BORDER = 3


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def load_rgb(kind: str, tag: str | None, name: str) -> Image.Image:
    if kind == "ir":
        path = IR_DIR / name
    elif kind == "vi":
        path = VI_DIR / name
    else:
        path = ROOT / "results" / "MSRS" / tag / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def draw_box(im: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out)
    x0, y0, x1, y1 = box
    for i in range(BOX_WIDTH):
        draw.rectangle([x0 - i, y0 - i, x1 + i, y1 + i], outline=BOX_COLOR)
    return out


def labeled(im: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (im.width, im.height + BAR), (255, 255, 255))
    canvas.paste(im, (0, BAR))
    draw = ImageDraw.Draw(canvas)
    f = font(20)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((im.width - tw) / 2, (BAR - th) / 2 - 1), text, fill=(0, 0, 0), font=f)
    return canvas


def resize_w(im: Image.Image, width: int) -> Image.Image:
    h = int(round(im.height * width / im.width))
    return im.resize((width, h), Image.Resampling.BICUBIC)


def build_one(spec: dict) -> Path:
    name = spec["name"]
    box = spec["box"]
    fulls = []
    zooms = []
    for title, tag, kind in PANELS:
        im = load_rgb(kind, tag, name)
        fulls.append(labeled(resize_w(draw_box(im, box), PANEL_W), title))
        crop = im.crop(box)
        zoom = resize_w(crop, PANEL_W)
        bordered = Image.new(
            "RGB",
            (zoom.width + 2 * ZOOM_BORDER, zoom.height + 2 * ZOOM_BORDER),
            BOX_COLOR,
        )
        bordered.paste(zoom, (ZOOM_BORDER, ZOOM_BORDER))
        zooms.append(bordered)

    row_h1 = max(im.height for im in fulls)
    row_h2 = max(im.height for im in zooms)
    n = len(fulls)
    width = n * PANEL_W + (n - 1) * GAP
    canvas = Image.new("RGB", (width, row_h1 + GAP + row_h2), (255, 255, 255))
    x = 0
    for full, zoom in zip(fulls, zooms):
        canvas.paste(full, (x, 0))
        canvas.paste(zoom, (x, row_h1 + GAP))
        x += PANEL_W + GAP

    out = OUTDIR / spec["out"]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    canvas.save(out, optimize=True)
    print("wrote", out, canvas.size)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=str(ROOT / "results" / "term_ablation"))
    args = parser.parse_args()
    global OUTDIR
    OUTDIR = Path(args.outdir)
    for spec in SCENES:
        build_one(spec)


if __name__ == "__main__":
    main()
