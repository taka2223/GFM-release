"""Assemble per-run strips under <base>/<tag>/strip.png into a labeled grid.

Usage: python scripts/build_ablation_grid.py <base_dir>
Writes <base_dir>/grid.png.
"""

import os
import sys
import glob
from PIL import Image, ImageDraw, ImageFont


def find_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def main(base_dir: str) -> None:
    tag_dirs = sorted(
        d for d in glob.glob(os.path.join(base_dir, "*"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "strip.png"))
    )
    if not tag_dirs:
        print(f"No strips found under {base_dir}")
        return

    label_w = 320
    rows = []
    for d in tag_dirs:
        tag = os.path.basename(d)
        strip = Image.open(os.path.join(d, "strip.png")).convert("RGB")
        rows.append((tag, strip))

    strip_h = rows[0][1].height
    strip_w = rows[0][1].width
    pad = 8

    total_w = label_w + strip_w + 2 * pad
    total_h = sum(strip_h + pad for _, _ in rows) + pad

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = find_font(28)

    y = pad
    for tag, strip in rows:
        # label box
        draw.text((pad, y + strip_h // 2 - 14), tag, fill=(0, 0, 0), font=font)
        canvas.paste(strip, (label_w, y))
        y += strip_h + pad

    out_path = os.path.join(base_dir, "grid.png")
    canvas.save(out_path)
    print(f"Saved {out_path}  ({len(rows)} rows, {total_w}x{total_h})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/build_ablation_grid.py <base_dir>")
        sys.exit(1)
    main(sys.argv[1])
