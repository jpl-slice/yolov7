#!/usr/bin/env python3
"""
Overlay COCO bounding boxes on the 1024-row quick-look PNGs produced earlier.
"""

import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True)
    ap.add_argument("--outdir", default="data/visualisations")
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    id2img = {im["id"]: im for im in coco["images"]}
    img2anns = {}
    for ann in coco["annotations"]:
        img2anns.setdefault(ann["image_id"], []).append(ann)

    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for img_id, anns in tqdm(img2anns.items(), desc="preview"):
        tif = id2img[img_id]["file_name"]
        stem = pathlib.Path(tif).stem.replace("_masked", "")
        plot_preview(tif, anns, outdir / f"{stem}_boxes.png")

    print("✓ Previews saved to", outdir)


def plot_preview(tif_path: str, anns, png_out: pathlib.Path, rows: int = 1024):
    with rasterio.open(tif_path) as src:
        scale = rows / src.height
        cols = int(src.width * scale)
        arr = src.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)
        tfm = src.transform * rasterio.Affine.scale(src.width / cols, src.height / rows)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    ax.imshow(np.nan_to_num(arr), cmap="gray")

    scale = rows / src.height  # already computed
    for a in anns:
        x, y, w, h = a["bbox"]
        tl_col = x * scale
        tl_row = y * scale
        br_col = (x + w) * scale
        br_row = (y + h) * scale
        ax.plot(
            [tl_col, br_col, br_col, tl_col, tl_col],
            [tl_row, tl_row, br_row, br_row, tl_row],
            color="red",
            lw=0.5,
        )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_out, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
