#!/usr/bin/env python3
"""
Overlay COCO bounding boxes on the 1024-row quick-look PNGs produced earlier.
"""

import argparse
import json
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
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


def plot_preview(tif_path: str, anns: list, png_out: pathlib.Path, rows: int = 4096):
    """
    Overlays COCO bounding boxes on a preview of a TIF image,
    adds a colorbar, clips low intensity values, and plots
    bounding boxes in the raster's coordinate system.
    """
    with rasterio.open(tif_path) as src:
        # Calculate dimensions for the preview image
        scale_factor = rows / src.height
        cols = int(src.width * scale_factor)

        # Read the data for the preview, resampling as needed
        preview_arr = src.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)
        # Transformation for the preview image pixels to its CRS coordinates
        preview_transform = src.transform * src.transform.scale(
            (src.width / cols), (src.height / rows)
        )

        # Set nodata to NaN for better visualization
        preview_arr[preview_arr == src.nodata] = np.nan

        # Original image transform (pixel to CRS)
        original_transform = src.transform

    # Clip the bottom 0.5 percentile for better visualization
    # Handle potential all-NaN or empty arrays gracefully
    # valid_pixels = preview_arr[~np.isnan(preview_arr)]
    # if valid_pixels.size > 0:
    vmin = np.nanpercentile(preview_arr, 0.5)
    clipped_arr = np.clip(preview_arr, vmin, None)
    # clipped_arr = np.nan_to_num(clipped_arr, nan=vmin) # Ensure NaNs are also set to vmin
    # else: # If all pixels are NaN or array is empty
    # clipped_arr = preview_arr # Or handle as an error/empty plot
    # vmin = None

    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)  # Increased figsize for colorbar

    # Define the extent of the image in its CRS coordinates
    # (left, right, bottom, top)
    img_extent = (
        preview_transform.c,  # x_min / west
        preview_transform.c + preview_transform.a * cols,  # x_max / east
        preview_transform.f + preview_transform.e * rows,  # y_min / south
        preview_transform.f,  # y_max / north
    )

    image_display = ax.imshow(
        clipped_arr,
        cmap="gray",
        vmin=vmin,  # Use calculated vmin for consistent scaling
        extent=img_extent,
        interpolation="nearest",  # Explicitly set interpolation
    )

    # Add colorbar
    cbar = fig.colorbar(image_display, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("SAR Intensity")

    # Plot bounding boxes using transformed coordinates
    for ann in anns:
        x_pixel, y_pixel, w_pixel, h_pixel = ann["bbox"]

        # Define the four corners of the bounding box in pixel coordinates (original image)
        # (col, row) format for rasterio.transform.xy
        corners_pixel = [
            (x_pixel, y_pixel),  # Top-left
            (x_pixel + w_pixel, y_pixel),  # Top-right
            (x_pixel + w_pixel, y_pixel + h_pixel),  # Bottom-right
            (x_pixel, y_pixel + h_pixel),  # Bottom-left
            (x_pixel, y_pixel),  # Close the polygon
        ]

        # Transform pixel coordinates to the raster's CRS coordinates
        # original_transform applies to the full-resolution image
        transformed_corners_x, transformed_corners_y = rasterio.transform.xy(
            transform=original_transform,
            rows=[p[1] for p in corners_pixel],  # list of y_pixels
            cols=[p[0] for p in corners_pixel],  # list of x_pixels
            offset="ul",  # Upper-left convention for pixel coordinates
        )

        ax.plot(
            transformed_corners_x,
            transformed_corners_y,
            color="red",
            lw=0.75,  # Slightly thicker line for potentially larger coordinate range
        )

    ax.set_xlabel("Longitude / Easting (units of CRS)")
    ax.set_ylabel("Latitude / Northing (units of CRS)")
    ax.set_title(f"Preview with Bounding Boxes: {pathlib.Path(tif_path).name}")
    ax.axis("on")  # Turn axis on to see coordinate values
    ax.tick_params(axis="x", rotation=45)  # Rotate x-axis labels for readability
    fig.tight_layout()  # Adjust layout to prevent overlap
    fig.savefig(png_out, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
