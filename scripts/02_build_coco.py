#!/usr/bin/env python3
"""
Create a **single-image COCO file** (bbox-only) for each pre-processed frame
found in configs/selected_files.json.

Assumes the masked image lives at:
    ${paths.processed_root}/<name>_masked.tif
"""

from __future__ import annotations

import argparse
import json
import math
import os

import pandas as pd
import rasterio
import rasterio.warp as riowarp
from omegaconf import OmegaConf

# from pyproj import Geod # Not strictly needed for axis-aligned bbox from diameters
# from pyproj.exceptions import GeodError
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg/preprocess_sar.yaml")
    ap.add_argument("--selected", default="cfg/selected_files.json")
    ap.add_argument("--excel", required=True)
    ap.add_argument("--coco_json", default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    # Expanduser and abspath for paths from config
    cfg.paths.processed_root = os.path.abspath(
        os.path.expandvars(cfg.paths.processed_root)
    )
    if args.coco_json is not None:
        cfg.paths.coco_json = os.path.abspath(os.path.expandvars(args.coco_json))
    else:
        # Default to cfg.paths.processed_root if not specified
        cfg.paths.coco_json = os.path.abspath(os.path.expandvars(cfg.paths.coco_json))
    print(f"Using COCO JSON path: {cfg.paths.coco_json}")
    os.makedirs(cfg.paths.processed_root, exist_ok=True)

    with open(args.selected) as fp:
        keep = set(json.load(fp)["files"])

    df = pd.read_excel(args.excel)

    images, annotations = _make_coco(cfg, keep, df)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "eddy", "supercategory": "eddy"}],
    }

    os.makedirs(os.path.dirname(cfg.paths.coco_json), exist_ok=True)
    with open(cfg.paths.coco_json, "w") as fp:
        json.dump(coco, fp, indent=2)
    print(f"✓ COCO written → {cfg.paths.coco_json}")


def _make_coco(cfg, keep, df):
    images = []
    annotations = []
    ann_id = 1
    # geod = Geod(ellps="WGS84") # Keep for potential future use with rotated ellipses

    for name in tqdm(keep, desc="COCO dataset generation"):
        # tif = pathlib.Path(cfg.paths.processed_root) / f"{name}_masked.tif"
        tif = os.path.join(cfg.paths.processed_root, f"{name}_masked.tif")
        if not os.path.exists(tif):
            print(f"!! {tif} missing - skip")
            continue

        with rasterio.open(tif) as src:
            img_id = len(images) + 1
            images.append(
                {
                    "id": img_id,
                    "file_name": tif,
                    "width": src.width,
                    "height": src.height,
                }
            )

            # loop over all eddies in this scene
            for _, row in df[df["File Name"] == name].iterrows():
                lon, lat = _parse_coord(row["Center Coordinate"])
                # Diameters are given, convert to semi-axes in meters
                major_m = row["Diameter long [km]"] * 500
                minor_m = row["Diameter short [km]"] * 500
                # The Stuhlmacher dataset doesn't explicitly state if "Diameter long"
                # aligns with longitude or latitude. A common interpretation for
                # axis-aligned bounding boxes from such data is to assume
                # semi_major_m is the extent in one direction (e.g. E-W) and
                # semi_minor_m in the other (N-S).
                # For now, let's assume semi_major_m is related to zonal (lon) extent
                # and semi_minor_m to meridional (lat) extent for simplicity.
                # A more advanced approach would use the 'Eastern Auxiliary Coordinate'
                # to determine ellipse orientation and find the minimal axis-aligned
                # bounding box containing the rotated ellipse.
                x0_px, y0_px, w_px, h_px = _ellipse_to_bbox_px(
                    lon, lat, major_m, minor_m, src
                )
                # Ensure width and height are positive
                if w_px <= 0 or h_px <= 0:
                    print(
                        f"Warning: Non-positive bbox w_px={w_px}, h_px={h_px} for an eddy in {name}. Skipping."
                    )
                    continue

                # Clip bounding box to image dimensions
                x0_px = max(0, x0_px)
                y0_px = max(0, y0_px)
                # x1_px = min(src.width, x0_px + w_px) # Coco format is x,y,w,h
                # y1_px = min(src.height, y0_px + h_px)
                # w_px = x1_px - x0_px
                # h_px = y1_px - y0_px
                if x0_px + w_px > src.width:
                    w_px = src.width - x0_px
                if y0_px + h_px > src.height:
                    h_px = src.height - y0_px

                if w_px <= 0 or h_px <= 0:  # Check again after clipping
                    # print(f"Warning: Bbox for an eddy in {name} is outside image boundaries after clipping or became non-positive. Skipping.")
                    continue

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x0_px, y0_px, w_px, h_px],
                        "area": w_px * h_px,
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    return images, annotations


def _parse_coord(coord_str: str):
    lon_str, lat_str = coord_str.split()
    lon = float(lon_str[:-2]) * (-1 if lon_str.endswith("W") else 1)  # Ost in German
    lat = float(lat_str[:-2]) * (-1 if lat_str.endswith("S") else 1)
    return lon, lat


def _ellipse_to_bbox_px(
    center_lon: float,
    center_lat: float,
    semi_major_m: float,
    semi_minor_m: float,
    src: rasterio.DatasetReader,
):
    """
    Converts geographic ellipse parameters (center, semi-axes sizes in meters)
    to an axis-aligned pixel bounding box within the source image.
    This version assumes semi_major_m is the E-W radius and semi_minor_m is N-S radius.
    """
    # Approximate meters per degree latitude (fairly constant for WGS84)
    m_per_deg_lat_approx = 111319.488

    # Approximate meters per degree longitude (varies with latitude)
    # WGS84 equatorial radius in meters
    R_EQUATORIAL_WGS84 = 6378137.0

    # Calculate geographic extents (deltas in degrees)
    if math.cos(math.radians(center_lat)) < 1e-9:  # Near poles
        # At poles, a small E-W extent can span many degrees of longitude.
        # This can lead to bboxes covering the entire image width.
        # For simplicity, if at pole, and major axis is non-zero,
        # take a very large lon extent, effectively covering the image.
        # This might need refinement based on specific use case at poles.
        if (
            semi_major_m > 1e-6
        ):  # Ellipse has a non-zero extent in the longitudinal direction
            # At a pole, any longitudinal extent covers all longitudes
            dx_deg = 180.0
        else:
            dx_deg = 0.0
    else:
        # Longitude delta depends on latitude
        center_lat_rad = math.radians(center_lat)
        cos_lat = math.cos(center_lat_rad)
        m_per_deg_lon_approx = (math.pi / 180.0) * R_EQUATORIAL_WGS84 * cos_lat
        dx_deg = semi_major_m / m_per_deg_lon_approx

    dy_deg = semi_minor_m / m_per_deg_lat_approx

    # Calculate geographic corners of the bounding box
    west = center_lon - dx_deg
    east = center_lon + dx_deg
    south = center_lat - dy_deg
    north = center_lat + dy_deg

    # Clip latitudes to the valid range [-90, 90]
    south = max(-90.0, min(90.0, south))
    north = max(-90.0, min(90.0, north))

    # Use rasterio.warp.transform_bounds to convert geo bounds to image's native CRS bounds
    try:
        native_bounds = riowarp.transform_bounds(
            "EPSG:4326", src.crs, west, south, east, north
        )
        native_west, native_south, native_east, native_north = native_bounds
    except Exception as e:
        print(f"Error in transform_bounds for {center_lon, center_lat}: {e}")
        return (0, 0, 0, 0)

    # Convert native CRS corners to pixel coordinates
    # Note: src.index maps (x, y) in src.crs to (row, col)
    # We need (col, row) for x, y in bounding boxes.
    # ~src.transform maps (x,y) in src.crs to (col_frac, row_frac) pixel coords
    # Top-left corner
    px_tl_col, px_tl_row = (~src.transform) * (native_west, native_north)
    # Bottom-right corner
    px_br_col, px_br_row = (~src.transform) * (native_east, native_south)

    # Pixel coordinates for COCO bbox [x_min, y_min, width, height]
    # Ensure order (min_col, min_row, max_col, max_row)
    x0_px = min(px_tl_col, px_br_col)
    y0_px = min(px_tl_row, px_br_row)  # row index is y
    x1_px = max(px_tl_col, px_br_col)
    y1_px = max(px_tl_row, px_br_row)

    # Ensure width and height are at least 0
    width_px = max(0, x1_px - x0_px)
    height_px = max(0, y1_px - y0_px)

    return int(x0_px), int(y0_px), int(width_px), int(height_px)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
