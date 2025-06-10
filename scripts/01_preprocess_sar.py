#!/usr/bin/env python3
"""
Mask land + (optional) high-tail clip for each selected SAR frame.

* Reads config from configs/base.yaml
* Uses configs/selected_files.json (made by 00_select_top_files.py)
* Always writes a 1024-row PNG preview in data/visualisations/
* Optionally writes the masked GeoTIFF (dtype = float32/uint16/uint8)
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import re
import sys  # Add this import

# Add the project root to the Python path
project_src = Path(__file__).resolve().parent.parent
print(f"Adding {project_src} to sys.path")
sys.path.insert(0, str(project_src))

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from omegaconf import OmegaConf
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from tqdm import tqdm

from utils.sar_transforms import MASK_VALUE, build_land_masker, mask_land_and_clip


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg/preprocess_sar/preprocess_sar.yaml")
    ap.add_argument("--selected", default="cfg/preprocess_sar/selected_files.json")
    return ap.parse_args()


def find_tif_files(selected_json: str, raw_root: Path) -> list[str]:
    try:
        with open(selected_json) as fp:
            return OmegaConf.create(fp.read()).files  # type: ignore[attr-defined]
    except FileNotFoundError:
        print(f"!! {selected_json} not found → using all *.tif in {raw_root}")
        return [p.stem for p in raw_root.glob("*.tif") if p.is_file()]

def main():
    args = parse_args()
    cfg = OmegaConf.load(args.cfg)

    cfg = OmegaConf.load(args.cfg)
    raw_root = Path(os.path.expandvars(cfg.paths.raw_root))
    proc_root = Path(os.path.expandvars(cfg.paths.processed_root))
    vis_root = Path("data/visualisations")

    proc_root.mkdir(parents=True, exist_ok=True)
    vis_root.mkdir(parents=True, exist_ok=True)

    # get selected files
    scene_names = find_tif_files(args.selected, raw_root)
    if not scene_names:
        print(f"!! No files found in {raw_root}, exiting.")
        sys.exit(1)

    # Build a *single* callable land masker (loads shapefile once)
    land_masker = build_land_masker(cfg.preprocess.land_shapefile)

    for name in tqdm(scene_names, desc="preprocess"):
        matching_files = sorted(raw_root.glob(f"{name}*.tif"))
        if not matching_files:
            print(f"!! No files matching {name}*.tif in {raw_root}")
            continue

        # skip if maked already
        # out_tif = proc_root / f"{name}.tif"
        # if out_tif.exists():
        #     print(f"✓ {out_tif} already exists, skipping.")
        #     continue

        with rasterio.open(matching_files[0]) as src:
            masked = mask_land_and_clip(
                src,
                land_masker,
                clip_percentile=cfg.preprocess.clip_percentile,
                dilate_px=12,
            )

            # convert to dB
            if getattr(cfg.preprocess, "convert_to_db", True):
                epsilon = 1e-10
                valid = np.isfinite(masked)
                masked[valid] = 10 * np.log10(masked[valid] + epsilon)
                masked = np.clip(masked, -35, None)  # set a -35 dB floor

            name = extract_scene_id(matching_files[0].name)
            if cfg.preprocess.save_masked:
                out_tif = proc_root / f"{name}.tif"
                write_masked(
                    src, masked, out_tif, cfg.preprocess.masked_dtype,
                    compress=cfg.preprocess.compress
                )

            quicklook_png = vis_root / f"{name}_preview.png"
            quicklook(src, masked, quicklook_png)

    print("✓ Preprocessing done.")


def extract_scene_id(filename):
    """Extract Sentinel-1 scene ID from filename."""
    basename = Path(filename).stem
    # regex to split on known suffixes and keep only the scene ID part
    scene_id = re.split(r"_Cal|_ML|_Spk|_TC|_Orb|_processed", basename)[0]
    return scene_id


def write_masked(
    src: rasterio.DatasetReader,
    masked: np.ndarray,
    out_tif: Path,
    dtype: str = "float32",
    compress: str = "DEFLATE",
    tiled: bool = True,
):
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype=dtype,
        nodata=MASK_VALUE,
        compress=compress,
        tiled=tiled,
        blockxsize=512,  # multiples of 16
        blockysize=512,
    )

    if compress in {"DEFLATE", "ZSTD"}:
        profile["predictor"] = 2  # horizontal differencing
        profile["zlevel"] = 9
        profile["num_threads"] = 12

    if dtype in ("uint8", "uint16"):
        # linear min-max stretch over valid ocean pixels
        valid = np.isfinite(masked)
        data = scale_valid_masked_data(masked, dtype, valid)
        data[~valid] = MASK_VALUE  # set nans to nodata value
    else:  # float32 passthrough
        data = np.where(np.isfinite(masked), masked, MASK_VALUE).astype("float32")
        # data = np.clip(data, 0, 1)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data, 1)

    return data

def scale_valid_masked_data(
    masked, dtype, valid, percentile: tuple[float, float] = (0, 99)
):
    """Scale valid masked data to the range of the specified dtype.

    To brighten the image and sacrifice information at the brightest parts,
    use a lower percentile for the high end. (i.e., lower white point).
    This will stretch the narrower band of  data from low_percentile - high_percentile over
    the same uint8/uint16 range.

    Args:
        masked (np.ndarray): The masked data array with NaNs for land.
        dtype (str): The desired output data type, e.g., "uint8" or "uint16".
        valid (np.ndarray): A boolean mask indicating valid pixels in `masked`.
        percentile (tuple[float, float]): Percentiles to use for scaling (default: (0, 99)).
    Returns:
        np.ndarray: The scaled data array with the same shape as `masked`,
                    with NaNs replaced by the nodata value.
    """
    if not np.any(valid):
        scaled = np.zeros_like(masked, dtype=dtype)
    else:
        lo, hi = np.nanpercentile(masked[valid], percentile)
        if lo == hi:
            # all valid pixels have the same value, scale to 1
            print(
                f"Warning: all valid pixels have the same value {lo}. Setting all to 1."
            )
            scaled = np.zeros_like(masked, dtype=dtype)
            scaled[valid] = 1
            return scaled

        scale = 255 if dtype == "uint8" else 65535
        usable_range = scale - 1  # 1 ... 255 for uint8, 1 ... 65535 for uint16
        denom = hi - lo if hi != lo else 1.0
        scaled = np.zeros_like(masked, dtype=dtype)  # nodata stays 0
        scaled[valid] = (
            np.round(((masked[valid] - lo) / denom) * usable_range) + 1
        ).astype(dtype)

    return scaled


def quicklook(
    src: rasterio.DatasetReader,
    masked: np.ndarray,
    out_png: Path,
    rows: int = 2048,
):
    """Save a lat-lon referenced grayscale preview (PNG)."""
    scale = rows / src.height
    cols = max(1, int(src.width * scale))

    # resample masked array instead using an in-memory rasterio dataset
    mem_profile = src.profile | {"driver": "MEM", "nodata": MASK_VALUE}
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**mem_profile) as mem:
            mem.write(masked, 1)
            arr = mem.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)

    # 2) Now, figure out the lon/lat bounds for plotting
    #    Rasterio’s `src.bounds` is in src.crs. If src.crs is not geographic,
    #    we reproject those bounds into EPSG:4326.
    try:
        crs = src.crs
    except AttributeError:
        crs = None

    if crs is None:
        b = src.bounds
        left, bottom, right, top = (b.left, b.bottom, b.right, b.top)
    else:
        left, bottom, right, top = transform_bounds(crs, "EPSG:4326", *src.bounds)

    # Now (left,bottom,right,top) are in lon/lat
    # 3) Build our lon & lat arrays (in degrees) exactly as before, but now
    #    they represent true longitudes and latitudes.
    lon = np.linspace(left, right, cols)
    lat = np.linspace(top, bottom, rows)

    # create matplotlib image with lat-lon coordinates
    fig, ax = plt.subplots(figsize=(cols / 200, rows / 200), dpi=300)
    # lat = np.linspace(src.bounds.top, src.bounds.bottom, rows)  # top to bottom
    # lon = np.linspace(src.bounds.left, src.bounds.right, cols)  # left to right
    im = ax.imshow(
        np.nan_to_num(arr),
        extent=(lon[0], lon[-1], lat[0], lat[-1]),
        cmap="gray",
        vmin=np.nanmin(arr),
        vmax=np.nanmax(arr),
    )
    ax.set_aspect("equal")
    ax.set(title=Path(src.name).name, xlabel="Longitude", ylabel="Latitude")
    plt.tight_layout()
    fig.colorbar(im, ax=ax, label="SAR intensity", fraction=0.046, pad=0.04)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
