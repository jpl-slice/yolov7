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
import pathlib
import sys  # Add this import

# Add the project root to the Python path
project_src = pathlib.Path(__file__).resolve().parent.parent
print(f"Adding {project_src} to sys.path")
sys.path.insert(0, str(project_src))

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from omegaconf import OmegaConf
from PIL import Image
from rasterio.enums import Resampling
from tqdm import tqdm

from utils.sar_transforms import MASK_VALUE, mask_land_and_clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="cfg/preprocess_sar/preprocess_sar.yaml")
    ap.add_argument("--selected", default="cfg/preprocess_sar/selected_files.json")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    raw_root = pathlib.Path(os.path.expandvars(cfg.paths.raw_root))
    proc_root = pathlib.Path(os.path.expandvars(cfg.paths.processed_root))
    proc_root.mkdir(parents=True, exist_ok=True)
    vis_root = pathlib.Path("data/visualisations")
    vis_root.mkdir(parents=True, exist_ok=True)
    # get selected files
    try:
        with open(args.selected) as fp:
            names = OmegaConf.create(fp.read()).files
    except FileNotFoundError:
        # default to all .tifs in raw_root
        print(f"!! {args.selected} not found, using all .tif files in {raw_root}")
        names = [p.stem for p in raw_root.glob("*.tif") if p.is_file()]
    if not names:
        print(f"!! No files found in {raw_root}, exiting.")
        sys.exit(1)

    for name in tqdm(names, desc="preprocess"):
        tif_pattern = raw_root / f"{name}*.tif"
        matching_files = list(glob.glob(str(tif_pattern)))
        if not matching_files:
            print(f"!! no files matching {tif_pattern}")
            continue
        tif = pathlib.Path(matching_files[0])
        if not tif.exists():
            print(f"!! missing {tif}")
            continue

        with rasterio.open(tif) as src:
            masked = mask_land_and_clip(
                src,
                cfg.preprocess.land_shapefile,
                cfg.preprocess.clip_percentile,
            )

            if cfg.preprocess.save_masked:
                out_tif = proc_root / f"{name}_masked.tif"
                write_masked(
                    src,
                    masked,
                    out_tif,
                    cfg.preprocess.masked_dtype,
                    compress=cfg.preprocess.compress,
                )

            quicklook_png = vis_root / f"{name}_preview.png"
            quicklook(src, masked, quicklook_png)

    print("✓ Preprocessing done.")


def write_masked(
    src: rasterio.DatasetReader,
    masked: np.ndarray,
    out_tif: pathlib.Path,
    dtype: str = "float32",
    compress: str = "LZW",
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
        invalid = ~valid
        data = scale_valid_masked_data(masked, dtype, valid)
        # set nans to nodata value
        data[invalid] = MASK_VALUE
        profile["dtype"] = dtype
    else:  # float32 passthrough
        profile["dtype"] = "float32"
        # replace NaNs with nodata
        masked = np.where(np.isfinite(masked), masked, MASK_VALUE)
        data = masked.astype(np.float32)

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(data, 1)


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
    out_png: pathlib.Path,
    rows: int = 2048,
):
    scale = rows / src.height
    cols = max(1, int(src.width * scale))
    # arr = src.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)

    # resample masked array instead using an in-memory rasterio dataset
    profile = src.profile.copy()
    profile.update(driver="MEM", nodata=MASK_VALUE)
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as mem:
            mem.write(masked, 1)
            arr = mem.read(1, out_shape=(rows, cols), resampling=Resampling.nearest)
    # create matplotlib image with lat-lon coordinates
    fig, ax = plt.subplots(figsize=(cols / 200, rows / 200), dpi=300)
    lat = np.linspace(src.bounds.top, src.bounds.bottom, rows)  # top to bottom
    lon = np.linspace(src.bounds.left, src.bounds.right, cols)  # left to right
    im = ax.imshow(
        np.nan_to_num(arr),
        extent=(lon[0], lon[-1], lat[0], lat[-1]),
        cmap="gray",
        vmin=np.nanmin(arr),
        vmax=np.nanmax(arr),
    )
    ax.set_aspect("equal")
    ax.set_title(os.path.basename(src.name))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    fig.colorbar(im, ax=ax, label="SAR intensity", fraction=0.046, pad=0.04)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    main()
