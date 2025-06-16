#!/usr/bin/env python3
"""
02_preprocess_sar_parallel.py - minimal parallel driver for 01_preprocess_sar.py
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import SourceFileLoader as SFL
from pathlib import Path

import numpy as np
import rasterio
from omegaconf import OmegaConf
from tqdm import tqdm

# ─── make project code importable ────────────────────────────────────────────
project_src = Path(__file__).resolve().parent.parent
print(f"Adding {project_src} to sys.path")
sys.path.insert(0, str(project_src))

from utils.sar_transforms import build_land_masker, mask_land_and_clip

# import the original single-thread driver as a module called “pre”
pre = SFL("pre", str(Path(__file__).with_name("01_preprocess_sar.py"))).load_module()


# ──────────────────────────────── main ───────────────────────────────────────
def main() -> None:
    args = _parse_cli()
    cfg = OmegaConf.load(args.cfg)

    raw_root = Path(os.path.expandvars(cfg.paths.raw_root))
    scene_names = pre.find_tif_files(args.selected, raw_root)
    if not scene_names:
        raise SystemExit(f"!! No scenes found in {raw_root}")

    # ensure output dirs exist before spawning workers
    Path(os.path.expandvars(cfg.paths.processed_root)).mkdir(
        parents=True, exist_ok=True
    )
    Path("data/visualisations").mkdir(parents=True, exist_ok=True)

    n_jobs = min(args.workers or os.cpu_count() or 1, len(scene_names))
    with ProcessPoolExecutor(
        max_workers=n_jobs, initializer=_init_worker, initargs=(args.cfg,)
    ) as pool:
        list(
            tqdm(
                pool.map(_process_scene, scene_names),
                total=len(scene_names),
                desc="preprocess-parallel",
            )
        )


# ───────────────────────── helper utilities ────────────────────────────────
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfg/preprocess_sar/preprocess_sar.yaml")
    p.add_argument("--selected", default="cfg/preprocess_sar/selected_files.json")
    p.add_argument(
        "-w",
        "--workers",
        type=int,
        help="parallel processes (default: all logical cores)",
    )
    return p.parse_args()


def _init_worker(cfg_path: str) -> None:
    """Runs once in each process - load config & pre-build heavy globals."""
    global CFG, RAW, PROC, VIS, LAND
    CFG = OmegaConf.load(cfg_path)
    RAW = Path(os.path.expandvars(CFG.paths.raw_root))
    PROC = Path(os.path.expandvars(CFG.paths.processed_root))
    VIS = Path("data/visualisations")
    LAND = build_land_masker(CFG.preprocess.land_shapefile)


def _process_scene(scene_name: str) -> str:
    files = sorted(RAW.glob(f"{scene_name}*.tif"))
    if not files:
        return scene_name  # nothing to do

    with rasterio.open(files[0]) as src:
        masked = mask_land_and_clip(
            src,
            LAND,
            clip_percentile=CFG.preprocess.clip_percentile,
            dilate_px=12,
        )

        if getattr(CFG.preprocess, "convert_to_db", True):
            masked = _to_db(masked)

        sid = pre.extract_scene_id(files[0].name)

        if CFG.preprocess.save_masked:
            pre.write_masked(
                src,
                masked,
                PROC / f"{sid}.tif",
                CFG.preprocess.masked_dtype,
                compress=CFG.preprocess.compress,
            )

        pre.quicklook(src, masked, VIS / f"{sid}_preview.png")

    return sid


def _to_db(arr: np.ndarray, floor: float = -35.0) -> np.ndarray:
    """Convert intensity → dB with a floor (tiny helper, 3 lines)."""
    valid = np.isfinite(arr)
    arr[valid] = 10 * np.log10(arr[valid] + 1e-10)  # add epsilon to avoid log(0)
    return np.clip(arr, floor, None)


# ────────────────────────────────── cli ──────────────────────────────────────
if __name__ == "__main__":
    main()
