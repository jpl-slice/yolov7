"""
Land masking, nodata handling and intensity post-processing
for single-polarisation SAR backscatter images.

Public API
──────────
• build_land_masker(shapefile_path) -> callable
• mask_land_and_clip(src, land_masker, clip_percentile, dilate_px) -> np.ndarray
• dilate_land_mask(arr, n_pixels)  (exposed for reuse)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Tuple

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features, windows
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely.geometry import box, mapping

# ──────────────── constants ─────────────────
NODATA_DEFAULT = 0.0
MASK_VALUE = 0.0  # value committed to disk


# ──────────────── public helpers ────────────
def build_land_masker(shapefile: str) -> Callable[[rasterio.CRS, dict], list[dict]]:
    """
    Load land polygons once; return a closure that clips/reprojects
    them on demand for *any* UTM scene.

    Returns
    -------
    land_masker(crs, bounds_geojson) -> list[dict]  (GeoJSON shapes in target CRS)
    """

    land_ll = gpd.read_file(shapefile)
    if land_ll.crs is None:
        land_ll = land_ll.set_crs("EPSG:4326")

    @lru_cache(maxsize=256)
    def _clip_to_scene(
        crs_wkt: str, bounds_tuple: tuple[float, float, float, float]
    ) -> tuple[dict, ...]:
        # 1) footprint as Shapely in lat-lon
        minx, miny, maxx, maxy = bounds_tuple
        footprint_ll = box(minx, miny, maxx, maxy)
        clipped = gpd.clip(
            land_ll, gpd.GeoDataFrame(geometry=[footprint_ll], crs="EPSG:4326")
        )
        if clipped.empty:
            clipped = land_ll  # rare offshore frame → keep full coastlines
        return tuple(
            transform_geom("EPSG:4326", crs_wkt, mapping(geom))  # GeoJSON dicts
            for geom in clipped.geometry
        )

    # The closure we hand back performs *only* cheap dict boxing
    def _land_masker(crs: rasterio.CRS, bounds) -> list[dict]:
        return list(_clip_to_scene(crs.to_string(), bounds))

    return _land_masker


def mask_land_and_clip(
    src: rasterio.DatasetReader,
    land_masker: Callable[
        [rasterio.CRS, tuple[float, float, float, float]], list[dict]
    ],
    *,
    clip_percentile: float | None | Tuple[float, float] = 99.9,
    dilate_px: int = 0,
) -> np.ndarray:
    """Main algorithm: (1) nodata→NaN (2) land mask (3) dilate (4) optional clip."""
    shapes_utm = land_masker(src.crs, _native_bounds_as_ll(src))
    masked_arr, _ = rio_mask(
        src, shapes_utm, invert=True, nodata=np.nan, filled=True, crop=False
    )
    out = masked_arr[0].astype("float32")
    out[out == _get_nodata(src)] = np.nan  # ensure nodata is NaN

    if dilate_px > 0:
        out = dilate_land_mask(out, dilate_px)

    if clip_percentile is not None:
        if isinstance(clip_percentile, tuple):
            lo, hi = clip_percentile
            lo = np.nanpercentile(out, lo)
            hi = np.nanpercentile(out, hi)
        elif isinstance(clip_percentile, (int, float)):
            lo = np.nanpercentile(out, 0)
            hi = np.nanpercentile(out, clip_percentile)
        else:
            raise ValueError(
                f"clip_percentile must be a tuple or a number, got {clip_percentile}"
            )
        out = np.clip(out, lo, hi)

    return out


def dilate_land_mask(arr: np.ndarray, n_pixels: int = 2) -> np.ndarray:
    """Morphologically grow NaN (land) regions by `n_pixels` in all directions."""
    land = np.isnan(arr)
    footprint = np.ones((2 * n_pixels + 1,) * 2, bool)
    dilated = ndimage.binary_dilation(land, structure=footprint)
    out = arr.copy()
    out[dilated] = np.nan
    return out


# ──────────────── private helpers ───────────
def _native_bounds_as_ll(
    src: rasterio.DatasetReader,
) -> tuple[float, float, float, float]:
    """Return (left, bottom, right, top) in EPSG:4326."""
    return rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)  # type: ignore[arg-type]


def _get_nodata(src: rasterio.DatasetReader) -> float:
    """Robust nodata detection with tiny warm-up cost."""
    if src.nodata is not None:
        return float(src.nodata)

    # Heuristic sample (≤128×128) from UL corner
    data = src.read(1)
    # return -9999.0 if np.any(sample < -9000) else NODATA_DEFAULT
    if np.any(data < -9000):
        return -9999.0
    elif np.any(np.isnan(data)):
        return np.nan
    else:
        return 0.0  # default nodata value


# ────────── scaling helper (exposed for re-use) ─────────
def scale_valid_masked_data(
    masked: np.ndarray,
    dtype: str,
    valid: np.ndarray,
    *,
    percentile: tuple[float, float] = (0, 99),
) -> np.ndarray:
    """Stretch valid pixels to full uint8/uint16 range; nodata stays 0."""
    if not np.any(valid):
        return np.zeros_like(masked, dtype=dtype)

    lo, hi = np.nanpercentile(masked[valid], percentile)
    if lo == hi:  # flat frame
        out = np.zeros_like(masked, dtype=dtype)
        out[valid] = 1
        return out

    scale = 255 if dtype == "uint8" else 65535
    usable = scale - 1
    out = np.zeros_like(masked, dtype=dtype)
    out[valid] = np.round(((masked[valid] - lo) / (hi - lo)) * usable + 1).astype(dtype)
    return out
