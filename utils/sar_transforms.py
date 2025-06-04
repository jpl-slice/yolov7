"""
Common SAR-image utilities: land masking + high-tail clipping.

All functions keep land / nodata pixels as NaN first; callers can then
convert NaN → 0 or rescale to uint{8,16} as required.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import box, shape
from rasterio import windows
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom
import numpy as np

NODATA_DEFAULT = 0  # input sentinel
MASK_VALUE = 0.0  # value we commit to disk

def get_nodata_from_src(src: rasterio.DatasetReader) -> float:
    # return self.src.nodata if self.src.nodata is not None else -9999.0
    if src.nodata is not None:
        return src.nodata
    else:
        data = src.read(1, window=windows.Window(0,0, min(128, src.width), min(128, src.height)) , boundless=True, fill_value=0) # Read a sample
        if np.any(data < -9000): # Heuristic from user
            # If nodata is not set, assume a common nodata value for SAR data
            return -9999.0
        else:
            return 0.0 # Default to 0 if not otherwise determined.

from scipy import ndimage

def dilate_land_mask(masked_data: np.ndarray, n_pixels: int = 2) -> np.ndarray:
    """
    Given a 2D array `masked_data` of float32 where:
      - ocean/backscatter pixels are numeric
      - land pixels are np.nan
    This will dilate the land mask by `n_pixels` in all directions, and
    return a new array in which those “grown” pixels are also set to np.nan.

    Parameters
    ----------
    masked_data : np.ndarray (2D, float32)
        Output from your existing “footprint‐clip → mask” step, where land = np.nan.
    n_pixels : int
        Number of pixels to dilate outward (in each direction).
        For a total morphological radius of `n_pixels`.

    Returns
    -------
    np.ndarray
        A new 2D float32 array, identical to `masked_data` except that
        the NaN land regions have been grown by `n_pixels` pixels.
    """
    # 1) Build a boolean mask: True where land (NaN), False where ocean/backscatter
    land_mask = np.isnan(masked_data)

    # 2) Create a square structuring element of size (2*n_pixels + 1)²
    #    For example, n_pixels=2 → footprint = 5×5 ones.
    footprint = np.ones((2 * n_pixels + 1, 2 * n_pixels + 1), dtype=bool)

    # 3) Perform binary dilation: expand True regions by that footprint
    dilated_mask = ndimage.binary_dilation(land_mask, structure=footprint)
    
    # try binary closing instead of dilation
    # dilated_mask = ndimage.binary_opening(land_mask, structure=footprint)

    # 4) Copy original data and set newly dilated pixels to NaN
    out = masked_data.copy()
    out[dilated_mask] = np.nan

    return out

def mask_land_and_clip(src: rasterio.DatasetReader, land_shapefile: str, clip_percentile: float | None) -> np.ndarray:
    """
    Updated mask_land_and_clip for UTM‐projected rasters:
    1) Read SAR band → convert native nodata → np.nan.
    2) Build the *exact* UTM footprint polygon from src.bounds.
    3) Reproject that footprint → EPSG:4326, convert to Shapely.
    4) Clip the EPSG:4326 land_shapefile to that exact footprint.
    5) Reproject clipped land polygons back to UTM.
    6) Call rio_mask with invert=True to set land→np.nan in the SAR array.
    7) Apply high‐tail clipping if requested.
    """
    import rasterio
    from shapely.geometry import box, shape, mapping
    import geopandas as gpd
    from rasterio.warp import transform_geom
    from rasterio.mask import mask as rio_mask

    # 1) Read SAR data and convert nodata → np.nan
    data = src.read(1).astype(np.float32)
    nodata_val = get_nodata_from_src(src)
    data[data == nodata_val] = np.nan

    # 2) Build the exact UTM footprint from src.bounds
    minx, miny, maxx, maxy = src.bounds
    footprint_utm = box(minx, miny, maxx, maxy)  # shapely Polygon in UTM

    # 3) Reproject footprint polygon → EPSG:4326 (GeoJSON dict)
    utm_crs = src.crs.to_string()
    footprint_geojson = transform_geom(
        utm_crs,           # e.g., "EPSG:32633"
        "EPSG:4326",       # lat–lon
        mapping(footprint_utm),
    )

    # Convert GeoJSON dict → Shapely geometry, then wrap in a GeoDataFrame
    footprint_shape_ll = shape(footprint_geojson)
    fp_gdf = gpd.GeoDataFrame(geometry=[footprint_shape_ll], crs="EPSG:4326")

    # 4) Read & clip the Natural Earth land shapefile (assumed EPSG:4326)
    land = gpd.read_file(land_shapefile)
    if land.crs is None:
        land = land.set_crs("EPSG:4326")

    # Clip to the rotated footprint in geographic coordinates
    clipped_ll = gpd.clip(land, fp_gdf)
    if clipped_ll.empty:
        # If no overlap, fall back to the full land layer
        clipped_ll = land.copy()

    # 5) Reproject clipped polygons back into UTM
    clipped_utm = clipped_ll.to_crs(src.crs)
    shapes_utm = [geom.__geo_interface__ for geom in clipped_utm.geometry]

    # 6) Mask the UTM raster: invert=True sets pixels *inside* clipped_utm → np.nan
    masked_arr, _ = rio_mask(
        src,
        shapes_utm,
        invert=True,
        nodata=np.nan,
        filled=True,
        crop=False
    )
    masked_data = masked_arr[0].astype(np.float32)

    # check land mask to see if there's land in this scene. if there's land, remove 99.9th percentile because that's probably unmasked land (imperfect land masking)
    # If at least one land pixel was masked, then
    land_masked_anywhere = np.any(np.isnan(masked_data) & ~np.isnan(data))
    if land_masked_anywhere:
        # grow the land‐mask by 12 pixels in every direction:
        masked_data = dilate_land_mask(masked_data, n_pixels=12)        
        # # Compute the 99.9th percentile over all non‐NaN values in masked_data
        # threshold = np.nanpercentile(masked_data, 99.9)
        # # Set any pixel brighter than that to NaN
        # masked_data[masked_data > threshold] = np.nan 
        # print(f"Land masking applied. Removed threshold: {threshold:.2f} to feather out land pixels.")
    
    
    # 7) Optional high‐tail clipping on the masked ocean pixels
    if clip_percentile is not None:
        hi = np.nanpercentile(masked_data, clip_percentile)
        # masked_data[masked_data > hi] = np.nan
        # actual clip; don't set to nan
        masked_data = np.clip(masked_data, a_min=None, a_max=hi)

    return masked_data
