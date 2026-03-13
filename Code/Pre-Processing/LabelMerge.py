import json
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os

# ── Load projected AOI (EPSG:32645) ──────────────────────────────────────────
with open(r"data\aoi_projected.geojson") as f:
    aoi_geom = json.load(f)
geom = [aoi_geom]

# ── Step 1: Merge tiles (nodata=0) ────────────────────────────────────────────
src1 = rasterio.open("E084_Map.tif")
src2 = rasterio.open("E087_Map.tif")

mosaic, out_transform = merge([src1, src2], nodata=0, method="first")
merge_profile = src1.profile.copy()
merge_profile.update({
    "transform": out_transform,
    "width": mosaic.shape[2],
    "height": mosaic.shape[1],
    "nodata": 0,
    "driver": "GTiff"
})
src1.close()
src2.close()

print("Merged shape:", mosaic.shape)

# ── Step 2: Reproject merged to EPSG:32645 ────────────────────────────────────
target_crs = "EPSG:32645"
src_crs = merge_profile["crs"]

from rasterio.transform import array_bounds
bounds = array_bounds(mosaic.shape[1], mosaic.shape[2], out_transform)

dst_transform, dst_width, dst_height = calculate_default_transform(
    src_crs, target_crs,
    mosaic.shape[2], mosaic.shape[1],
    left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3],
    resolution=10
)

reprojected = np.zeros((1, dst_height, dst_width), dtype=mosaic.dtype)

reproject(
    source=mosaic,
    destination=reprojected,
    src_transform=out_transform,
    src_crs=src_crs,
    dst_transform=dst_transform,
    dst_crs=target_crs,
    resampling=Resampling.nearest,
    src_nodata=0,
    dst_nodata=0
)

reproject_profile = merge_profile.copy()
reproject_profile.update({
    "crs": target_crs,
    "transform": dst_transform,
    "width": dst_width,
    "height": dst_height,
    "nodata": 0
})

with rasterio.open("merged_labels_utm.tif", "w", **reproject_profile) as dst:
    dst.write(reprojected)

print(f"Reprojected shape: {dst_width} x {dst_height}")

# ── Step 3: Clip to projected AOI ─────────────────────────────────────────────
with rasterio.open("merged_labels_utm.tif") as src:
    clipped, clipped_transform = mask(src, geom, crop=True, nodata=0)
    profile = src.profile.copy()
    profile.update({
        "transform": clipped_transform,
        "width": clipped.shape[2],
        "height": clipped.shape[1],
        "nodata": 0
    })
    with rasterio.open("labels_clipped.tif", "w", **profile) as dst:
        dst.write(clipped)

os.remove("merged_labels_utm.tif")

# ── Verify ────────────────────────────────────────────────────────────────────
with rasterio.open("labels_clipped.tif") as src:
    data = src.read(1)
    print("Size:        ", src.width, "x", src.height)
    print("CRS:         ", src.crs)
    print("Bounds:      ", src.bounds)
    print("Unique vals: ", np.unique(data))
    print("% nodata:    ", round((data == 0).sum() / data.size * 100, 2), "%")