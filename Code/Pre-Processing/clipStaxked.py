import rasterio
from rasterio.mask import mask
import json

with open(r"data\aoi_projected.geojson") as f:
    aoi = [json.load(f)]

def clip(src, name):
    clipped, transform = mask(src, aoi, crop=True)
    profile = src.profile
    profile.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": transform
    })
    with rasterio.open(name,"w",**profile) as dst:
        dst.write(clipped)

clip(rasterio.open("optical_stack.tif") , "optical_clipped.tif")
clip(rasterio.open("sar_stack.tif"), "sar_clipped.tif")