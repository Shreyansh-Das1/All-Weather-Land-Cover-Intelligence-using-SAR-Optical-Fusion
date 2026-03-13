import rasterio
import numpy as np

with rasterio.open("data\Processed\Clipped_VV.tif") as src:
    vv = src.read(1)
    profile = src.profile

with rasterio.open("data\Processed\Clipped_VH.tif") as src:
    vh = src.read(1)

sar_stack = np.stack([vv,vh])
profile.update(count=2)

with rasterio.open("sar_stack.tif","w",**profile) as dst:
    dst.write(sar_stack)