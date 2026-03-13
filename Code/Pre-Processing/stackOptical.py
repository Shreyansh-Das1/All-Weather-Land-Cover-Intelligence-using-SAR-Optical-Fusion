import rasterio
import numpy as np

bands = []

for b in ["B02.jp2","B03.jp2","B04.jp2","B08.jp2"]:
    with rasterio.open(b) as src:
        bands.append(src.read(1))
        profile = src.profile

optical_stack = np.stack(bands)
print(optical_stack.shape)

profile.update(count=4)

with rasterio.open("optical_stack.tif","w",**profile) as dst:
    dst.write(optical_stack)