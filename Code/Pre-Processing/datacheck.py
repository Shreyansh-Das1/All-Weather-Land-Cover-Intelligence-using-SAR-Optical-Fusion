import rasterio
import numpy as np

import matplotlib.pyplot as plt

with rasterio.open("labels_clipped.tif") as src:
    data = src.read(1).astype(float)
    data[data == 0] = np.nan  # mask nodata as transparent
    
plt.figure(figsize=(10, 8))
plt.axis("off")
plt.imshow(data, cmap="tab10")
plt.colorbar(label="Land Cover Classes")
plt.title("Land Cover Ground Truth Labels")
plt.show()
'''
with rasterio.open("sar_stack.tif") as src:
    data = src.read(1)
    #print(f"\n{path}")
    print(f"  Bounds: {src.bounds}")
    print(f"  Size: {src.width} x {src.height}")
    print(f"  Unique values: {np.unique(data)}")
    print(f"  % black (0): {(data == 0).sum() / data.size * 100:.1f}%")

path = "labels_clipped.tif"

with rasterio.open("labels_clipped.tif") as src:
    print("Width:", src.width)
    print("Height:", src.height)
    print("CRS:", src.crs)
    print("Resolution:", src.res)
    print(src.transform)
    print(src.bounds)
    print(src.crs)'''
