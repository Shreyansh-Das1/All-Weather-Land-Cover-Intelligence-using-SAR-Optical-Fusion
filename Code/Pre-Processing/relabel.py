import rasterio
import numpy as np
import glob

label_files = glob.glob(r"data\dataset\labels\*.tif")

for f in label_files:

    with rasterio.open(f) as src:
        labels = src.read(1)
        meta = src.meta

    remapped = np.zeros_like(labels)

    remapped[np.isin(labels,[10,20,30,40])] = 1
    remapped[labels==50] = 2
    remapped[labels==60] = 3
    remapped[labels==80] = 4

    with rasterio.open(f, "w", **meta) as dst:
        dst.write(remapped,1)
print("Done")
print(np.unique(tile))