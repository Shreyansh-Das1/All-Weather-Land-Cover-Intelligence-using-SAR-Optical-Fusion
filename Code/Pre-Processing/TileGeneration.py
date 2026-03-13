import numpy as np
import rasterio
import os

# ── Load clipped stacks ───────────────────────────────────────────────────────
with rasterio.open(r"data\Processed\optical_clipped.tif") as src:
    optical_stack = src.read().astype(np.float32)

with rasterio.open(r"data\Processed\sar_clipped.tif") as src:
    sar_stack = src.read().astype(np.float32)

with rasterio.open("labels_clipped.tif") as src:
    labels = src.read(1)

# Crop to match optical/SAR dimensions
labels = labels[:optical_stack.shape[1], :optical_stack.shape[2]]

print("Labels cropped to:", labels.shape)

print("Optical stack:", optical_stack.shape)
print("SAR stack:    ", sar_stack.shape)
print("Labels:       ", labels.shape)

# ── Verify all same spatial size ──────────────────────────────────────────────
assert optical_stack.shape[1:] == sar_stack.shape[1:] == labels.shape, \
    f"Size mismatch! optical={optical_stack.shape[1:]}, sar={sar_stack.shape[1:]}, labels={labels.shape}"

# ── Normalize ─────────────────────────────────────────────────────────────────
optical_norm = np.clip(optical_stack / 10000.0, 0, 1)

sar_db = 10 * np.log10(sar_stack + 1e-6)
sar_norm = np.zeros_like(sar_db)
for c in range(sar_db.shape[0]):
    mn, mx = sar_db[c].min(), sar_db[c].max()
    sar_norm[c] = (sar_db[c] - mn) / (mx - mn + 1e-8)

# ── Remap label values to 0-based indices ────────────────────────────────────
label_map = {10:1, 20:1, 30:1, 40:1, 50:2, 60:3, 80:4}
labels_remapped = np.zeros_like(labels)
for orig, new in label_map.items():
    labels_remapped[labels == orig] = new

print("Label classes:", np.unique(labels_remapped))

# ── Tile ──────────────────────────────────────────────────────────────────────
os.makedirs(r"data/dataset/optical", exist_ok=True)
os.makedirs(r"data/dataset/sar",     exist_ok=True)
os.makedirs(r"data/dataset/labels",  exist_ok=True)

TILE = 128
H, W = optical_norm.shape[1], optical_norm.shape[2]
tile_id = 1

for i in range(0, H - TILE + 1, TILE):
    for j in range(0, W - TILE + 1, TILE):
        name = f"{tile_id:04d}"
        np.save(f"data/dataset/optical/opt_{name}.npy", optical_norm[:, i:i+TILE, j:j+TILE])
        np.save(f"data/dataset/sar/sar_{name}.npy",     sar_norm[:,     i:i+TILE, j:j+TILE])
        np.save(f"data/dataset/labels/lbl_{name}.npy",  labels_remapped[i:i+TILE, j:j+TILE])
        tile_id += 1

print(f"Tiles saved: {tile_id - 1}")
print(f"Expected:    ~1595 (55×29)")
print(f"\nTile shapes:")
print(f"  optical: (4, 128, 128)")
print(f"  SAR:     (2, 128, 128)")
print(f"  labels:  (128, 128)")