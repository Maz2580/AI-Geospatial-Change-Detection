import rasterio
import numpy as np
import os

t1_path = "EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
t2_path = "EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"

os.makedirs("output", exist_ok=True)
out_path = "output/change_detection_magnitude.tif"

print("Reading older date...")
with rasterio.open(t1_path) as src1:
    meta = src1.meta.copy()
    data1 = src1.read().astype(np.float32)

print("Reading newer date...")
with rasterio.open(t2_path) as src2:
    data2 = src2.read().astype(np.float32)

print("Calculating image difference (magnitude)...")
min_h = min(data1.shape[1], data2.shape[1])
min_w = min(data1.shape[2], data2.shape[2])

data1 = data1[:, :min_h, :min_w]
data2 = data2[:, :min_h, :min_w]

# Calculate magnitude of change across RGB channels
# (Absolute difference between the two dates)
diff = np.abs(data2 - data1)

# To make it more visible, we can just save it as-is, or average the channels for a heatmap
# Let's just output the RGB difference directly
meta.update(
    dtype=rasterio.float32,
    height=min_h,
    width=min_w
)

print("Writing output GeoTIFF...")
with rasterio.open(out_path, 'w', **meta) as dst:
    dst.write(diff)

print(f"Success! Saved change detection result to {out_path}")
