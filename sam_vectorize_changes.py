import os
import rasterio
import numpy as np
from scipy import ndimage
import geopandas as gpd
from shapely.geometry import Point
from samgeo import SamGeo
import rasterio.transform

def extract_blob_centroids(mask_path, min_pixels=50):
    """
    Reads a binary change mask (e.g. from CFNet), finds connected blobs,
    and returns their (lon, lat) centroids.
    """
    print(f"Reading change mask from {mask_path}...")
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    # Threshold the mask (in case it's 0-255)
    binary_mask = mask > 128
    
    # Find connected components
    labeled_array, num_features = ndimage.label(binary_mask)
    print(f"Found {num_features} raw change blobs.")
    
    # Calculate sizes and filter small noise
    sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
    valid_blobs = np.where(sizes >= min_pixels)[0] + 1
    print(f"Filtered down to {len(valid_blobs)} significant blobs (>{min_pixels} pixels).")
    
    # Find centroids (row, col)
    centroids = ndimage.center_of_mass(binary_mask, labeled_array, valid_blobs)
    
    # Convert pixel (row, col) to geospatial (lon, lat)
    points_geo = []
    for row, col in centroids:
        lon, lat = rasterio.transform.xy(transform, row, col)
        points_geo.append([lon, lat])
        
    return points_geo, crs

def main():
    change_map_path = "output/cfnet_change_map.tif"
    image_2026_path = "EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
    output_vector = "output/sam_change_polygons.geojson"
    
    if not os.path.exists(change_map_path):
        print(f"Error: Could not find {change_map_path}. Wait for CFNet to finish first!")
        return

    # 1. Extract centroids from the AI change map
    centroids, crs = extract_blob_centroids(change_map_path, min_pixels=100)
    
    if len(centroids) == 0:
        print("No significant changes found to vectorize.")
        return

    # 2. Initialize SAM (Segment Anything)
    print("\nInitializing SAM (Segment Anything Model)...")
    # For CPU, we use the smallest model (vit_b) or you can switch to FastSamGeo if installed
    # By default, SamGeo downloads the checkpoint automatically if not present.
    sam = SamGeo(
        model_type="vit_b", 
        device="cpu", # Force CPU to avoid CUDA errors on your machine
        sam_kwargs=None
    )
    
    # 3. Prompt SAM with our coordinates to generate exact building footprints
    print(f"Prompting SAM with {len(centroids)} locations on the 2026 image...")
    # point_labels=[1] means "foreground" (we want to select the object at the point)
    labels = [1] * len(centroids)
    
    sam.predict(
        image=image_2026_path, 
        point_coords=centroids, 
        point_labels=labels, 
        output=output_vector,
        crs=crs
    )
    
    print(f"\nSuccess! Highly accurate vector polygons saved to: {output_vector}")

if __name__ == "__main__":
    main()
