import os
import cv2
import rasterio
from rasterio.windows import Window
import geopandas as gpd
from shapely.geometry import box
from tqdm import tqdm

def main():
    # To run this, you would need `pip install ultralytics`
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Please run: pip install ultralytics")
        return

    # You can swap this to the 2021 map to generate pools_2021.geojson
    image_path = "../../data/input/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
    output_geojson = "../../data/output/pools_2026.geojson"
    
    # For this script to function flawlessly on swimming pools, you can load a specialized YOLO weight.
    # For now, we load the standard YOLOv8n (which detects generic objects like cars, people, boats) 
    # to demonstrate the pipeline structure.
    print("Loading Object Detection Model (YOLO)...")
    model = YOLO('yolov8n.pt') 

    print(f"Opening {image_path} to extract semantic features...")
    
    detected_polygons = []
    
    with rasterio.open(image_path) as src:
        transform = src.transform
        patch_size = 640 # YOLO optimal input size
        
        for row in tqdm(range(0, src.height, patch_size), desc="Detecting Features"):
            for col in range(0, src.width, patch_size):
                window = Window(col, row, min(patch_size, src.width - col), min(patch_size, src.height - row))
                
                # rasterio reads as (C, H, W). YOLO expects (H, W, C).
                img_data = src.read(window=window)[:3].transpose(1, 2, 0)
                
                # Run YOLO inference
                results = model.predict(img_data, verbose=False)
                
                for r in results:
                    boxes = r.boxes
                    for b in boxes:
                        # Get bounding box coordinates in pixel space (relative to the tile)
                        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
                        cls = int(b.cls[0].cpu().numpy())
                        conf = float(b.conf[0].cpu().numpy())
                        
                        # Only keep high confidence detections
                        if conf > 0.5:
                            # Convert local tile pixel coords to global image pixel coords
                            global_x1, global_y1 = col + x1, row + y1
                            global_x2, global_y2 = col + x2, row + y2
                            
                            # Convert pixel coords to real-world Geospatial coords (Lat/Lon)
                            geo_x1, geo_y1 = rasterio.transform.xy(transform, global_y1, global_x1)
                            geo_x2, geo_y2 = rasterio.transform.xy(transform, global_y2, global_x2)
                            
                            # Create a Shapely polygon for the bounding box footprint
                            geom = box(geo_x1, geo_y2, geo_x2, geo_y1)
                            
                            detected_polygons.append({
                                'geometry': geom,
                                'confidence': conf,
                                'class': cls
                            })
                
    if len(detected_polygons) > 0:
        import pandas as pd
        df = pd.DataFrame(detected_polygons)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=src.crs)
        
        os.makedirs(os.path.dirname(output_geojson), exist_ok=True)
        gdf.to_file(output_geojson, driver='GeoJSON')
        print(f"Saved {len(detected_polygons)} features to {output_geojson}")
    else:
        print("No features detected.")

if __name__ == "__main__":
    main()
