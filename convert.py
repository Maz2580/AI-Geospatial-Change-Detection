import os
import rasterio
from rasterio.transform import Affine

def convert_jpg_jgw_to_tif(folder):
    jpg_files = [f for f in os.listdir(folder) if f.lower().endswith('.jpg')]
    jgw_files = [f for f in os.listdir(folder) if f.lower().endswith('.jgw')]
    
    if not jpg_files or not jgw_files:
        return
        
    jpg_file = jpg_files[0]
    jgw_file = jgw_files[0]
    
    jpg_path = os.path.join(folder, jpg_file)
    jgw_path = os.path.join(folder, jgw_file)
    tif_path = os.path.join(folder, jpg_file.replace('.jpg', '.tif'))
    
    print(f"Reading world file: {jgw_path}")
    with open(jgw_path, 'r') as f:
        lines = [float(line.strip()) for line in f.readlines() if line.strip()]
        
    A, D, B, E, C, F = lines
    transform = Affine(A, D, C, B, E, F)
    
    print(f"Reading image: {jpg_path}")
    with rasterio.open(jpg_path) as src:
        data = src.read()
        profile = {
            'driver': 'GTiff',
            'dtype': src.dtypes[0],
            'nodata': None,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'crs': 'EPSG:7855',
            'transform': transform,
            'photometric': 'RGB' if src.count == 3 else 'MINISBLACK'
        }
        
        with rasterio.open(tif_path, 'w', **profile) as dst:
            dst.write(data)
            
    print(f"Created GeoTIFF: {tif_path}")

base_dir = r"C:\Users\maz.ghasemi\Downloads\Maz - 2 July 2025\python\change detection"
folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('venv')]

for folder in folders:
    try:
        convert_jpg_jgw_to_tif(os.path.join(base_dir, folder))
    except Exception as e:
        print(f"Error processing {folder}: {e}")
