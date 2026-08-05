import os
import geopandas as gpd
import pandas as pd

def detect_semantic_change(old_geojson, new_geojson, new_output, demo_output):
    """
    Compares two periods of extracted features (e.g., pools in 2021 vs pools in 2026).
    Identifies completely new pools, and pools that were demolished/filled in.
    """
    print(f"Loading 2021 features from {old_geojson}...")
    if not os.path.exists(old_geojson):
        print(f"Error: {old_geojson} not found. Did you run extract_features.py on the 2021 map?")
        return
    gdf_old = gpd.read_file(old_geojson)
    
    print(f"Loading 2026 features from {new_geojson}...")
    if not os.path.exists(new_geojson):
        print(f"Error: {new_geojson} not found. Did you run extract_features.py on the 2026 map?")
        return
    gdf_new = gpd.read_file(new_geojson)
    
    print(f"Found {len(gdf_old)} objects in 2021, and {len(gdf_new)} objects in 2026.")

    # 1. FIND NEW CONSTRUCTIONS (Features in 2026 that DO NOT intersect any feature in 2021)
    # We use a spatial join. 'predicate=intersects' means we find where they overlap.
    # If a 2026 polygon doesn't overlap with any 2021 polygon, it's brand new!
    print("Calculating New Constructions...")
    joined_new = gpd.sjoin(gdf_new, gdf_old, how="left", predicate="intersects")
    # Features where index_right is NaN had no overlap in 2021.
    new_features = joined_new[joined_new["index_right"].isna()].copy()
    
    # 2. FIND DEMOLITIONS (Features in 2021 that DO NOT intersect any feature in 2026)
    print("Calculating Demolitions...")
    joined_old = gpd.sjoin(gdf_old, gdf_new, how="left", predicate="intersects")
    demolished_features = joined_old[joined_old["index_right"].isna()].copy()
    
    # Save outputs
    os.makedirs(os.path.dirname(new_output), exist_ok=True)
    
    if len(new_features) > 0:
        new_features.drop(columns=['index_right'], inplace=True, errors='ignore') # clean up
        new_features.to_file(new_output, driver="GeoJSON")
        print(f"SUCCESS: Found {len(new_features)} brand new constructions! Saved to {new_output}")
    else:
        print("No new constructions found.")
        
    if len(demolished_features) > 0:
        demolished_features.drop(columns=['index_right'], inplace=True, errors='ignore')
        demolished_features.to_file(demo_output, driver="GeoJSON")
        print(f"SUCCESS: Found {len(demolished_features)} demolished features! Saved to {demo_output}")
    else:
        print("No demolitions found.")

if __name__ == "__main__":
    # In a real workflow, you'd run extract_features.py twice to generate these inputs:
    old_file = "../../data/output/pools_2021.geojson"
    new_file = "../../data/output/pools_2026.geojson"
    
    new_out = "../../data/output/new_constructions.geojson"
    demo_out = "../../data/output/demolished_constructions.geojson"
    
    # To prevent errors if the user hasn't run the extract scripts yet:
    if not os.path.exists(old_file) or not os.path.exists(new_file):
        print("Waiting for feature extraction GeoJSON files to be generated first.")
    else:
        detect_semantic_change(old_file, new_file, new_out, demo_out)
