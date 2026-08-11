import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.features import bounds as get_bounds
from rasterio.warp import transform_bounds, transform_geom
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("preview_candidates")

def _draw_polygon(img, geom_crs, src, window):
    """Draw a polygon geometry on the image."""
    img_copy = img.copy()
    
    # Handle both Polygon and MultiPolygon
    if geom_crs["type"] == "Polygon":
        rings = geom_crs["coordinates"]
    elif geom_crs["type"] == "MultiPolygon":
        rings = [ring for poly in geom_crs["coordinates"] for ring in poly]
    else:
        return img_copy
        
    for ring in rings:
        pts = []
        for lon, lat in ring:
            row, col = src.index(lon, lat)
            rel_row = int(row - window.row_off)
            rel_col = int(col - window.col_off)
            pts.append([rel_col, rel_row])
            
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
        
    return img_copy

def generate_previews(before_path, after_path, geojson_path, output_dir, max_candidates=None, buffer_m=30):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(geojson_path) as f:
        data = json.load(f)
        
    features = data.get("features", [])
    if not features:
        logger.info("No candidates found in GeoJSON.")
        return
        
    logger.info(f"Generating previews for {len(features)} candidates...")
    
    # Optionally cap the number of previews to prevent taking forever
    if max_candidates is not None and len(features) > max_candidates:
        logger.warning(f"Capping previews to top {max_candidates}.")
        features = features[:max_candidates]
        
    with rasterio.open(before_path) as src_b, rasterio.open(after_path) as src_a:
        for i, feature in enumerate(features):
            geom = feature["geometry"]
            props = feature["properties"]
            cand_id = props.get("candidate_id", i)
            area = props.get("area_m2", 0)
            sources = ", ".join(props.get("candidate_sources", [])) if "candidate_sources" in props else props.get("classification", "unknown")
            
            # Get bounds in EPSG:4326
            west, south, east, north = get_bounds(geom)
            
            # Transform to image CRS
            if src_a.crs and src_a.crs.to_epsg() != 4326:
                w, s, e, n = transform_bounds("EPSG:4326", src_a.crs, west, south, east, north)
                geom_crs = transform_geom("EPSG:4326", src_a.crs, geom)
            else:
                w, s, e, n = west, south, east, north
                geom_crs = geom
                
            # Add buffer
            w -= buffer_m
            s -= buffer_m
            e += buffer_m
            n += buffer_m
            
            # Read window
            window_a = src_a.window(w, s, e, n)
            window_b = src_b.window(w, s, e, n)
            
            img_a = src_a.read((1,2,3), window=window_a)
            img_b = src_b.read((1,2,3), window=window_b)
            
            if img_a.size == 0 or img_b.size == 0:
                continue
                
            img_a = np.transpose(img_a, (1, 2, 0))
            img_b = np.transpose(img_b, (1, 2, 0))
            
            # Normalize for display
            def norm(img):
                out = np.zeros_like(img, dtype=np.uint8)
                for b in range(3):
                    band = img[:,:,b]
                    v = band[band > 0]
                    if v.size > 0:
                        p2, p98 = np.percentile(v, (2, 98))
                        out[:,:,b] = np.clip((band - p2) * 255.0 / max(p98 - p2, 1.0), 0, 255)
                return out
                
            img_a = norm(img_a)
            img_b = norm(img_b)
            
            # Draw outlines
            img_a = _draw_polygon(img_a, geom_crs, src_a, window_a)
            img_b = _draw_polygon(img_b, geom_crs, src_b, window_b)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.imshow(img_b)
            ax1.set_title("Before (2021)")
            ax1.axis("off")
            
            ax2.imshow(img_a)
            ax2.set_title(f"After (2026)\nID: {cand_id} | Area: {area:.1f} m² | {sources}")
            ax2.axis("off")
            
            out_path = output_dir / f"candidate_{cand_id:03d}.png"
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved preview: {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--geojson", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args()
    
    generate_previews(
        before_path=args.before,
        after_path=args.after,
        geojson_path=args.geojson,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates
    )
