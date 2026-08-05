# AI Geospatial Change Detection

This repository contains an end-to-end suite of Python pipelines for detecting changes in ultra-high-resolution aerial imagery (GeoTIFFs). It replaces premium subscriptions (like Nearmap AI) with state-of-the-art open-source models that you can run locally on your own data.

## Repository Structure

The project is organized into a modular pipeline:

```text
change_detection/
├── src/                                  # AI Pipelines
│   ├── pixel_change/                     # Finds "WHERE" a change occurred
│   │   ├── cfnet_inference.py            # Supervised CNN detection
│   │   └── dinov2_inference.py           # Zero-shot Vision Transformer detection
│   │
│   ├── semantic_change/                  # Finds "WHAT" changed (e.g. Pools/Buildings)
│   │   ├── extract_features.py           # YOLO/HF Object footprint extraction
│   │   └── detect_semantic_changes.py    # Spatial subtraction of features
│   │
│   ├── vectorization/                    # Pixel-to-Polygon refinement
│   │   └── sam_vectorize_changes.py      # SAM-based perfect footprint generator
│   │
│   └── utils/                            # Legacy & Helper scripts
│       ├── convert.py                    
│       └── run_change_detection.py       
│
├── data/                                 # Data storage (Ignored by Git)
│   ├── input/                            # Raw TIFs and Zips
│   ├── weights/                          # AI Model Weights
│   └── output/                           # Generated TIFs and GeoJSONs
```

## Setup & Installation

1. Create a Python virtual environment: `python -m venv venv`
2. Activate the environment: `venv\Scripts\activate`
3. Install PyTorch for CPU:
   `pip install torch torchvision torchaudio`
4. Install geospatial requirements:
   `pip install rasterio geopandas shapely tqdm python-dotenv`
5. Install AI requirements:
   `pip install transformers huggingface_hub ultralytics segment-geospatial`
6. Add your HuggingFace Token to a `.env` file in the root folder:
   `HF_TOKEN=your_token_here`

## Pipeline 1: Pixel-Based Change Detection

These scripts compare two raw GeoTIFFs (2021 vs 2026) and generate a black-and-white "Change Mask" showing exactly where pixels have changed.

- **CFNet**: Run `python src/pixel_change/cfnet_inference.py` for highly accurate, supervised change detection.
- **DINOv2**: Run `python src/pixel_change/dinov2_inference.py` for zero-shot semantic change detection based on Meta's foundation models.

*Output: `data/output/cfnet_change_map.tif`*

## Pipeline 2: Semantic Change Detection (Object level)

If you only want to know about specific newly built objects (e.g., "Show me all the swimming pools built between 2021 and 2026"):

1. Update `extract_features.py` to point to your 2021 map, and run it:
   `python src/semantic_change/extract_features.py` (Outputs `pools_2021.geojson`)
2. Update `extract_features.py` to point to your 2026 map, and run it:
   `python src/semantic_change/extract_features.py` (Outputs `pools_2026.geojson`)
3. Run the spatial difference script:
   `python src/semantic_change/detect_semantic_changes.py`

*Output: `data/output/new_constructions.geojson`*

## Pipeline 3: AI Vectorization (SAM)

If you have a rough change mask from Pipeline 1, you can use Meta's Segment Anything Model (SAM) to draw perfect vector polygons around the changed objects automatically.

1. Ensure Pipeline 1 has successfully output a `change_map.tif`.
2. Run `python src/vectorization/sam_vectorize_changes.py`

*Output: `data/output/sam_change_polygons.geojson`*
