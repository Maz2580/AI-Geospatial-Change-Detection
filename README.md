# Nearmap Change Detection Pipeline

This repository contains a Python-based change detection pipeline designed specifically for analyzing massive, high-resolution aerial imagery (such as Nearmap data). 

It leverages **CFNet**, a state-of-the-art content-aware deep learning architecture for remote sensing change detection, and seamlessly processes extremely large GeoTIFF images by tiling them, performing inference, and reconstructing the output seamlessly.

## 🚀 Features
- **Large-Scale Tiling Engine**: Automatically slices massive ~5000x8000+ `.tif` imagery into 512x512 patches suitable for deep learning.
- **Deep Learning Backbone**: Uses an EfficientNet-B5 based `CFNet` for robust detection of infrastructure changes, mitigating style variations caused by lighting and weather.
- **GeoTIFF Generation**: Automatically reassembles the predictions and outputs a perfectly aligned binary change mask as a new `.tif` file.
- **Automated Pre-processing**: Provides utility scripts (`convert.py`) to convert `.jpg` and `.jgw` world files directly into georeferenced TIF datasets.

## 🛠️ Components
- `cfnet_inference.py`: The core AI inference pipeline that slices the two temporal GeoTIFFs, runs the deep learning model, and stitches the output back together.
- `calculate_change.py`: A baseline algorithmic script that calculates absolute difference thresholds (useful for rapid sanity-checking without deep learning).
- `convert.py`: A utility script for transforming JPGs into TIF format utilizing corresponding JGW world files.
- `CFNet/`: A submodule containing the model definition and architecture.

## 📦 Setup Instructions

1. **Clone the repo with submodules**:
```bash
git clone --recursive https://github.com/Maz2580/change_detection.git
```

2. **Setup your Environment**:
Ensure you have Python installed, then install the PyTorch CPU version (or GPU if available) and the required scientific packages:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install timm einops scipy rasterio tqdm numpy
```

3. **Download Model Weights**:
Download the pre-trained `clcd.pth` weights and place them in the root directory as `cfnet_weights.pth`.
```python
import urllib.request
urllib.request.urlretrieve('https://huggingface.co/wifibk/CFNet/resolve/main/clcd.pth', 'cfnet_weights.pth')
```

## 🗺️ Usage
Execute the CFNet change detection inference script:
```bash
python cfnet_inference.py
```
This will read the two temporal GeoTIFF files specified in the script and output the AI's binary change map into the `output/` folder.

## 📋 License
This codebase is a bespoke pipeline leveraging the open-source research from CFNet (Optimizing Remote Sensing Change Detection through Content-Aware Enhancement).
