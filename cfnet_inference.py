import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torchvision import transforms
from tqdm import tqdm

# Add CFNet to python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'CFNet'))
from model.CFNet import CFNet

def main():
    t1_path = "EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
    t2_path = "EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
    out_path = "output/cfnet_change_map.tif"
    checkpoint_path = "cfnet_weights.pth" # We will download this next
    patch_size = 512
    device = torch.device("cpu") # User confirmed CPU only

    os.makedirs("output", exist_ok=True)

    print("Loading CFNet Model...")
    model = CFNet(3, 3)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        print("Weights loaded successfully!")
    else:
        print(f"WARNING: No weights found at {checkpoint_path}. Using random weights for testing pipeline.")
    
    model = model.to(device)
    model.eval()

    # Preprocessing transforms (assuming standard ImageNet normalization used by EfficientNet)
    # The paper uses images scaled to [0,1], no normalization specified in test.py, so we just scale to 0-1.
    def preprocess(img_array):
        # input is shape (C, H, W) in range 0-255
        img = torch.from_numpy(img_array).float() / 255.0
        return img.unsqueeze(0) # Add batch dim

    print(f"Opening images...")
    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        meta = src1.meta.copy()
        
        min_h = min(src1.height, src2.height)
        min_w = min(src1.width, src2.width)
        
        meta.update(
            dtype=rasterio.uint8,
            count=1, # Binary change map (1 channel)
            height=min_h,
            width=min_w
        )

        with rasterio.open(out_path, 'w', **meta) as dst:
            # We iterate in chunks
            for row in tqdm(range(0, min_h, patch_size), desc="Rows"):
                for col in range(0, min_w, patch_size):
                    window = Window(col, row, min(patch_size, min_w - col), min(patch_size, min_h - row))
                    
                    data1 = src1.read(window=window)[:3] # Ensure 3 channels
                    data2 = src2.read(window=window)[:3]
                    
                    # Pad to patch_size if at the edges
                    pad_h = patch_size - data1.shape[1]
                    pad_w = patch_size - data1.shape[2]
                    
                    if pad_h > 0 or pad_w > 0:
                        data1 = np.pad(data1, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')
                        data2 = np.pad(data2, ((0,0), (0,pad_h), (0,pad_w)), mode='reflect')

                    x1 = preprocess(data1).to(device)
                    x2 = preprocess(data2).to(device)

                    with torch.no_grad():
                        with torch.amp.autocast(device.type):                
                            y_change, _, _, _, _ = model(x1, x2, device)
                        
                        # y_change shape is (1, 512, 512), so we just take [0] to get (512, 512)
                        pred = (y_change > 0.5).byte().cpu().numpy()[0] # shape (patch_size, patch_size)

                    # Crop padding out
                    if pad_h > 0 or pad_w > 0:
                        pred = pred[:window.height, :window.width]
                        
                    if pred.shape != (window.height, window.width):
                        print(f"Shape mismatch! pred: {pred.shape}, window: {window.height}x{window.width}, y_change: {y_change.shape}")
                        pred = pred.reshape(window.height, window.width)

                    # Save (multiply by 255 so it shows up bright white in viewer)
                    dst.write((pred * 255).astype(np.uint8), 1, window=window)

    print(f"Success! Change map saved to {out_path}")

if __name__ == "__main__":
    main()
