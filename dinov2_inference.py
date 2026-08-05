import os
import sys
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from transformers import AutoModel
from tqdm import tqdm
from dotenv import load_dotenv

def main():
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    t1_path = "EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
    t2_path = "EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
    out_path = "output/dinov2_change_map.tif"
    
    # Must be multiple of 14 for DINOv2. 14 * 37 = 518
    patch_size = 518 
    device = torch.device("cpu")
    
    os.makedirs("output", exist_ok=True)
    
    print("Loading DINOv2 Foundation Model from HuggingFace...")
    # dinov2-base is ~86M parameters (very fast compared to EfficientNet-B5)
    model = AutoModel.from_pretrained('facebook/dinov2-base', token=HF_TOKEN)
    model = model.to(device)
    model.eval()

    # DINOv2 ImageNet Normalization
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Opening images...")
    with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
        meta = src1.meta.copy()
        min_h = min(src1.height, src2.height)
        min_w = min(src1.width, src2.width)
        
        meta.update(
            dtype=rasterio.uint8,
            count=1, 
            height=min_h,
            width=min_w
        )

        with rasterio.open(out_path, 'w', **meta) as dst:
            for row in tqdm(range(0, min_h, patch_size), desc="Rows"):
                for col in range(0, min_w, patch_size):
                    window = Window(col, row, min(patch_size, min_w - col), min(patch_size, min_h - row))
                    
                    # Read and reorder to H,W,C for transforms
                    # rasterio reads as (C, H, W)
                    data1 = src1.read(window=window)[:3].transpose(1, 2, 0)
                    data2 = src2.read(window=window)[:3].transpose(1, 2, 0)
                    
                    pad_h = patch_size - data1.shape[0]
                    pad_w = patch_size - data1.shape[1]
                    
                    if pad_h > 0 or pad_w > 0:
                        data1 = np.pad(data1, ((0,pad_h), (0,pad_w), (0,0)), mode='reflect')
                        data2 = np.pad(data2, ((0,pad_h), (0,pad_w), (0,0)), mode='reflect')

                    # preprocess expects H,W,C numpy or PIL, returns C,H,W tensor
                    x1 = preprocess(data1).unsqueeze(0).to(device)
                    x2 = preprocess(data2).unsqueeze(0).to(device)

                    with torch.no_grad():
                        out1 = model(x1)
                        out2 = model(x2)
                        
                        # Extract patch tokens (discard CLS token at index 0)
                        # out.last_hidden_state shape: (1, seq_len, hidden_dim) -> (1, 1370, 768)
                        # 1369 = 37 * 37
                        feat1 = out1.last_hidden_state[:, 1:, :] 
                        feat2 = out2.last_hidden_state[:, 1:, :]
                        
                        # Cosine similarity along the feature dimension
                        sim = F.cosine_similarity(feat1, feat2, dim=-1) # Shape: (1, 1369)
                        
                        # Reshape to grid
                        grid_size = patch_size // 14 # 37
                        sim_grid = sim.view(1, 1, grid_size, grid_size) # Shape: (1, 1, 37, 37)
                        
                        # Upsample to patch_size (518x518)
                        sim_up = F.interpolate(sim_grid, size=(patch_size, patch_size), mode='bilinear', align_corners=False)
                        
                        # Threshold (similarity < threshold means change). 
                        # DINOv2 features are highly clustered. < 0.6 is a good starting threshold for differences.
                        change_mask = (sim_up < 0.6).byte().squeeze().cpu().numpy() # (518, 518)
                        
                    # Crop padding out
                    if pad_h > 0 or pad_w > 0:
                        change_mask = change_mask[:window.height, :window.width]
                        
                    # Write to TIF (255 for change, 0 for no change)
                    dst.write((change_mask * 255).astype(np.uint8), 1, window=window)

    print(f"Success! DINOv2 Change map saved to {out_path}")

if __name__ == "__main__":
    main()
