import whitebox
import os

wbt = whitebox.WhiteboxTools()

# Set working directory to the current folder so paths can be relative
wbt.set_working_dir(os.path.dirname(os.path.abspath(__file__)))

# Define input files (older date and newer date)
t1 = "EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20211201_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"
t2 = "EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_VertJPEG-0000/EPSG7855_Date20260418_Lat-36.336606_Lon145.406921_Mpp0.075_Vert.tif"

# Create output folder
os.makedirs("output", exist_ok=True)
out_diff = "output/difference_magnitude.tif"
out_abs = "output/difference_absolute.tif"
out_signed = "output/difference_signed.tif"

print("Running Whitebox Image Difference Change Detection...")
# Run the tool (passing strings with semicolon separator if multiple, but here it's just one file each)
result = wbt.run_tool(
    "image_difference_change_detection",
    [
        f"--t1_inputs={t1}",
        f"--t2_inputs={t2}",
        f"--out_diff={out_diff}",
        f"--out_abs={out_abs}",
        f"--out_signed={out_signed}",
        "--difference_mode=magnitude"
    ]
)

print(f"Tool finished with result: {result}")
if result == 0:
    print("Success! Check the 'output' folder for the generated TIFFs.")
