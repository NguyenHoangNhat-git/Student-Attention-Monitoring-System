import os
from PIL import Image

"""
For each img in the folder, crop 2 out of 4 regions
Ex: img 1: crop regions 1-3, img 2: crop regions 2-4, img 3:crop regions 1-3, ....
"""

# Folder paths
folder_path = "../manual data"
output_root = os.path.join(folder_path, "cropped_diagonals")

# Define 4 crop regions (adjust these to fit your images!)
CROP_REGIONS = [  # (left, upper, right, lower)
    (0, 400, 640, 1040),  # Region 1
    (640, 400, 1280, 1040),  # Region 2
    (1280, 400, 1920, 1040),  # Region 3
    # (100, 0, 740, 640),  # Region 4
]

# Make subfolders for each region
region_folders = []
for i in range(4):
    region_folder = os.path.join(output_root, f"region{i + 1}")
    os.makedirs(region_folder, exist_ok=True)
    region_folders.append(region_folder)

# List all image files
image_files = [
    f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))
]

for index, filename in enumerate(sorted(image_files)):
    file_path = os.path.join(folder_path, filename)

    try:
        with Image.open(file_path) as img:
            img_w, img_h = img.size

            # Select regions based on even/odd index
            # if index % 2 == 0:
            #     selected_indices = [0, 2]  # Regions 1 & 3
            # else:
            #     selected_indices = [1, 3]  # Regions 2 & 4
            selected_indices = [0, 1, 2]

            for idx in selected_indices:
                crop_box = CROP_REGIONS[idx]

                # Ensure the crop box is inside image bounds
                crop_box = (
                    min(crop_box[0], img_w),
                    min(crop_box[1], img_h),
                    min(crop_box[2], img_w),
                    min(crop_box[3], img_h),
                )

                cropped_img = img.crop(crop_box)

                base_name = os.path.splitext(filename)[0]
                output_name = f"{base_name}_region{idx + 1}.jpg"
                output_path = os.path.join(region_folders[idx], output_name)

                cropped_img.save(output_path)
                print(f"Saved to {region_folders[idx]}: {output_name}")

    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("All images processed with diagonal cropping into region folders.")
