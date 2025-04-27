import os
from PIL import Image

# Folder paths
input_folder = "../converted_jpgs"
output_folder = os.path.join(input_folder, "resized_640x640")
os.makedirs(output_folder, exist_ok=True)

# List all image files
image_files = [
    f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))
]

for filename in sorted(image_files):
    input_path = os.path.join(input_folder, filename)

    try:
        with Image.open(input_path) as img:
            resized_img = img.resize((640, 640))
            base_name = os.path.splitext(filename)[0]
            output_name = f"{base_name}_640x640.jpg"
            output_path = os.path.join(output_folder, output_name)

            resized_img.save(output_path, format="JPEG")
            print(f"Saved: {output_name}")

    except Exception as e:
        print(f"Skipping {filename}: {e}")

print("All images resized to 640x640 and saved.")
