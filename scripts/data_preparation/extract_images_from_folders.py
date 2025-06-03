import os
import shutil

# Set your source and destination folders
source_root = '/home/nihal/Thesis/BasicSR/datasets/Train/Original/LSDIR/LR_Bicubic/train_x4/train'
destination_folder = '/home/nihal/Thesis/BasicSR/datasets/Train/Original/LSDIR/LR_Bicubic/X4'

# Create destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Allowed image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(source_root):
    for file in files:
        if file.lower().endswith(image_extensions):
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)

            # Avoid overwriting existing files
            if os.path.exists(destination_path):
                base, ext = os.path.splitext(file)
                count = 1
                while os.path.exists(destination_path):
                    new_filename = f"{base}_{count}{ext}"
                    destination_path = os.path.join(destination_folder, new_filename)
                    count += 1

            shutil.move(source_path, destination_path)
            print(f"Moved: {source_path} -> {destination_path}")
