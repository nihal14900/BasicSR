import os

# Set the target folder where images are located
target_folder = '/home/nihal/Thesis/BasicSR/datasets/Train/Original/LSDIR/LR_Bicubic/X2'

# Set this to True if you want to process subfolders as well
include_subfolders = True

# Allowed image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Choose how to walk the directory
walker = os.walk(target_folder) if include_subfolders else [(target_folder, [], os.listdir(target_folder))]

for root, dirs, files in walker:
    for filename in files:
        if filename.lower().endswith(image_extensions) and '_1' in filename:
            new_filename = filename.replace('_1', '', 1)
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_filename)

            # Avoid overwriting
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
            else:
                print(f"Skipped (exists): {new_filename}")
