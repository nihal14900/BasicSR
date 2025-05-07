import torch
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
from pathlib import Path
import tqdm
from concurrent.futures import ProcessPoolExecutor

# ======================
# Configuration
# ======================
IMAGE_FOLDER = r"E:\Codes\BasicSR\datasets\DF2K\HR"  # <-- Change this to your folder path
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
NUM_WORKERS = 8  # Set to number of CPU cores you want to use

# ======================
# Function to process a single image
# ======================
def process_single_image(image_path):
    try:
        img = read_image(str(image_path), mode=ImageReadMode.RGB)  # [3, H, W]
    except Exception as e:
        print(f"Skipping file {image_path}: {e}")
        return None

    c, h, w = img.shape
    num_pixels = h * w

    # Normalize to [0, 1]
    img = img.to(dtype=torch.float64) / 255.0

    # Sum pixel values per channel
    img_sum = img.view(c, -1).sum(dim=1)  # shape: [3]

    return img_sum, num_pixels

# ======================
# Main function to compute RGB mean
# ======================
def compute_rgb_mean_normalized_mp(image_folder, extensions, num_workers):
    image_paths = []
    for ext in extensions:
        image_paths += list(Path(image_folder).glob(f'*{ext.lower()}'))

    image_paths = list(set(image_paths))  # Remove duplicates

    if not image_paths:
        raise ValueError(f"No images found in '{image_folder}' with extensions {extensions}")

    print(f"Found {len(image_paths)} images. Processing with {num_workers} workers...")

    total = torch.zeros(3, dtype=torch.float64)
    num_pixels_total = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm.tqdm(executor.map(process_single_image, image_paths),
                                total=len(image_paths),
                                desc="Processing Images"))

    for result in results:
        if result is None:
            continue
        img_sum, num_pixels = result
        total += img_sum
        num_pixels_total += num_pixels

    mean = total / num_pixels_total
    return mean  # Tensor of shape [3], order: R, G, B

# ======================
# Main Execution
# ======================
if __name__ == "__main__":
    rgb_mean = compute_rgb_mean_normalized_mp(IMAGE_FOLDER, IMAGE_EXTENSIONS, NUM_WORKERS)
    print("Computed RGB Mean (normalized to [0, 1]):")
    print(f"Red: {rgb_mean[0].item():.6f}, Green: {rgb_mean[1].item():.6f}, Blue: {rgb_mean[2].item():.6f}")

# DF2K: [0.468898, 0.448961, 0.403436]