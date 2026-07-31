import os
import numpy as np
from PIL import Image

# ===================== Configuration =====================
ROOT_DIR = "demo_synthetic_dataset"
SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["0_primary", "1_cataclastic", "2_granulated", "3_mylonitic"]
# Number of images generated for each class in each subset
NUM_PER_CLASS = {
    "train": 8,
    "val": 4,
    "test": 4
}
IMAGE_SIZE = 224
# =========================================================

def generate_random_gray_image(size=224):
    # Generate synthetic grayscale images to simulate ERMI logging images
    arr = np.random.randint(low=30, high=220, size=(size, size), dtype=np.uint8)
    # Add random noise to simulate texture features of electrical imaging logs
    noise = np.random.normal(0, 12, arr.shape).astype(np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


if __name__ == "__main__":
    for split in SPLITS:
        for cls in CLASS_NAMES:
            folder_path = os.path.join(ROOT_DIR, split, cls)
            os.makedirs(folder_path, exist_ok=True)
            count = NUM_PER_CLASS[split]
            for idx in range(count):
                img = generate_random_gray_image(IMAGE_SIZE)
                save_name = f"demo_{idx:03d}.png"
                save_path = os.path.join(folder_path, save_name)
                img.save(save_path)
    print(f"✅ Synthetic demo dataset generated successfully! Root directory: {ROOT_DIR}")
    print("The directory structure is consistent with the data loading path of the training script.")
