import os
import numpy as np
from PIL import Image

# =====================配置区=====================
ROOT_DIR = "demo_synthetic_dataset"
SPLITS = ["train", "val", "test"]
CLASS_NAMES = ["0_原生", "1_碎裂", "2_碎粒", "3_糜棱"]
# 每个划分内每一类生成图片数量
NUM_PER_CLASS = {
    "train": 8,
    "val": 4,
    "test": 4
}
IMAGE_SIZE = 224
# ================================================

def generate_random_gray_image(size=224):
    # 生成随机灰度模拟图像
    arr = np.random.randint(low=30, high=220, size=(size, size), dtype=np.uint8)
    # 增加一点简单纹理区分各类（可选）
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
    print(f"✅ 模拟数据集生成完成！根目录：{ROOT_DIR}")
    print("目录结构与真实数据集完全对齐，可以直接用于代码测试")