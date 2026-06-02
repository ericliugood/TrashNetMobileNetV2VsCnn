# 建立二分類 + 切分 train/val/test 並複製到對應資料夾

import os
import shutil
import random
from pathlib import Path

# 專案根目錄：目前假設 step1.py 在 scripts 資料夾裡
project_root = Path(__file__).resolve().parent.parent

original_path = project_root / "trashnet" / "trash"
base_path = project_root / "trashnet" / "trashnew"

print("original_path =", original_path)
print("base_path =", base_path)

# 檢查原始資料夾是否存在
if not original_path.exists():
    raise FileNotFoundError(f"找不到原始資料夾：{original_path}")

# 建立資料夾
for split in ["train", "val", "test"]:
    for category in ["garbage", "non_garbage"]:
        os.makedirs(base_path / split / category, exist_ok=True)

non_garbage_classes = ["glass", "paper", "cardboard", "plastic", "metal"]

# 收集所有圖片
data = []

for category in os.listdir(original_path):
    category_path = original_path / category

    # 避免讀到不是資料夾的檔案，例如 .DS_Store
    if not category_path.is_dir():
        continue

    for img in os.listdir(category_path):
        img_path = category_path / img

        # 避免複製非圖片檔
        if not img_path.is_file():
            continue

        label = "non_garbage" if category in non_garbage_classes else "garbage"
        data.append((img_path, label))

# 打亂資料
random.seed(42)
random.shuffle(data)

# 切分比例
train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]


def copy_data(dataset, split):
    for src, label in dataset:
        filename = src.name
        dst = base_path / split / label / filename
        shutil.copy(src, dst)


copy_data(train_data, "train")
copy_data(val_data, "val")
copy_data(test_data, "test")

print("資料切分完成")
print("train:", len(train_data))
print("val:", len(val_data))
print("test:", len(test_data))
print("輸出位置:", base_path)