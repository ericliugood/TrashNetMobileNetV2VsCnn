# step1.py
# 功能：
# 1. 將 TrashNet 原始六分類資料轉成二分類
# 2. 切分 train / val / test
# 3. 複製圖片到 trashnet/trashnew 對應資料夾

import os
import shutil
import random
from pathlib import Path


# =========================
# 1. 設定路徑
# =========================

# 假設 step1.py 放在 scripts/ 裡
project_root = Path(__file__).resolve().parent.parent

# 原始資料集位置
original_path = project_root / "trashnet" / "trash"

# 輸出資料集位置
base_path = project_root / "trashnet" / "trashnew"

print("project_root =", project_root)
print("original_path =", original_path)
print("base_path =", base_path)


# =========================
# 2. 檢查原始資料夾
# =========================

if not original_path.exists():
    raise FileNotFoundError(f"找不到原始資料夾：{original_path}")


# =========================
# 3. 可選：清空舊的 trashnew
# =========================

if base_path.exists():
    print("偵測到舊的 trashnew，正在刪除...")
    shutil.rmtree(base_path)


# =========================
# 4. 建立目標資料夾
# =========================

splits = ["train", "val", "test"]
labels = ["garbage", "non_garbage"]

for split in splits:
    for label in labels:
        os.makedirs(base_path / split / label, exist_ok=True)


# =========================
# 5. 定義二分類規則
# =========================

non_garbage_classes = [
    "glass",
    "paper",
    "cardboard",
    "plastic",
    "metal"
]

# trash 類別會被歸為 garbage


# =========================
# 6. 收集所有圖片
# =========================

image_extensions = [".jpg", ".jpeg", ".png"]

data = []

for category in os.listdir(original_path):
    category_path = original_path / category

    if not category_path.is_dir():
        continue

    for img in os.listdir(category_path):
        img_path = category_path / img

        if not img_path.is_file():
            continue

        if img_path.suffix.lower() not in image_extensions:
            continue

        if category in non_garbage_classes:
            label = "non_garbage"
        else:
            label = "garbage"

        data.append((img_path, label))


print("總圖片數量:", len(data))

if len(data) == 0:
    raise ValueError("沒有讀到任何圖片，請檢查 original_path 是否正確。")


# =========================
# 7. 打亂資料並切分
# =========================

random.seed(2150)
random.shuffle(data)

train_size = int(0.7 * len(data))
val_size = int(0.15 * len(data))

train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]


# =========================
# 8. 複製資料
# =========================

def copy_data(dataset, split):
    for src, label in dataset:
        filename = src.name
        dst = base_path / split / label / filename
        shutil.copy(src, dst)


copy_data(train_data, "train")
copy_data(val_data, "val")
copy_data(test_data, "test")


# =========================
# 9. 統計每個 split 的數量
# =========================

def count_files(folder):
    return len([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])


print("\n===== Split Result =====")

for split in splits:
    print(f"\n{split}:")
    for label in labels:
        folder = base_path / split / label
        print(f"{label}: {count_files(folder)}")

print("\n資料整理完成")
print("輸出位置:", base_path)