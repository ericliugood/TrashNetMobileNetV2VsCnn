# test_image.py
# 功能：
# 直接在程式中指定模型路徑與圖片路徑，測試單張圖片分類結果

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# =========================
# 1. 直接指定路徑
# =========================

project_root = Path(__file__).resolve().parent.parent

# 改成你自己的模型資料夾時間戳
MODEL_PATH = project_root / "models" / "20260607_115102" / "MobileNetV2.pth"

# 可選：mobilenetv2 或 cnn
MODEL_TYPE = "mobilenetv2"

# 放你要測試的圖片
IMAGE_PATH = project_root / "test_images" / "img_1.png"

IMAGE_SIZE = 224


# =========================
# 2. 建立 MobileNetV2
# =========================

def build_mobilenetv2():
    model = models.mobilenet_v2(weights=None)

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 1)
    )

    return model


# =========================
# 3. 建立 CNN
# 要和 step2.py 的 SimpleCNN 完全一致
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 4. 載入模型
# =========================

def load_model(model_path, model_type, device):
    checkpoint = torch.load(model_path, map_location=device)

    if model_type == "mobilenetv2":
        model = build_mobilenetv2()
    elif model_type == "cnn":
        model = SimpleCNN()
    else:
        raise ValueError("MODEL_TYPE 只能是 mobilenetv2 或 cnn")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    class_to_idx = checkpoint["params"]["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    return model, class_to_idx, idx_to_class


# =========================
# 5. 圖片前處理
# =========================

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    return image


# =========================
# 6. 預測圖片
# =========================

def predict_image(model, image_tensor, class_to_idx, idx_to_class, device):
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)

        # output 是 logit，sigmoid 後代表 index 1 的機率
        prob_class_1 = torch.sigmoid(output).item()

        pred_idx = 1 if prob_class_1 >= 0.5 else 0
        pred_label = idx_to_class[pred_idx]

    garbage_idx = class_to_idx["garbage"]
    non_garbage_idx = class_to_idx["non_garbage"]

    if garbage_idx == 0:
        garbage_prob = 1 - prob_class_1
        non_garbage_prob = prob_class_1
    else:
        garbage_prob = prob_class_1
        non_garbage_prob = 1 - prob_class_1

    return pred_label, garbage_prob, non_garbage_prob


# =========================
# 7. 主程式
# =========================

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("device =", device)
    print("MODEL_PATH =", MODEL_PATH)
    print("MODEL_TYPE =", MODEL_TYPE)
    print("IMAGE_PATH =", IMAGE_PATH)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到模型檔案：{MODEL_PATH}")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到圖片檔案：{IMAGE_PATH}")

    model, class_to_idx, idx_to_class = load_model(
        model_path=MODEL_PATH,
        model_type=MODEL_TYPE,
        device=device
    )

    print("class_to_idx =", class_to_idx)

    image_tensor = preprocess_image(IMAGE_PATH)

    pred_label, garbage_prob, non_garbage_prob = predict_image(
        model=model,
        image_tensor=image_tensor,
        class_to_idx=class_to_idx,
        idx_to_class=idx_to_class,
        device=device
    )

    print("\n===== Prediction Result =====")
    print("Predicted Label:", pred_label)
    print(f"Garbage Probability    : {garbage_prob:.4f}")
    print(f"Non-garbage Probability: {non_garbage_prob:.4f}")


if __name__ == "__main__":
    main()