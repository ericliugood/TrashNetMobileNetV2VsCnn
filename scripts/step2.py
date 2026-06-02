# step2.py
# 功能：
# 1. 使用 PyTorch 讀取 train / val / test 圖片資料
# 2. 訓練 MobileNetV2 與自建 CNN
# 3. 比較 Accuracy / Precision / Recall / F1-score / Confusion Matrix
# 4. 儲存模型版本與實驗結果

from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================
# 1. 設定路徑與裝置
# =========================

# 假設 step2.py 放在 scripts/ 裡面
project_root = Path(__file__).resolve().parent.parent

base_path = project_root / "trashnet" / "trashnew"

print("project_root =", project_root)
print("base_path =", base_path)

if not base_path.exists():
    raise FileNotFoundError(f"找不到資料夾：{base_path}")

# Mac M 系列使用 mps，其他使用 cpu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device =", device)


# =========================
# 2. 訓練參數設定
# =========================

EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMAGE_SIZE = 224

# 每次執行建立一個模型版本資料夾
RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = project_root / "models" / RUN_TIME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("MODEL_DIR =", MODEL_DIR)


# =========================
# 3. 建立 transforms
# =========================

# train：有資料增強
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# val / test：不做資料增強
test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# =========================
# 4. 建立 Dataset 與 DataLoader
# =========================

train_dataset = datasets.ImageFolder(
    root=base_path / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=base_path / "val",
    transform=test_transform
)

test_dataset = datasets.ImageFolder(
    root=base_path / "test",
    transform=test_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("classes =", train_dataset.classes)
print("class_to_idx =", train_dataset.class_to_idx)
print("train size =", len(train_dataset))
print("val size =", len(val_dataset))
print("test size =", len(test_dataset))


# =========================
# 5. 建立 MobileNetV2 模型
# =========================

def build_mobilenetv2():
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    # 凍結特徵提取層
    for param in model.features.parameters():
        param.requires_grad = False

    # 修改分類層為二分類
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 1)
    )

    return model


# =========================
# 6. 建立自建 CNN 模型
# =========================

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 26 * 26, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 7. 訓練函式
# =========================

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_name):
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    history = {
        "train_acc": [],
        "val_acc": [],
        "train_loss": [],
        "val_loss": []
    }

    for epoch in range(epochs):
        # ---------- Training ----------
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        train_epoch_loss = train_loss / train_total
        train_epoch_acc = train_correct / train_total

        # ---------- Validation ----------
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().unsqueeze(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()

                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        history["train_loss"].append(train_epoch_loss)
        history["val_loss"].append(val_epoch_loss)
        history["train_acc"].append(train_epoch_acc)
        history["val_acc"].append(val_epoch_acc)

        print(
            f"{model_name} | "
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_epoch_loss:.4f}, "
            f"Train Acc: {train_epoch_acc:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f}, "
            f"Val Acc: {val_epoch_acc:.4f}"
        )

    return model, history


# =========================
# 8. 評估函式
# =========================

def evaluate_model(model, test_loader, model_name):
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long().cpu().numpy().flatten()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n===== {model_name} Test Evaluation =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=test_dataset.classes,
            zero_division=0
        )
    )

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

    return metrics


# =========================
# 9. 畫 Accuracy / Loss 圖
# =========================

def plot_history(history, title):
    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.title(title + " Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(title + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()


# =========================
# 10. 畫 Confusion Matrix
# =========================

def plot_confusion_matrix(cm, class_names, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j, i, cm[i][j], ha="center", va="center")

    plt.colorbar()


# =========================
# 11. 儲存模型版本
# =========================

def save_model_version(model, history, metrics, model_name):
    model_path = MODEL_DIR / f"{model_name}.pth"
    info_path = MODEL_DIR / f"{model_name}_info.json"

    save_data = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "metrics": metrics,
        "class_to_idx": train_dataset.class_to_idx,
        "params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "device": str(device)
        }
    }

    torch.save(save_data, model_path)

    info = {
        "model_name": model_name,
        "saved_model": str(model_path),
        "metrics": metrics,
        "class_to_idx": train_dataset.class_to_idx,
        "params": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "image_size": IMAGE_SIZE,
            "device": str(device)
        }
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print(f"\n{model_name} model saved to:", model_path)
    print(f"{model_name} info saved to:", info_path)


# =========================
# 12. 訓練 MobileNetV2
# =========================

mobilenet_model = build_mobilenetv2()

mobilenet_model, mobilenet_history = train_model(
    model=mobilenet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    model_name="MobileNetV2"
)

plot_history(mobilenet_history, "MobileNetV2")

mobilenet_metrics = evaluate_model(
    model=mobilenet_model,
    test_loader=test_loader,
    model_name="MobileNetV2"
)

plot_confusion_matrix(
    cm=mobilenet_metrics["confusion_matrix"],
    class_names=test_dataset.classes,
    title="MobileNetV2 Confusion Matrix"
)

save_model_version(
    model=mobilenet_model,
    history=mobilenet_history,
    metrics=mobilenet_metrics,
    model_name="MobileNetV2"
)


# =========================
# 13. 訓練 CNN
# =========================

cnn_model = SimpleCNN()

cnn_model, cnn_history = train_model(
    model=cnn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    model_name="CNN"
)

plot_history(cnn_history, "CNN")

cnn_metrics = evaluate_model(
    model=cnn_model,
    test_loader=test_loader,
    model_name="CNN"
)

plot_confusion_matrix(
    cm=cnn_metrics["confusion_matrix"],
    class_names=test_dataset.classes,
    title="CNN Confusion Matrix"
)

save_model_version(
    model=cnn_model,
    history=cnn_history,
    metrics=cnn_metrics,
    model_name="CNN"
)


# =========================
# 14. 比較結果
# =========================

print("\n===== Model Comparison =====")

print("\nMobileNetV2:")
print(f"Accuracy : {mobilenet_metrics['accuracy']:.4f}")
print(f"Precision: {mobilenet_metrics['precision']:.4f}")
print(f"Recall   : {mobilenet_metrics['recall']:.4f}")
print(f"F1-score : {mobilenet_metrics['f1_score']:.4f}")

print("\nCNN:")
print(f"Accuracy : {cnn_metrics['accuracy']:.4f}")
print(f"Precision: {cnn_metrics['precision']:.4f}")
print(f"Recall   : {cnn_metrics['recall']:.4f}")
print(f"F1-score : {cnn_metrics['f1_score']:.4f}")

plt.show()