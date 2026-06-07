# step2.py
# 功能：
# 1. 使用 PyTorch 訓練 MobileNetV2 與 CNN
# 2. 使用 Focal Loss + Class Weight + Garbage Penalty 處理類別不平衡
# 3. MobileNetV2 與 CNN 使用不同 learning rate
# 4. 評估 Accuracy、Precision、Recall、F1-score、Macro F1、Confusion Matrix
# 5. 儲存模型版本與實驗結果

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

project_root = Path(__file__).resolve().parent.parent
base_path = project_root / "trashnet" / "trashnew"

print("project_root =", project_root)
print("base_path =", base_path)

if not base_path.exists():
    raise FileNotFoundError(f"找不到資料夾：{base_path}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device =", device)


# =========================
# 2. 訓練參數設定
# =========================

EPOCHS = 10
BATCH_SIZE = 64
IMAGE_SIZE = 224
RANDOM_SEED = 42

# MobileNetV2：只訓練最後分類層，learning rate 可以較大
MOBILENET_LR = 0.001

# CNN：從零開始訓練，learning rate 建議較小
CNN_LR = 0.0001

# Focal Loss 參數
# gamma 越大，越重視難分樣本與答錯樣本
FOCAL_GAMMA = 2.0

# 額外提高 garbage 類別的錯誤懲罰
# 如果模型還是完全抓不到 garbage，可以提高到 2.0
# 如果模型過度預測 garbage，可以降低到 1.2
GARBAGE_PENALTY_MULTIPLIER = 1.5

torch.manual_seed(RANDOM_SEED)

RUN_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = project_root / "models" / RUN_TIME
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("MODEL_DIR =", MODEL_DIR)


# =========================
# 3. 建立 transforms
# =========================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor()
])

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

print("\n===== Dataset Info =====")
print("classes =", train_dataset.classes)
print("class_to_idx =", train_dataset.class_to_idx)
print("train size =", len(train_dataset))
print("val size =", len(val_dataset))
print("test size =", len(test_dataset))


# =========================
# 5. 計算 class weight
# =========================

class_counts = [0] * len(train_dataset.classes)

for _, label in train_dataset.samples:
    class_counts[label] += 1

total_samples = sum(class_counts)

# 原始 class weight：
# total_samples / (num_classes * class_count)
#
# 這裡使用平方根版本，避免少數類別權重過大造成訓練震盪
class_weights = [
    (total_samples / (len(class_counts) * count)) ** 0.5
    for count in class_counts
]

GARBAGE_IDX = train_dataset.class_to_idx["garbage"]
NON_GARBAGE_IDX = train_dataset.class_to_idx["non_garbage"]

# 額外提高 garbage 類別懲罰
GARBAGE_WEIGHT = class_weights[GARBAGE_IDX] * GARBAGE_PENALTY_MULTIPLIER
NON_GARBAGE_WEIGHT = class_weights[NON_GARBAGE_IDX]

print("\n===== Class Weight Info =====")
print("class_counts =", class_counts)
print("original_class_weights =", class_weights)
print("GARBAGE_IDX =", GARBAGE_IDX)
print("NON_GARBAGE_IDX =", NON_GARBAGE_IDX)
print("GARBAGE_WEIGHT =", GARBAGE_WEIGHT)
print("NON_GARBAGE_WEIGHT =", NON_GARBAGE_WEIGHT)
print("FOCAL_GAMMA =", FOCAL_GAMMA)
print("GARBAGE_PENALTY_MULTIPLIER =", GARBAGE_PENALTY_MULTIPLIER)


# =========================
# 6. 建立 MobileNetV2
# =========================

def build_mobilenetv2():
    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    # 凍結特徵提取層，只訓練最後分類層
    for param in model.features.parameters():
        param.requires_grad = False

    # 改成二分類輸出
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, 1)
    )

    return model


# =========================
# 7. 建立 CNN
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
# 8. Focal Loss + Class Weight
# =========================

def focal_loss(outputs, labels):
    """
    outputs: model raw logits, shape [batch_size, 1]
    labels : 0 or 1, shape [batch_size, 1]

    功能：
    1. BCEWithLogitsLoss：基本二分類 loss
    2. Focal factor：提高難分與答錯樣本懲罰
    3. Class weight：提高 garbage 類別懲罰
    """

    bce_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs,
        labels,
        reduction="none"
    )

    probs = torch.sigmoid(outputs)

    # p_t 是模型對正確類別的信心
    # label = 1 時，p_t = probs
    # label = 0 時，p_t = 1 - probs
    p_t = torch.where(
        labels == 1,
        probs,
        1 - probs
    )

    # Focal Loss：答得越差，p_t 越小，懲罰越大
    focal_factor = (1 - p_t) ** FOCAL_GAMMA

    # Class Weight：garbage 類別給更大權重
    sample_weights = torch.where(
        labels == GARBAGE_IDX,
        torch.tensor(GARBAGE_WEIGHT, device=device),
        torch.tensor(NON_GARBAGE_WEIGHT, device=device)
    )

    loss = bce_loss * focal_factor * sample_weights

    return loss.mean()


# =========================
# 9. 訓練函式
# =========================

def train_model(model, train_loader, val_loader, epochs, learning_rate, model_name):
    model = model.to(device)

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
            loss = focal_loss(outputs, labels)

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
                loss = focal_loss(outputs, labels)

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
# 10. 評估函式
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

    precision_non_garbage = precision_score(
        all_labels,
        all_preds,
        pos_label=NON_GARBAGE_IDX,
        zero_division=0
    )

    recall_non_garbage = recall_score(
        all_labels,
        all_preds,
        pos_label=NON_GARBAGE_IDX,
        zero_division=0
    )

    f1_non_garbage = f1_score(
        all_labels,
        all_preds,
        pos_label=NON_GARBAGE_IDX,
        zero_division=0
    )

    precision_garbage = precision_score(
        all_labels,
        all_preds,
        pos_label=GARBAGE_IDX,
        zero_division=0
    )

    recall_garbage = recall_score(
        all_labels,
        all_preds,
        pos_label=GARBAGE_IDX,
        zero_division=0
    )

    f1_garbage = f1_score(
        all_labels,
        all_preds,
        pos_label=GARBAGE_IDX,
        zero_division=0
    )

    macro_precision = precision_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n===== {model_name} Test Evaluation =====")
    print(f"Accuracy              : {acc:.4f}")
    print(f"Garbage Precision     : {precision_garbage:.4f}")
    print(f"Garbage Recall        : {recall_garbage:.4f}")
    print(f"Garbage F1-score      : {f1_garbage:.4f}")
    print(f"Non-garbage Precision : {precision_non_garbage:.4f}")
    print(f"Non-garbage Recall    : {recall_non_garbage:.4f}")
    print(f"Non-garbage F1-score  : {f1_non_garbage:.4f}")
    print(f"Macro Precision       : {macro_precision:.4f}")
    print(f"Macro Recall          : {macro_recall:.4f}")
    print(f"Macro F1-score        : {macro_f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes,
        zero_division=0
    )
    print(report)

    metrics = {
        "accuracy": acc,

        "garbage_precision": precision_garbage,
        "garbage_recall": recall_garbage,
        "garbage_f1_score": f1_garbage,

        "non_garbage_precision": precision_non_garbage,
        "non_garbage_recall": recall_non_garbage,
        "non_garbage_f1_score": f1_non_garbage,

        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1_score": macro_f1,

        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }

    return metrics


# =========================
# 11. 畫 Accuracy / Loss
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
    plt.savefig(MODEL_DIR / f"{title}_accuracy.png")

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title(title + " Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(MODEL_DIR / f"{title}_loss.png")


# =========================
# 12. 畫 Confusion Matrix
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
    plt.savefig(MODEL_DIR / f"{title}.png")


# =========================
# 13. 儲存模型版本
# =========================

def save_model_version(model, history, metrics, model_name, learning_rate):
    model_path = MODEL_DIR / f"{model_name}.pth"
    info_path = MODEL_DIR / f"{model_name}_info.json"

    params = {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": learning_rate,
        "image_size": IMAGE_SIZE,
        "random_seed": RANDOM_SEED,
        "device": str(device),
        "class_to_idx": train_dataset.class_to_idx,
        "class_counts": class_counts,
        "original_class_weights": class_weights,
        "garbage_weight": GARBAGE_WEIGHT,
        "non_garbage_weight": NON_GARBAGE_WEIGHT,
        "focal_gamma": FOCAL_GAMMA,
        "garbage_penalty_multiplier": GARBAGE_PENALTY_MULTIPLIER,
        "loss_function": "Focal Loss + Class Weight + Garbage Penalty"
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "metrics": metrics,
            "params": params
        },
        model_path
    )

    info = {
        "model_name": model_name,
        "saved_model": str(model_path),
        "metrics": metrics,
        "params": params
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print(f"\n{model_name} model saved to:", model_path)
    print(f"{model_name} info saved to:", info_path)


# =========================
# 14. 訓練 MobileNetV2
# =========================

mobilenet_model = build_mobilenetv2()

mobilenet_model, mobilenet_history = train_model(
    model=mobilenet_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    learning_rate=MOBILENET_LR,
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
    title="MobileNetV2_confusion_matrix"
)

save_model_version(
    model=mobilenet_model,
    history=mobilenet_history,
    metrics=mobilenet_metrics,
    model_name="MobileNetV2",
    learning_rate=MOBILENET_LR
)


# =========================
# 15. 訓練 CNN
# =========================

cnn_model = SimpleCNN()

cnn_model, cnn_history = train_model(
    model=cnn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=EPOCHS,
    learning_rate=CNN_LR,
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
    title="CNN_confusion_matrix"
)

save_model_version(
    model=cnn_model,
    history=cnn_history,
    metrics=cnn_metrics,
    model_name="CNN",
    learning_rate=CNN_LR
)


# =========================
# 16. 比較結果
# =========================

print("\n===== Model Comparison =====")

print("\nMobileNetV2:")
print(f"Accuracy        : {mobilenet_metrics['accuracy']:.4f}")
print(f"Garbage Recall  : {mobilenet_metrics['garbage_recall']:.4f}")
print(f"Garbage F1      : {mobilenet_metrics['garbage_f1_score']:.4f}")
print(f"Macro F1        : {mobilenet_metrics['macro_f1_score']:.4f}")

print("\nCNN:")
print(f"Accuracy        : {cnn_metrics['accuracy']:.4f}")
print(f"Garbage Recall  : {cnn_metrics['garbage_recall']:.4f}")
print(f"Garbage F1      : {cnn_metrics['garbage_f1_score']:.4f}")
print(f"Macro F1        : {cnn_metrics['macro_f1_score']:.4f}")

plt.show()