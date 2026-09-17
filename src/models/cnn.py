"""
Custom CNN Mimarisi ve Eğitim / Değerlendirme Modülü.
"""

from pathlib import Path
from typing import Dict, Tuple
import copy
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


class SteelDefectCNN(nn.Module):
    """4 konvolüsyon bloğu ve Global Average Pooling içeren hafif CNN mimarisi."""

    def __init__(self, num_classes: int = 6):
        super(SteelDefectCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_cnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 15,
    lr: float = 1e-3,
    save_path: str = "models/cnn_baseline.pth"
) -> Tuple[nn.Module, Dict]:
    """CNN modelini eğitir, en iyi doğrulama başarısına sahip ağırlıkları kaydeder."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)

    print(f"model egiti baslatildi: ({device}) - Toplam Epoch: {epochs}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # Eğitim
        model.train()
        t_loss, t_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            t_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            t_correct += torch.sum(preds == labels).item()

        epoch_t_loss = t_loss / train_size
        epoch_t_acc = t_correct / train_size

        # Doğrulama
        model.eval()
        v_loss, v_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)

                v_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += torch.sum(preds == labels).item()

        epoch_v_loss = v_loss / val_size
        epoch_v_acc = v_correct / val_size
        elapsed = time.time() - t0

        history['train_loss'].append(epoch_t_loss)
        history['train_acc'].append(epoch_t_acc)
        history['val_loss'].append(epoch_v_loss)
        history['val_acc'].append(epoch_v_acc)

        if epoch_v_acc > best_val_acc:
            best_val_acc = epoch_v_acc
            best_weights = copy.deepcopy(model.state_dict())
            flag = "En İyi"
        else:
            flag = ""

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.2f}s) | "
              f"Train Loss: {epoch_t_loss:.4f} Acc: %{epoch_t_acc*100:.2f} | "
              f"Val Loss: {epoch_v_loss:.4f} Acc: %{epoch_v_acc*100:.2f} {flag}")

    # En iyi ağırlıkları geri yükle ve diske yaz
    model.load_state_dict(best_weights)
    out_file = Path(save_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, out_file)
    print(f"\nEğitim Tamamlandı! En İyi Val Acc: %{best_val_acc*100:.2f}")
    print(f"Model Kaydedildi -> {out_file}")

    return model, history


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    class_names: list
) -> Dict:
    """Doğrulama kümesi üzerinde tahmin toplar, metrik raporu ve hata matrisi üretir."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'labels': all_labels
    }