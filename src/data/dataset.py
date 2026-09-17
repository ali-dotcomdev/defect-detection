"""
Çelik yüzey kusurları için PyTorch Dataset ve DataLoader modülü.
"""

from pathlib import Path
from typing import Tuple, Dict
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def get_class_mappings(df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Sınıf isimleri ve sayısal indeksler arasında iki yönlü eşleme sözlüğü üretir."""
    class_names = sorted(df['defect_class'].unique())
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    idx_to_class = {idx: cls_name for cls_name, idx in class_to_idx.items()}
    return class_to_idx, idx_to_class


class SteelDefectDataset(Dataset):
    """Çelik kusur görsellerini diskten okuyup tensöre çeviren PyTorch Dataset sınıfı."""

    def __init__(self, metadata_df: pd.DataFrame, split: str = 'train', class_to_idx: Dict[str, int] = None):
        if split == 'train':
            self.data = metadata_df[metadata_df['split'].str.lower() == 'train'].reset_index(drop=True)
        else:
            self.data = metadata_df[metadata_df['split'].str.lower().isin(['val', 'validation', 'test'])].reset_index(drop=True)

        self.class_to_idx = class_to_idx

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.data.iloc[idx]
        img = cv2.imread(row['file_path'], cv2.IMREAD_GRAYSCALE)

        # [0, 255] uint8 -> [0.0, 1.0] float tensör (1, 200, 200)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        # Standartlaştırma (mean: 0.5, std: 0.2)
        img_tensor = (img_tensor - 0.5) / 0.2

        label_idx = self.class_to_idx[row['defect_class']]
        return img_tensor, torch.tensor(label_idx, dtype=torch.long)


def build_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Train ve Val DataLoader nesnelerini oluşturur."""
    class_to_idx, idx_to_class = get_class_mappings(df)

    train_dataset = SteelDefectDataset(df, split='train', class_to_idx=class_to_idx)
    val_dataset = SteelDefectDataset(df, split='val', class_to_idx=class_to_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, class_to_idx, idx_to_class