from pathlib import Path
from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


def extract_single_image_glcm(
    image_path: str,
    distances: Optional[List[int]] = None,
    angles: Optional[List[float]] = None
) -> Optional[Dict[str, float]]:

    if distances is None:
        distances = [1, 3, 5]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    features: Dict[str, float] = {
        'pixel_mean': float(np.mean(img)),
        'pixel_std': float(np.std(img))
    }

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']

    glcm = graycomatrix(
        img,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )

    for prop in properties:
        prop_matrix = graycoprops(glcm, prop)
        for d_idx, dist in enumerate(distances):
            angle_values = prop_matrix[d_idx, :]
            features[f'{prop}_mean_d{dist}'] = float(np.mean(angle_values))
            features[f'{prop}_range_d{dist}'] = float(np.ptp(angle_values))

    return features


def build_feature_dataset(metadata_df: pd.DataFrame, save_path: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="GLCM Çıkarımı"):
        feats = extract_single_image_glcm(row['file_path'])
        if feats is not None:
            feats['file_path'] = row['file_path']
            feats['defect_class'] = row['defect_class']
            feats['split'] = row['split']
            rows.append(feats)

    features_df = pd.DataFrame(rows)

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if out_file.suffix == '.parquet':
            features_df.to_parquet(out_file, index=False)
        else:
            features_df.to_csv(out_file, index=False)
        print(f"Öznitelikler kaydedildi -> {out_file}")

    return features_df