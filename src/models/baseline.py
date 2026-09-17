from pathlib import Path
from typing import Dict, Tuple
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def prepare_train_val_data(features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    ignore_cols = ['file_path', 'defect_class', 'split']
    feature_cols = [c for c in features_df.columns if c not in ignore_cols]

    train_mask = features_df['split'] == 'train'
    val_mask = features_df['split'] == 'validation'

    X_train = features_df.loc[train_mask, feature_cols]
    y_train = features_df.loc[train_mask, 'defect_class']

    X_val = features_df.loc[val_mask, feature_cols]
    y_val = features_df.loc[val_mask, 'defect_class']

    return X_train, y_train, X_val, y_val


def train_and_evaluate_rf(
    features_df: pd.DataFrame,
    model_save_path: str = "models/rf_baseline.joblib",
    n_estimators: int = 150,
    max_depth: int = 12
) -> Dict:
    X_train, y_train, X_val, y_val = prepare_train_val_data(features_df)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    # Modeli Diske Kaydet
    out_path = Path(model_save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, out_path)
    print(f"Model başarıyla kaydedildi -> {out_path}")

    return {
        'model': rf,
        'accuracy': acc,
        'classification_report': report,
        'confusion_matrix': cm
    }