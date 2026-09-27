from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.glcm import extract_single_image_glcm


def test_glcm_feature_vector_dimension(tmp_path):
    """
    Feature Boyut ve Tip Testi:
    200x200 boyutundaki test görselinin, feature çıkarıcı tarafından
    tam olarak 32 anahtara sahip bir sözlüğe (dict) dönüştürüldüğünü doğrular.
    """
    # 1. 200x200 rastgele gri seviye [0, 255] test matrisi üret
    dummy_image = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

    # 2. Geçici sahte görseli kaydet
    temp_img_path = tmp_path / "dummy_steel.jpg"
    cv2.imwrite(str(temp_img_path), dummy_image)

    # 3. Fonksiyonu çalıştır
    features = extract_single_image_glcm(str(temp_img_path))

    # 4. Doğrulamalar
    assert features is not None, "Feature çıkarımı başarısız oldu (None döndü)."
    assert isinstance(features, dict), f"Beklenen çıktı tipi dict, alınan: {type(features)}"
    assert len(features) == 32, f"Feature sözlüğü 32 anahtar içermeli, alınan: {len(features)}"
    
    # Tüm değerlerin sayısal (float/int) olduğunu kontrol et
    for key, val in features.items():
        assert isinstance(val, (int, float, np.floating)), f"{key} sayısal bir değer değil: {type(val)}"