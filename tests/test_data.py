from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]

def test_yolo_labels_normalized_and_valid():
    """YOLO etiketlerinin varlığını ve [0.0, 1.0] aralığında olduğunu doğrular."""
    labels_dir = ROOT_DIR / "data" / "processed" / "yolo" / "labels"
    label_files = list(labels_dir.rglob("*.txt"))
    
    assert len(label_files) > 0, f"Etiket klasörü boş veya bulunamadı: {labels_dir}"
    
    total_boxes = 0
    for label_file in label_files:
        with open(label_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                
                assert len(parts) == 5, f"{label_file.name} Satır {line_num}: 5 sütun bekleniyordu."
                
                cls_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                assert 0 <= cls_id <= 5, f"Geçersiz Sınıf: {cls_id} ({label_file.name})"
                
                for val in coords:
                    assert 0.0 <= val <= 1.0, f"Aşım tespit edildi: {val} ({label_file.name})"
                
                total_boxes += 1
                
    print(f"\n[INFO] {len(label_files)} dosya ve {total_boxes} kusur kutusu doğrulandı.")