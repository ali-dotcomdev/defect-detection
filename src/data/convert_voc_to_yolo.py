"""
NEU-DET Veri Seti için Pascal VOC (XML) -> YOLO (TXT) Format Dönüştürücü.
Görselleri ve dönüştürülen etiketleri 'data/processed/yolo/' altına organize eder.
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# Sınıf Sıralaması (CNN ve RF ile birebir aynı indeksler)
CLASSES = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}


def convert_bbox(size, box):
    """VOC (xmin, ymin, xmax, ymax) koordinatlarını YOLO (x_center, y_center, w, h) formatına çevirir."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) * dw
    y_center = ((ymin + ymax) / 2.0) * dh
    w = (xmax - xmin) * dw
    h = (ymax - ymin) * dh
    
    # [0.0, 1.0] sınırlarında tut
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    
    return x_center, y_center, w, h


def process_split(raw_dir: Path, out_dir: Path, split: str):
    """İlgili split'in (train/validation) XML ve görsellerini işler."""
    xml_dir = raw_dir / split / "annotations"
    img_dir = raw_dir / split / "images"

    # Hedef split adı: validation klasörünü YOLO için 'val' yapıyoruz
    target_split = 'val' if split.startswith('val') else 'train'
    
    out_img_dir = out_dir / "images" / target_split
    out_lbl_dir = out_dir / "labels" / target_split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    xml_files = list(xml_dir.glob("*.xml"))
    print(f"İşleniyor: {split} -> {len(xml_files)} XML dosyası")

    for xml_file in tqdm(xml_files, desc=f"{split} dönüşümü"):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        size_elem = root.find('size')
        width = float(size_elem.find('width').text)
        height = float(size_elem.find('height').text)

        yolo_lines = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in CLASS_TO_IDX:
                continue

            cls_id = CLASS_TO_IDX[cls_name]
            xml_box = obj.find('bndbox')
            box = (
                float(xml_box.find('xmin').text),
                float(xml_box.find('ymin').text),
                float(xml_box.find('xmax').text),
                float(xml_box.find('ymax').text)
            )

            xc, yc, w, h = convert_bbox((width, height), box)
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        # TXT Etiketini Kaydet
        txt_filename = xml_file.stem + ".txt"
        with open(out_lbl_dir / txt_filename, "w", encoding="utf-8") as f:
            f.writelines(yolo_lines)

        # Karşılık gelen görseli hedef klasöre kopyala (jpg veya bmp)
        img_name = xml_file.stem + ".jpg"
        src_img = img_dir / img_name
        if not src_img.exists():
            src_img = img_dir / (xml_file.stem + ".bmp")

        if src_img.exists():
            shutil.copy(src_img, out_img_dir / src_img.name)


def main():
    root = Path.cwd()
    # Eğer src altından çalıştırılırsa bir üste çık
    if root.name == "src" or root.name == "data":
        root = root.parent

    raw_dir = root / "data/raw/NEU-DET"
    out_dir = root / "data/processed/yolo"

    print(f"Pascal VOC -> YOLO Dönüşümü Başlatılıyor...")
    print(f"Kaynak: {raw_dir}")
    print(f"Hedef : {out_dir}\n")

    process_split(raw_dir, out_dir, split="train")
    process_split(raw_dir, out_dir, split="validation")

    print("\nDönüşüm başarıyla tamamlandı!")
    print(f"YOLO veri seti hazır -> {out_dir}")


if __name__ == "__main__":
    main()