# Endüstriyel Çelik Yüzey Kusurlarının Otomatik Tespiti ve Konumlandırılması

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5-ee4c2c.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://ultralytics.com)
[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Bu proje, sıcak haddelenmiş çelik yüzeylerdeki mikro ve makro kusurları tespit etmek, sınıflandırmak ve koordinat bazlı (bounding box) konumlandırmak amacıyla geliştirilmiş uçtan uca bir bilgisayarlı görü (computer vision) projesidir. Çalışmada endüstriyel kıyaslama standardı olan **NEU-DET** veri seti temel alınmıştır.

Proje süreci, geleneksel istatistiksel doku analizinden derin öğrenme sınıflandırmasına ve nihayetinde endüstriyel kalite kontrolde gerçek zamanlı sınır kutusu (bounding box) tespiti sağlayan YOLOv8 mimarisine uzanan **3 aşamalı bir mühendislik kıyaslaması (benchmarking)** üzerine kurulmuştur.

---

## Problem Tanımı ve Mühendislik Evrimi

Çelik üretim hatlarında yüzey kusurlarının alt saniyede tespiti, hatalı malzemenin sonraki aşamalara veya müşteriye ulaşmasını önlemek açısından kritik önem taşır. Bu depoda geliştirilen çözümün aşamaları şu şekildedir:

```text
[ NEU-DET Ham Veri Seti ]
        │
        ├── 1. Geleneksel Makine Öğrenmesi: GLCM Doku Analizi + Random Forest
        │      └── Yorumlanabilirliği yüksek, fakat farklı ölçeklerdeki (birbirine çok yakın değerlerde ortalama, standart sapma) kusurlara genelleme kabiliyeti sınırlı, 1st Order. 
        │
        ├── 2. Derin Öğrenme Tabanlı Sınıflandırma: Özel 4 Bloklu CNN (PyTorch)
        │      └── Başarılı görsel seviyesinde sınıflandırma; ancak kusurun yüzeydeki koordinatını veremiyor.
        │
        └── 3. Endüstriyel Nihai Çözüm: YOLOv8 Nesne Tespiti (Object Detection)
               └── Kusurun türünü ve hassas piksel konumunu eş zamanlı tespit eden, JSON çıktısı üretebilen hat.
```

---

## Temel Özellikler

* **İstatistiksel Doku Analizi:** Çok yönlü ve çok mesafeli Gri Seviye Eş-Oluşum Matrisi (`GLCM`) ile feature çıkarımı(extraction) ve Apache Parquet formatında yüksek performanslı depolama.
* **Modüler Veri Hattı:** Pascal VOC (XML) etiketlerini normalize YOLO (TXT) formatına dönüştüren, PyTorch Dataset ve DataLoader sınıflarını izole eden modüler mimari.
* **Endüstriyel Entegrasyona Uygun Çıktı (JSON Payload):** SCADA, PLC veya fabrika üretim yürütme sistemlerine (MES) doğrudan aktarılabilecek yapılandırılmış JSON denetim raporu üretimi.
* **Detaylı Başarım Değerlendirmesi:** Sınıf bazlı $mAP@50$ skorları, normalize edilmiş hata matrisi (confusion matrix) ve zayıf sınıflar için arıza modu (failure mode) analizi.

---

## 📂 Proje Dizin Yapısı

```text
steel_defect_detection/
├── configs/
│   └── steel_yolo.yaml             # YOLO veri seti yolları ve 6 kusur sınıfının tanımları
├── data/
│   ├── processed/
│   │   ├── glcm_features.parquet   # Çıkarılan GLCM doku feature tablosu
│   │   └── yolo/                   # Pascal VOC'tan dönüştürülmüş YOLO formatlı veri
│   │       ├── images/ (train, val)
│   │       └── labels/ (train, val)
│   └── raw/
│       └── NEU-DET/                # Ham görseller ve Pascal VOC formatındaki XML etiketler
├── models/
│   ├── rf_baseline.joblib          # Eğitilmiş Random Forest model ağırlığı
│   ├── cnn_baseline.pth            # Eğitilmiş PyTorch CNN ağırlığı
│   └── yolo_runs/
│       └── baseline_yolov8n/       # Eğitim eğrileri, ağırlıklar (best.pt, last.pt), metrikler
├── notebooks/
│   ├── 01_data_audit.ipynb         # Keşifçi veri analizi (EDA) ve sınıf dağılım kontrolü
│   ├── 02_deep_learning_baseline.ipynb # Özel CNN mimarisinin eğitimi ve değerlendirmesi
│   └── 03_yolo_detection.ipynb     # YOLOv8 eğitimi, çıkarım testleri ve görselleştirme
└── src/
    ├── data/
    │   ├── convert_voc_to_yolo.py  # Pascal VOC XML -> YOLO TXT koordinat dönüştürücü
    │   └── dataset.py              # PyTorch Dataset ve DataLoader modülü
    ├── features/
    │   └── glcm.py                 # GLCM doku feature çıkarım modülü
    └── models/
        ├── baseline.py             # Random Forest eğitim ve değerlendirme hattı
        └── cnn.py                  # 4 bloklu özel PyTorch CNN sınıflandırıcı mimarisi
```

---

## Mimari ve Metodoloji

### 1. İstatistiksel Doku Analizi (GLCM)
* Çelik yüzeyindeki dokusal yönelimleri modellemek için $d \in \{1, 3, 5\}$ piksel mesafelerinde ve 4 farklı açıda ($0^\circ, 45^\circ, 90^\circ, 135^\circ$) eş-oluşum matrisleri hesaplanmıştır.
* Çıkarılan özellikler(features): **Kontrast (Contrast)**, **Farklılık (Dissimilarity)**, **Homojenlik (Homogeneity)**, **Enerji (Energy)** ve **Korelasyon (Correlation)**. 
Ek olarak birinci dereceden istatistiksel momentler (Piksel Ortalaması ve Standart Sapması) eklenmiştir.

### 2. Özel 4 Bloklu CNN Mimarisi
* **Girdi Boyutu:** $200 \times 200$ tek kanallı gri seviye (grayscale) görseller ($\mu = 0.5, \sigma = 0.2$ normalizasyonu ile).
* **Feature Çıkarıcı:** 4 ardışık konvolüsyon bloğu: `[Conv2D (3x3) -> BatchNorm -> ReLU -> MaxPool (2x2)]` ile kanal sayısı $1 \rightarrow 16 \rightarrow 32 \rightarrow 64 \rightarrow 128$ şeklinde artırılmıştır.
* **Sınıflandırma Başlığı:** Global Average Pooling (`AdaptiveAvgPool2d(1, 1)`), Dropout ($p = 0.3$) ve 2 katmanlı MLP.
* *Kısıt:* Standart CNN sınıflandırması görüntüde kusur olduğunu doğrulamış; ancak robotik kesme ya da kaynak kafalarını yönlendirecek piksel koordinatlarını sağlayamamıştır.

### 3. Endüstriyel Nesne Tespiti (YOLOv8n)
* Eş zamanlı sınıflandırma ve koordinat tespiti (bounding box) için Ultralytics YOLOv8 nano (`yolov8n.pt`) mimarisine geçilmiştir.
* Pascal VOC formatındaki $[x_{min}, y_{min}, x_{max}, y_{max}]$ mutlak koordinatları, normalize edilmiş YOLO $[x_{merkez}, y_{merkez}, w, h]$ formatına otomatik dönüştürülmüştür.

---

## 📊 Deneysel Sonuçlar ve Metrikler

Modeller, NEU-DET veri setinde yer alan 6 endüstriyel kusur sınıfı üzerinde doğrulanmıştır:

| Kusur Sınıfı | Tanım | YOLOv8n $mAP@50$ | Başarım Özeti ve Gözlemler |
|---|---|:---:|---|
| **Scratches** | Çizikler | **%93.0** | Belirgin doğrusal yönelim; arka plandan yüksek doğrulukla ayrım. |
| **Patches** | Plaklar | **%92.0** | Yüksek yüzey kontrastı ve net sınır geçişleri. |
| **Inclusion** | Katışkı / Kalıntı | **%85.0** | Yüksek yoğunluklu noktasal kusurlarda başarılı tespit. |
| **Pitted Surface** | Çukurlaşmış Yüzey | **%78.0** | İnce gözenekli yapı; iyi düzeyde başarım. |
| **Rolled-in Scale** | Haddelenmiş Tufal | **%64.0** | Çelik yüzeyiyle değişken kontrast nedeniyle orta düzey başarım. |
| **Crazing** | Çatlak (Kılcal Ağ) | **%35.0** | Mikro yarıkların temel sac yüzeyiyle karışması (Birincil iyileştirme odağı). |

### Genel Model Başarımı (YOLOv8n Baseline - 25 Epoch)
* **Genel $mAP@50$:** **%74.8**
* **Genel $mAP@50-95$:** **%39.2**
* **Kesinlik (Precision):** **~%68.4**
* **Duyarlılık (Recall):** **~%71.2**

---

## 🏭 Endüstriyel Muayene Çıktı Formatı (JSON Entegrasyonu)

Modelin fabrika otomasyon sistemlerine entegrasyonunu sağlamak amacıyla çıkarım (inference) sonuçları standart JSON formatında dışa aktarılmaktadır:

```json
{
  "image_name": "scratches_241.jpg",
  "image_width": 200,
  "image_height": 200,
  "total_defects_found": 1,
  "defects": [
    {
      "class_id": 5,
      "class_name": "scratches",
      "confidence": 0.8921,
      "bbox": [18.25, 42.10, 182.40, 76.50]
    }
  ]
}
```

---

## Kurulum ve Ortam Hazırlığı

1. **Depoyu klonlayın:**
   ```bash
   git clone https://github.com/kullanici-adiniz/defect-detection.git
   cd defect-detection
   ```

2. **Python sanal ortamı oluşturun ve aktif edin:**
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Gerekli paketleri yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Kullanım

### 1. Etiket Formatı Dönüştürme (Pascal VOC -> YOLO)
Ham XML formatındaki etiketleri YOLO formatına çevirmek için:
```bash
python src/data/convert_voc_to_yolo.py
```

### 2. GLCM Doku Feature'ları Çıkarma
```bash
python -c "import pandas as pd; from src.features.glcm import build_feature_dataset; build_feature_dataset(pd.read_csv('data/metadata.csv'), save_path='data/processed/glcm_features.parquet')"
```

### 3. Eğitilmiş Modeli Değerlendirme (Validation)
```python
from ultralytics import YOLO

# En iyi model ağırlıklarını yükleme
model = YOLO("models/yolo_runs/baseline_yolov8n/weights/best.pt")

# Doğrulama kümesinde çalıştırma
metrics = model.val(data="configs/steel_yolo.yaml")
print(f"Genel mAP@50: %{metrics.box.map50 * 100:.2f}")
```

---


---

## 📜 Lisans
Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.
```