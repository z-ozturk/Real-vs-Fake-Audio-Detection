# Gerçek vs. Sahte Ses Tespiti 🎙️🤖

Bu proje, gerçek insan sesleri ile **Chatterbox TTS** modeli kullanılarak üretilmiş sentetik (deepfake) sesleri makine öğrenimi ile birbirinden ayırt eden bir sistemdir.

## Özellikler
- `librosa` ile gelişmiş ses özelliği çıkarma (MFCC, Spektral Merkez, ZCR, RMS vb.)
- RBF çekirdekli Destek Vektör Makinesi (SVM) ile ikili sınıflandırma
- Test veri kümesinde **%93.75 doğruluk** oranı

---

## Proje Yapısı

```
Real-vs-Fake-Audio-Detection/
├── data/
│   ├── real/           # Gerçek insan sesi kayıtları (.wav)
│   └── fake/           # Chatterbox ile üretilmiş sentetik sesler (.wav)
├── src/
│   ├── generator.py    # Chatterbox TTS aracılığıyla sahte ses üretimi
│   ├── classifier.py   # Özellik çıkarma ve SVM modeli eğitimi
│   └── __init__.py
└── requirements.txt
```

---

## Ön Koşullar

- **Python 3.9** veya üzeri
- **pip**
- (İsteğe bağlı) CUDA destekli NVIDIA GPU — yalnızca `generator.py` içindir; CPU'da da çalışır ancak çok yavaş olabilir

---

## Kurulum

### 1. Depoyu klonla

```bash
git clone https://github.com/z-ozturk/Real-vs-Fake-Audio-Detection.git
cd Real-vs-Fake-Audio-Detection
```

### 2. Sanal ortam oluştur ve etkinleştir

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. PyTorch'u kur (CPU veya CUDA seç)

**Yalnızca CPU:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8 (GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> Diğer CUDA sürümleri için [pytorch.org/get-started](https://pytorch.org/get-started/locally/) adresine bakın.

### 4. Kalan bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

> `chatterbox-tts` paketi yalnızca `generator.py` için gereklidir. `classifier.py`'yi çalıştırmak için bu pakete ihtiyaç duyulmaz.

---

## Kullanım

### Adım 1 — Sınıflandırıcıyı çalıştır (önerilen başlangıç noktası)

`data/real/` ve `data/fake/` klasörlerindeki ses dosyaları depoya dahil edilmiştir. `generator.py` çalıştırılmadan doğrudan sınıflandırma yapılabilir:

```bash
python src/classifier.py
```

### Adım 2 — (İsteğe bağlı) Sahte ses üret

Veri setini sıfırdan yeniden oluşturmak istiyorsanız:

```bash
python src/generator.py
```

> ⚠️ **Bilinen sorun:** `generator.py` içindeki `PROJECT_ROOT` değişkeni yanlış hesaplanmaktadır — proje kökü yerine `src/` dizinine işaret etmektedir. Bu durum `data/real/` klasörünün bulunamamasına ve betiğin hata vererek durmasına yol açar. Geçici çözüm: `generator.py`'nin 7. satırındaki `os.path.dirname(os.path.abspath(__file__))` ifadesini `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` olarak düzeltin.

---

## Beklenen Çıktı

`classifier.py` çalıştırıldığında şu adımlar gerçekleşir:

1. Her `.wav` dosyasından 32 boyutlu bir özellik vektörü çıkarılır.
2. Veri seti %80 eğitim / %20 test olarak ayrılır.
3. RBF-SVM modeli (C=15.0, gamma='scale') eğitilir.
4. Konsola sınıflandırma raporu ve her örnek için tahmin tablosu yazdırılır.
5. Ayrı bir pencerede karışıklık matrisi gösterilir.

Örnek konsol çıktısı:

```
==================================================
             PROJECT PERFORMANCE REPORT
==================================================
OVERALL ACCURACY: 93.75%

CLASSIFICATION DETAILS:
              precision    recall  f1-score   support

    Real (0)       0.88      1.00      0.93         7
    Fake (1)       1.00      0.88      0.93         9

    accuracy                           0.94        16
```

---

## Sonuçlar

| Metrik | Değer |
|---|---|
| Genel Doğruluk | %93.75 |
| Sahte Hassasiyeti (Precision) | 1.00 |
| Gerçek Geri Çağırma (Recall) | 1.00 |
| SVM Çekirdeği | RBF (C=15.0, gamma='scale') |

---

## Katkıda Bulunanlar

- Eda TEKEŞ (eda.t.23@ogr.iu.edu.tr)
- Selen GÜNEL (seleng@ogr.iu.edu.tr)
- Zehra ÖZTÜRK (zehraozturk2023@ogr.iu.edu.tr)

---
---

# Real vs. Fake Audio Detection 🎙️🤖

A machine learning pipeline that distinguishes real human voices from synthetic (deepfake) voices generated with the **Chatterbox TTS** model.

## Features
- Advanced audio feature extraction with `librosa` (MFCC, Spectral Centroid, ZCR, RMS, etc.)
- Binary classification with an RBF-kernel Support Vector Machine (SVM)
- **93.75% accuracy** on the held-out test set

---

## Project Structure

```
Real-vs-Fake-Audio-Detection/
├── data/
│   ├── real/           # Real human voice recordings (.wav)
│   └── fake/           # Chatterbox-generated synthetic voices (.wav)
├── src/
│   ├── generator.py    # Synthetic voice generation via Chatterbox TTS
│   ├── classifier.py   # Feature extraction and SVM training
│   └── __init__.py
└── requirements.txt
```

---

## Prerequisites

- **Python 3.9** or higher
- **pip**
- (Optional) CUDA-capable NVIDIA GPU — only needed for `generator.py`; a CPU fallback exists but is very slow for TTS inference

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/z-ozturk/Real-vs-Fake-Audio-Detection.git
cd Real-vs-Fake-Audio-Detection
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install PyTorch (choose CPU or CUDA)

**CPU only:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8 (GPU):**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> For other CUDA versions see [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
```

> `chatterbox-tts` is only required for `generator.py`. You can run `classifier.py` without it.

---

## Usage

### Step 1 — Run the classifier (recommended starting point)

The `data/real/` and `data/fake/` audio files are already committed to the repository. You can run the classifier immediately without running the generator first:

```bash
python src/classifier.py
```

### Step 2 — (Optional) Generate synthetic audio

To recreate the fake audio dataset from scratch:

```bash
python src/generator.py
```

> ⚠️ **Known issue:** `generator.py` computes `PROJECT_ROOT` incorrectly — it resolves to the `src/` directory instead of the project root, so the script fails to locate `data/real/` and exits immediately. Workaround: change line 7 of `generator.py` from `os.path.dirname(os.path.abspath(__file__))` to `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))`.

---

## Expected Output

When `classifier.py` runs:

1. A 32-dimensional feature vector is extracted from every `.wav` file.
2. Data is split into 80 % training / 20 % test sets (stratified).
3. An RBF-SVM model is trained (C=15.0, gamma='scale').
4. A classification report and a per-sample prediction table are printed to the console.
5. A confusion-matrix window opens as a separate plot.

Sample console output:

```
==================================================
             PROJECT PERFORMANCE REPORT
==================================================
OVERALL ACCURACY: 93.75%

CLASSIFICATION DETAILS:
              precision    recall  f1-score   support

    Real (0)       0.88      1.00      0.93         7
    Fake (1)       1.00      0.88      0.93         9

    accuracy                           0.94        16
```

---

## Results

| Metric | Value |
|---|---|
| Overall Accuracy | 93.75 % |
| Fake Precision | 1.00 |
| Real Recall | 1.00 |
| SVM Kernel | RBF (C=15.0, gamma='scale') |

---

## Contributors

- Eda TEKEŞ (eda.t.23@ogr.iu.edu.tr)
- Selen GÜNEL (seleng@ogr.iu.edu.tr)
- Zehra ÖZTÜRK (zehraozturk2023@ogr.iu.edu.tr)
