# Real vs. Fake Audio Detection (Chatterbox) 🎙️🤖

This project is a machine learning-based system designed to distinguish between real human voices and synthetic (deepfake) voices generated using the **Chatterbox TTS** model.

## 🚀 Key Features
- **Signal Processing:** Advanced feature extraction using `librosa` (MFCC, Spectral Centroid, ZCR, etc.).
- **Machine Learning:** Classification powered by Support Vector Machines (SVM) with RBF kernel.
- **Accuracy:** Achieved **93.75% accuracy** on the test dataset.
- **Ready to Use:** Automated generation and classification scripts.


## 📂 Project Structure
```text
Real-vs-Fake-Audio-Detection/
├── data/               # Dataset (Real & Fake .wav files)
├── src/
│   ├── generator.py    # Generates synthetic samples via Chatterbox
│   └── classifier.py   # Feature extraction and SVM training
├── reports/            # Project reports and documentation
└── requirements.txt    # Python dependencies
```
🛠️ Installation & Usage

1. Clone the repository:

```
git clone [https://github.com/z-ozturk/Real-vs-Fake-Audio-Detection.git](https://github.com/z-ozturk/Real-vs-Fake-Audio-Detection.git)
cd Real-vs-Fake-Audio-Detection
```

2. Install dependencies:

```pip install -r requirements.txt```

3. Run the classifier:

```python src/classifier.py```


📊 Results & Performance

The model was evaluated using a confusion matrix and standard classification metrics.

Overall Accuracy: 93.75%
Fake Precision: 1.00 (Zero false alarms for real voices)
RBF SVM Parameters: C=15.0, Gamma='scale'

👥 Contributors
Eda TEKEŞ (eda.t.23@ogr.iu.edu.tr)
Selen GÜNEL (seleng@ogr.iu.edu.tr)

Zehra ÖZTÜRK (zehraozturk2023@ogr.iu.edu.tr)


