import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. DYNAMIC CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "REAL_FOLDER": os.path.join(PROJECT_ROOT, "data", "real"),
    "FAKE_FOLDER": os.path.join(PROJECT_ROOT, "data", "fake"),
    "SAMPLE_RATE": 16000,
    "TRIM_DB": 25
}

def extract_comprehensive_features(path):
    """
    Extracts spectral and statistical features from an audio file.
    """
    try:
        y, sr = librosa.load(path, sr=CONFIG["SAMPLE_RATE"], mono=True)
        y, _ = librosa.effects.trim(y, top_db=CONFIG["TRIM_DB"])
        
        # Spectral Features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        zcr = librosa.feature.zero_crossing_rate(y)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        flatness = librosa.feature.spectral_flatness(y=y)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)

        # Statistical Summarization
        feature_vector = []
        for feat in [mfcc, delta_mfcc, delta2_mfcc, zcr, centroid, flatness, rolloff, rms]:
            feature_vector.extend([np.mean(feat), np.std(feat), np.max(feat), np.min(feat)])
        
        return np.array(feature_vector)
    except Exception as e:
        print(f"Error: Could not process {path}. {e}")
        return None

# --- 2. DATA COLLECTION ---
data, labels = [], []
print(f"Project Directory: {PROJECT_ROOT}")
print("Processing data and extracting features...")

for label_val, folder in [(0, CONFIG["REAL_FOLDER"]), (1, CONFIG["FAKE_FOLDER"])]:
    if not os.path.exists(folder):
        print(f"WARNING: Path {folder} not found! Please check your data directory.")
        continue
    
    files = [f for f in os.listdir(folder) if f.endswith(".wav")]
    folder_type = "Real" if label_val == 0 else "Fake"
    print(f"Scanning {len(files)} files from {folder_type} folder...")
    
    for file in files:
        feat = extract_comprehensive_features(os.path.join(folder, file))
        if feat is not None:
            data.append(feat)
            labels.append(label_val)

X = np.array(data)
y = np.array(labels)

if len(X) < 2:
    print("ERROR: Not enough data collected for training! Check your folders.")
else:
    # --- 3. MODEL TRAINING ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # SVM Model
    model = SVC(kernel='rbf', C=15.0, gamma='scale', probability=True)
    model.fit(X_train, y_train)

    # --- 4. DETAILED REPORTING ---
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\n" + "="*50)
    print("             PROJECT PERFORMANCE REPORT")
    print("="*50)
    print(f"OVERALL ACCURACY: {accuracy_score(y_test, y_pred)*100:.2f}%")

    print("\nCLASSIFICATION DETAILS:")
    print(classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

    print("\n" + "-"*50)
    print(f"{'No':<4} | {'Predict':<8} | {'Actual':<8} | {'Conf. Score':<12}")
    print("-" * 50)
    
    for i in range(len(y_test)):
        pred = y_pred[i]
        pred_label = "FAKE" if pred == 1 else "REAL"
        true_label = "FAKE" if y_test[i] == 1 else "REAL"
        confidence = probabilities[i][pred]
        print(f"{i+1:<4} | {pred_label:<8} | {true_label:<8} | {confidence:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)

    # Visualization
    plt.figure(figsize=(7,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', 
                xticklabels=['Pred: Real', 'Pred: Fake'], 
                yticklabels=['Actual: Real', 'Actual: Fake'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()