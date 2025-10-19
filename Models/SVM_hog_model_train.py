import os
import time
import joblib
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\HOG_features"
MODELS_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Models"
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================
# LOAD FEATURES
# =========================
print("📥 Loading HOG features...")
X_train = np.load(os.path.join(FEATURES_DIR, "hog_X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "hog_y_train.npy"))

X_test = np.load(os.path.join(FEATURES_DIR, "hog_X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "hog_y_test.npy"))

print(f"✅ Features loaded! Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

# =========================
# OPTIONAL PCA (SPEED-UP)
# =========================
USE_PCA = True
if USE_PCA and X_train.shape[1] > 500:
    print("🔄 Applying PCA to reduce dimensions...")
    pca = PCA(n_components=300)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print(f"✅ PCA completed! New feature size: {X_train.shape[1]}")

# =========================
# CHOOSE SVM MODEL
# =========================
if X_train.shape[0] > 5000:
    print("⚡ Using Linear SVM (Large dataset detected)")
    svm_clf = LinearSVC(C=1, max_iter=5000, verbose=1)
else:
    print("🌀 Using RBF SVM (Smaller dataset detected)")
    svm_clf = SVC(kernel="rbf", C=10, gamma="scale", verbose=True)

# =========================
# TRAINING WITH PROGRESS BAR
# =========================
print("\n🚀 Training SVM Classifier...")
start_time = time.time()

# Wrap fit() with tqdm to track progress
with tqdm(total=X_train.shape[0], desc="Training Progress", unit="samples") as pbar:
    svm_clf.fit(X_train, y_train)
    pbar.update(X_train.shape[0])

training_time = time.time() - start_time
print(f"✅ Training completed in {training_time:.2f} seconds ({training_time/60:.2f} mins)")

# =========================
# PREDICTION & EVALUATION
# =========================
print("\n🔍 Evaluating the model...")
y_pred = svm_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {acc:.4f}")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\n📌 Confusion Matrix:")
print(cm)

# =========================
# SAVE MODEL
# =========================
model_path = os.path.join(MODELS_DIR, "svm_hog_model.pkl")
joblib.dump(svm_clf, model_path)
print(f"\n💾 SVM model saved at: {model_path}")
