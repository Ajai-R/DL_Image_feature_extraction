import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# ============================
# CONFIGURATION
# ============================
FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\DL_features"  # ResNet50 extracted features folder
MODELS_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================
# LOAD RESNET50 FEATURES
# ============================
print("Loading ResNet50 extracted features...")

X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

print(f"Train Features: {X_train.shape}, Train Labels: {y_train.shape}")
print(f"Test  Features: {X_test.shape}, Test Labels: {y_test.shape}")

# ============================
# TRAIN RANDOM FOREST CLASSIFIER
# ============================
print("\nTraining Random Forest on ResNet50 features...")
rf_clf = RandomForestClassifier(
    n_estimators=300,   # number of trees (slightly higher for DL features)
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train, y_train)

# ============================
# PREDICTION & EVALUATION
# ============================
print("\nEvaluating model...")
y_pred = rf_clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================
# SAVE MODEL
# ============================
model_path = os.path.join(MODELS_DIR, "rf_resnet50_model.pkl")
joblib.dump(rf_clf, model_path)
print(f"\n💾 Model saved at: {model_path}")
