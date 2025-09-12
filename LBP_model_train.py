import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# ============================
# CONFIGURATION
# ============================
FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\LBP_features"  # Change for LBP, Edge, DL
MODELS_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================
# LOAD FEATURES
# ============================
X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))

X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

# ============================
# TRAIN RANDOM FOREST
# ============================
print("Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(
    n_estimators=200,   # number of trees
    max_depth=None,     # full depth
    random_state=42,
    n_jobs=-1           # use all cores
)
rf_clf.fit(X_train, y_train)

# ============================
# PREDICTION & EVALUATION
# ============================
y_pred = rf_clf.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Detailed Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================
# SAVE MODEL
# ============================
import joblib
model_path = os.path.join(MODELS_DIR, "rf_lbp_model.pkl")
joblib.dump(rf_clf, model_path)
print(f"\nRandom Forest model saved at: {model_path}")
