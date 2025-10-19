import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import joblib

# ============================
# CONFIGURATION
# ============================
FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\DL_features"
MODELS_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================
# LOAD FEATURES
# ============================
X_train = np.load(os.path.join(FEATURES_DIR, "X_train.npy"))
X_test = np.load(os.path.join(FEATURES_DIR, "X_test.npy"))
y_train = np.load(os.path.join(FEATURES_DIR, "y_train.npy"))
y_test = np.load(os.path.join(FEATURES_DIR, "y_test.npy"))

print(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")

# ============================
# ENCODE LABELS
# ============================
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# ============================
# PCA FOR DIMENSIONALITY REDUCTION (OPTIONAL)
# ============================
print("\nApplying PCA for dimensionality reduction...")
pca = PCA(n_components=512)  # Reduce from 2048 → 512
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# ============================
# TRAIN MLP CLASSIFIER
# ============================
print("\nTraining MLP Classifier on ResNet50 features...")
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    verbose=True
)

mlp_clf.fit(X_train, y_train)

# ============================
# EVALUATION
# ============================
y_pred = mlp_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ MLP Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ============================
# SAVE MODEL & PCA
# ============================
model_path = os.path.join(MODELS_DIR, "MLP_resnet50_model.pkl")
pca_path = os.path.join(MODELS_DIR, "MLP_resnet50_pca.pkl")

joblib.dump(mlp_clf, model_path)
joblib.dump(pca, pca_path)

print(f"\nMLP Model saved at: {model_path}")
print(f"PCA Model saved at: {pca_path}")
