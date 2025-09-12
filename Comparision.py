import os
import time
import numpy as np
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score,
    classification_report
)
from sklearn.preprocessing import LabelBinarizer

# -------------------------------
# Configuration
# -------------------------------
BASE_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment"
MODELS_DIR = os.path.join(BASE_DIR, "Models")

MODEL_TO_FEATURES = {
    "RF_HOG": "HOG_features",
    "RF_LBP": "LBP_features",
    "RF_EDGE": "Edge_features",
    "RF_DL": "DL_features",
    "MLP_DL": "DL_features"
}

MODEL_FILES = {
    "RF_HOG": os.path.join(MODELS_DIR, "rf_hog_model.pkl"),
    "RF_LBP": os.path.join(MODELS_DIR, "rf_lbp_model.pkl"),
    "RF_EDGE": os.path.join(MODELS_DIR, "rf_edge_model.pkl"),
    "RF_DL": os.path.join(MODELS_DIR, "rf_resnet50_model.pkl"),
    "MLP_DL": os.path.join(MODELS_DIR, "MLP_resnet50_model.pkl")
}

PCA_FILE = os.path.join(MODELS_DIR, "MLP_resnet50_pca.pkl")
LABEL_ENCODER_FILE = os.path.join(MODELS_DIR, "label_encoder.pkl")

# -------------------------------
# Load Features
# -------------------------------
def load_features(model_name):
    feat_dir = os.path.join(BASE_DIR, MODEL_TO_FEATURES[model_name])
    if model_name == "RF_HOG":
        X_test = np.load(os.path.join(feat_dir, "hog_X_test.npy"))
        y_test = np.load(os.path.join(feat_dir, "hog_y_test.npy"))
    else:
        X_test = np.load(os.path.join(feat_dir, "X_test.npy"))
        y_test = np.load(os.path.join(feat_dir, "y_test.npy"))
    return X_test, y_test

# -------------------------------
# Load Label Encoder for MLP
# -------------------------------
label_encoder = None
if os.path.exists(LABEL_ENCODER_FILE):
    label_encoder = joblib.load(LABEL_ENCODER_FILE)

# -------------------------------
# Evaluate Models
# -------------------------------
results = []
conf_matrices = {}

for model_name, model_path in MODEL_FILES.items():
    print(f"\n🔹 Evaluating {model_name}...")

    # Load trained model
    model = joblib.load(model_path)

    # Load features
    X_test, y_test = load_features(model_name)

    # Apply PCA for MLP_DL if available
    if model_name == "MLP_DL" and os.path.exists(PCA_FILE):
        pca = joblib.load(PCA_FILE)
        X_test = pca.transform(X_test)

    # Convert labels to numeric for MLP if label encoder exists
    if model_name == "MLP_DL" and label_encoder is not None:
        y_test_enc = label_encoder.transform(y_test)
    else:
        y_test_enc = y_test

    # Measure prediction time
    start_time = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start_time

    # Convert predictions back to original labels for MLP
    if model_name == "MLP_DL" and label_encoder is not None:
        y_pred_labels = label_encoder.inverse_transform(y_pred)
        y_test_labels = np.array(y_test).astype(str)
    else:
        y_pred_labels = np.array(y_pred).astype(str)
        y_test_labels = np.array(y_test).astype(str)

    # Confusion matrix
    conf_matrices[model_name] = confusion_matrix(y_test_labels, y_pred_labels)

    # Metrics
    acc = accuracy_score(y_test_labels, y_pred_labels)
    prec = precision_score(y_test_labels, y_pred_labels, average='macro', zero_division=0)
    rec = recall_score(y_test_labels, y_pred_labels, average='macro', zero_division=0)
    f1 = f1_score(y_test_labels, y_pred_labels, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y_test_labels, y_pred_labels)

    # ROC-AUC (if binary or multi-class)
    try:
        lb = LabelBinarizer()
        y_test_bin = lb.fit_transform(y_test_labels)
        y_pred_bin = lb.transform(y_pred_labels)
        if y_test_bin.shape[1] == 1:  # Binary
            roc_auc = roc_auc_score(y_test_bin, y_pred_bin)
        else:  # Multi-class
            roc_auc = roc_auc_score(y_test_bin, y_pred_bin, average='macro', multi_class='ovr')
    except Exception:
        roc_auc = np.nan

    results.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "Cohen_Kappa": kappa,
        "ROC_AUC": roc_auc,
        "Prediction_Time(s)": pred_time
    })

# -------------------------------
# Save Results to CSV
# -------------------------------
df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(BASE_DIR, "Model_Metrics_Comparison.csv"), index=False)
print("\n✅ Model metrics saved to 'Model_Metrics_Comparison.csv'")

# -------------------------------
# Visualize Accuracy Comparison
# -------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=df_results, x="Model", y="Accuracy", palette="viridis")
plt.title("Model Accuracy Comparison", fontsize=14)
plt.ylabel("Accuracy")
plt.xlabel("Model")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# -------------------------------
# Visualize Confusion Matrices
# -------------------------------
for model_name, cm in conf_matrices.items():
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
