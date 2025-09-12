import os
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
DATASET_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset"  # Change if needed
LBP_FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\LBP_features"
TRAIN_DIR = os.path.join(DATASET_DIR, "seg_train")
TEST_DIR = os.path.join(DATASET_DIR, "seg_test")

# Create folder to save LBP features
os.makedirs(LBP_FEATURES_DIR, exist_ok=True)

# ==============================
# FUNCTION: Extract LBP histogram
# ==============================
def extract_lbp_features(image, P=8, R=1):
    """
    Extract LBP features for an image.
    P = number of circular points (neighbors)
    R = radius of circle
    """
    # Convert image to grayscale (0-1)
    gray = rgb2gray(image)
    # Convert grayscale float image to 8-bit unsigned integer (0-255)
    gray_uint8 = img_as_ubyte(gray)

    # Compute LBP
    lbp = local_binary_pattern(gray_uint8, P, R, method="uniform")

    # Build histogram with fixed bins = P + 2 (as recommended for uniform patterns)
    n_bins = P + 2
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)

    return hist

# ==============================
# FUNCTION: Load dataset & extract LBP features
# ==============================
def load_and_extract_features(dataset_path):
    features = []
    labels = []
    classes = sorted(os.listdir(dataset_path))

    for class_label in classes:
        class_path = os.path.join(dataset_path, class_label)
        if not os.path.isdir(class_path):
            continue

        print(f"\n🔹 Processing class: {class_label}")
        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            try:
                image = imread(img_path)
                hist = extract_lbp_features(image)
                features.append(hist)
                labels.append(class_label)
            except Exception as e:
                print(f"⚠️ Skipping corrupted image: {img_path} | Error: {e}")

    return np.array(features), np.array(labels), classes

# ==============================
# MAIN SCRIPT
# ==============================
if __name__ == "__main__":
    print("=== Extracting LBP features for TRAIN set ===")
    X_train, y_train, classes = load_and_extract_features(TRAIN_DIR)

    print("\n=== Extracting LBP features for TEST set ===")
    X_test, y_test, _ = load_and_extract_features(TEST_DIR)

    # Save features as .npy files
    np.save(os.path.join(LBP_FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(LBP_FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(LBP_FEATURES_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(LBP_FEATURES_DIR, "y_test.npy"), y_test)

    # Final summary
    print("\n✅ LBP Feature Extraction Completed Successfully!")
    print(f"Train Features: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test Features:  {X_test.shape}, Labels: {y_test.shape}")
    print(f"Classes: {classes}")
    print(f"Features saved in: {LBP_FEATURES_DIR}")
