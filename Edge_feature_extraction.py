import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage import img_as_ubyte
from skimage.transform import resize
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================
DATASET_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset"  # Change if needed
EDGE_FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Edge_features"
TRAIN_DIR = os.path.join(DATASET_DIR, "seg_train")
TEST_DIR = os.path.join(DATASET_DIR, "seg_test")

# Create folder to save edge features
os.makedirs(EDGE_FEATURES_DIR, exist_ok=True)

# ==============================
# FUNCTION: Extract edge features
# ==============================
def extract_edge_features(image, sigma=1.0, size=(150, 150)):
    """
    Extract edge features using Canny detector.
    Resize all images to 'size' to ensure consistent feature length.
    """
    # Resize image
    image_resized = resize(image, size, anti_aliasing=True)

    # Convert to grayscale
    gray = rgb2gray(image_resized)
    gray_uint8 = img_as_ubyte(gray)

    # Apply Canny edge detection
    edges = canny(gray_uint8, sigma=sigma)

    # Flatten edges to 1D feature vector
    features = edges.astype(np.float32).ravel()
    return features

# ==============================
# FUNCTION: Load dataset & extract edge features
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
                feat = extract_edge_features(image)
                features.append(feat)
                labels.append(class_label)
            except Exception as e:
                print(f"⚠️ Skipping corrupted image: {img_path} | Error: {e}")

    return np.array(features), np.array(labels), classes

# ==============================
# MAIN SCRIPT
# ==============================
if __name__ == "__main__":
    print("=== Extracting Edge features for TRAIN set ===")
    X_train, y_train, classes = load_and_extract_features(TRAIN_DIR)

    print("\n=== Extracting Edge features for TEST set ===")
    X_test, y_test, _ = load_and_extract_features(TEST_DIR)

    # Save features as .npy files
    np.save(os.path.join(EDGE_FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(EDGE_FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(EDGE_FEATURES_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(EDGE_FEATURES_DIR, "y_test.npy"), y_test)

    # Final summary
    print("\n✅ Edge Feature Extraction Completed Successfully!")
    print(f"Train Features: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test Features:  {X_test.shape}, Labels: {y_test.shape}")
    print(f"Classes: {classes}")
    print(f"Features saved in: {EDGE_FEATURES_DIR}")
