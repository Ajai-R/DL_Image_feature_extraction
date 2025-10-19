import os
import cv2
import numpy as np
from skimage.feature import hog
from tqdm import tqdm

# ==== PATHS ====
DATASET_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset"     # <-- change this to your dataset path
HOG_FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\HOG_features"  # Folder for saving HOG .npy files

# Create folder if not exists
os.makedirs(HOG_FEATURES_DIR, exist_ok=True)

# ==== HOG PARAMETERS ====
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
IMAGE_SIZE = (150, 150)  # Keep same as dataset resolution

def extract_hog_features(image):
    """Extract HOG features from a single image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True
    )
    return features

def process_dataset(dataset_path):
    """Process dataset folder (train/test) and extract HOG features."""
    X = []
    y = []
    class_names = sorted(os.listdir(dataset_path))  # Ensures consistent class order

    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"\nProcessing class: {class_name}")
        for img_name in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                # Resize image to fixed size
                img = cv2.resize(img, IMAGE_SIZE)

                # Extract HOG features
                features = extract_hog_features(img)

                X.append(features)
                y.append(class_idx)
            except:
                print(f"⚠️ Error reading {img_path}")

    return np.array(X), np.array(y), class_names

print("=== Extracting HOG features for TRAIN set ===")
X_train, y_train, class_names = process_dataset(os.path.join(DATASET_DIR, "seg_train"))

print("\n=== Extracting HOG features for TEST set ===")
X_test, y_test, _ = process_dataset(os.path.join(DATASET_DIR, "seg_test"))

# ==== SAVE FEATURES ====
np.save(os.path.join(HOG_FEATURES_DIR, "hog_X_train.npy"), X_train)
np.save(os.path.join(HOG_FEATURES_DIR, "hog_y_train.npy"), y_train)
np.save(os.path.join(HOG_FEATURES_DIR, "hog_X_test.npy"), X_test)
np.save(os.path.join(HOG_FEATURES_DIR, "hog_y_test.npy"), y_test)

print("\n✅ HOG Feature Extraction Completed!")
print(f"Train Features: {X_train.shape}, Labels: {y_train.shape}")
print(f"Test Features: {X_test.shape}, Labels: {y_test.shape}")
print(f"Classes: {class_names}")
print(f"Features saved in: {HOG_FEATURES_DIR}")
