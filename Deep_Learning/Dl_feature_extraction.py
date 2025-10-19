import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import models, transforms

# ==============================
# CONFIGURATION
# ==============================
DATASET_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset"
DL_FEATURES_DIR = r"D:\5th sem\Deep Learning\Dl_Assignment\DL_features"
TRAIN_DIR = os.path.join(DATASET_DIR, "seg_train")
TEST_DIR = os.path.join(DATASET_DIR, "seg_test")

# Create folder to save DL features
os.makedirs(DL_FEATURES_DIR, exist_ok=True)

# ==============================
# DEVICE
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================
# MODEL: Pretrained ResNet50
# ==============================
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))  # Remove last fc layer
resnet.to(device)
resnet.eval()  # Set to evaluation mode

# ==============================
# TRANSFORMS
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

# ==============================
# FUNCTION: Extract DL features
# ==============================
def extract_dl_features(image_path):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor)
    features = features.cpu().numpy().flatten()  # Flatten 2048-dim feature vector
    return features

# ==============================
# FUNCTION: Load dataset & extract features
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
                feat = extract_dl_features(img_path)
                features.append(feat)
                labels.append(class_label)
            except Exception as e:
                print(f"⚠️ Skipping corrupted image: {img_path} | Error: {e}")

    return np.array(features), np.array(labels), classes

# ==============================
# MAIN SCRIPT
# ==============================
if __name__ == "__main__":
    print("=== Extracting DL features for TRAIN set ===")
    X_train, y_train, classes = load_and_extract_features(TRAIN_DIR)

    print("\n=== Extracting DL features for TEST set ===")
    X_test, y_test, _ = load_and_extract_features(TEST_DIR)

    # Save features as .npy files
    np.save(os.path.join(DL_FEATURES_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DL_FEATURES_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DL_FEATURES_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DL_FEATURES_DIR, "y_test.npy"), y_test)

    print("\n✅ DL Feature Extraction Completed Successfully!")
    print(f"Train Features: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test Features:  {X_test.shape}, Labels: {y_test.shape}")
    print(f"Classes: {classes}")
    print(f"Features saved in: {DL_FEATURES_DIR}")
