import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# =============================
# CONFIGURATION
# =============================
# Input image path
image_path = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset\seg_pred\1026.jpg"

# Output directory to save LBP visualizations
output_dir = r"D:\5th sem\Deep Learning\Dl_Assignment"
os.makedirs(output_dir, exist_ok=True)

# Parameters for LBP
radius = 3                # Radius of the circle
n_points = 8 * radius     # Number of sampling points
method = 'uniform'        # Method: 'default', 'ror', 'uniform', 'var'

# =============================
# LOAD IMAGE
# =============================
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert BGR to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale for LBP
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =============================
# LBP FEATURE EXTRACTION
# =============================
lbp = local_binary_pattern(gray, n_points, radius, method)

# Normalize LBP image for visualization
lbp_normalized = (lbp - lbp.min()) / (lbp.max() - lbp.min())

# =============================
# SAVE LBP VISUALIZATION
# =============================
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = os.path.join(output_dir, f"{image_name}_lbp.jpg")

plt.imsave(output_path, lbp_normalized, cmap='gray')
print(f"✅ LBP visualized image saved at: {output_path}")

# =============================
# VISUALIZE ORIGINAL VS LBP
# =============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(lbp_normalized, cmap='gray')
plt.title("LBP Features")
plt.axis("off")

plt.tight_layout()
plt.show()
