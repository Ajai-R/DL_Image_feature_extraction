import cv2
import os
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# =============================
# CONFIGURATION
# =============================
# Input image path
image_path = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset\seg_pred\754.jpg"

# Output directory to save HOG visualizations
output_dir = r"D:\5th sem\Deep Learning\Dl_Assignment"
os.makedirs(output_dir, exist_ok=True)

# =============================
# LOAD IMAGE
# =============================
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert BGR to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale for HOG
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# =============================
# HOG FEATURE EXTRACTION
# =============================
hog_features, hog_image = hog(
    gray,
    orientations=9,        # Number of orientation bins
    pixels_per_cell=(8, 8), # Size of each cell
    cells_per_block=(2, 2), # Number of cells per block
    visualize=True,        # Get the HOG image
    block_norm='L2-Hys'
)

# Rescale the HOG image for better visibility
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# =============================
# SAVE HOG VISUALIZATION
# =============================
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = os.path.join(output_dir, f"{image_name}_hog.jpg")

plt.imsave(output_path, hog_image_rescaled, cmap='gray')
print(f"✅ HOG visualized image saved at: {output_path}")

# =============================
# VISUALIZE ORIGINAL VS HOG
# =============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(hog_image_rescaled, cmap='gray')
plt.title("HOG Features")
plt.axis("off")

plt.tight_layout()
plt.show()
