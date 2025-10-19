import cv2
import os
import matplotlib.pyplot as plt

# =============================
# CONFIGURATION
# =============================
# Input image path
image_path = r"D:\5th sem\Deep Learning\Dl_Assignment\Dataset\seg_pred\1020.jpg"

# Output directory to save extracted edge images
output_dir = r"D:\5th sem\Deep Learning\Dl_Assignment"
os.makedirs(output_dir, exist_ok=True)

# =============================
# LOAD IMAGE
# =============================
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Convert image from BGR to RGB for visualization
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# =============================
# EDGE FEATURE EXTRACTION
# =============================

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny Edge Detection
edges = cv2.Canny(blurred, threshold1=100, threshold2=200)

# =============================
# SAVE EDGE IMAGE
# =============================
# Create a proper filename for output
image_name = os.path.splitext(os.path.basename(image_path))[0]
output_path = os.path.join(output_dir, f"{image_name}_edges.jpg")

cv2.imwrite(output_path, edges)
print(f"✅ Edge-detected image saved at: {output_path}")

# =============================
# VISUALIZE ORIGINAL VS EDGES
# =============================
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Features (Canny)")
plt.axis("off")

plt.tight_layout()
plt.show()
