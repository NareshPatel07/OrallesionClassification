from PIL import Image
import numpy as np

# Load image
img = Image.open("dataset/cancerous/1.jpg")
img_array = np.array(img)

print("Original Pixel Matrix:\n", img_array)
print("Original Min:", img_array.min(), "Max:", img_array.max())

# ---------------- NORMALIZATION ----------------
# Convert to float
img_norm = img_array.astype("float32") / 255.0

print("\nNormalized Pixel Matrix:\n", img_norm)
print("Normalized Min:", img_norm.min(), "Max:", img_norm.max())
