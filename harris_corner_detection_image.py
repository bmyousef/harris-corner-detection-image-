"""
Lab 3 - Harris Corner Detection on a still image
Usage: python harris_corner_detection_image.py
Place harris_corner_detection_image.jpg in the same folder as this script.
"""
import os
import cv2
import numpy as np

# ---------- Parameters ----------
BLOCK_SIZE = 2        # neighborhood size considered for each pixel
APERTURE_SIZE = 3     # Sobel operator aperture (odd number)
K = 0.04              # Harris free parameter (0.04 - 0.06)
THRESH_RATIO = 0.05   # keep responses > 5% of the max response
RADIUS = 4            # circle radius used for drawing corners

# ---------- Load image ----------
here = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(here, "harris_corner_detection_image.jpg")
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Could not read image: {path}")

original = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)  # cornerHarris requires float32

# ---------- Harris corner detection ----------
dst = cv2.cornerHarris(gray, BLOCK_SIZE, APERTURE_SIZE, K)

# Dilate the response map to make corner points easier to see / merge neighbors
dst = cv2.dilate(dst, None)

# Threshold: filter out weak corners
threshold = THRESH_RATIO * dst.max()
ys, xs = np.where(dst > threshold)

# Draw circles on the original image
result = img.copy()
for x, y in zip(xs, ys):
    cv2.circle(result, (int(x), int(y)), RADIUS, (0, 0, 255), 1)

print(f"Corner pixels detected: {len(xs)}")

# ---------- Save before/after screenshots (for submission) ----------
cv2.imwrite(os.path.join(here, "image_before.png"), original)
cv2.imwrite(os.path.join(here, "image_after.png"), result)

# ---------- Display ----------
cv2.imshow("Original Image", original)
cv2.imshow("Harris Corners", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
