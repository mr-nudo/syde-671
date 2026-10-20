import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

def unsharp_mask(image, alpha=1.5, sigma=2.0):
    """
    Sharpens an image (2D grayscale or 3D RGB) using Unsharp Masking.
    """
    # Generate 2D Gaussian kernel
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    g1d = cv2.getGaussianKernel(ksize, sigma)
    G = g1d @ g1d.T
    
    # Process RGB channels independently if color image
    if image.ndim == 3:
        sharpened = np.zeros_like(image)
        for c in range(3):
            low = convolve2d(image[:, :, c], G, mode='same', boundary='symm')
            high = image[:, :, c] - low
            sharpened[:, :, c] = image[:, :, c] + alpha * high
    else:
        low = convolve2d(image, G, mode='same', boundary='symm')
        high = image - low
        sharpened = image + alpha * high
        
    return np.clip(sharpened, 0.0, 1.0)

# --- 1. Process Taj Mahal ---
taj = Image.open("media/taj.jpg")
taj_arr = np.array(taj, dtype=np.float64) / 255.0
taj_sharp = unsharp_mask(taj_arr, alpha=1.5, sigma=2.0)

plt.imsave("media/taj_sharp.jpg", taj_sharp)

# --- 2. Process Custom Image ---
custom = Image.open("media/custom_building.jpg")
custom_arr = np.array(custom, dtype=np.float64) / 255.0
custom_sharp = unsharp_mask(custom_arr, alpha=1.5, sigma=2.0)

plt.imsave("media/custom_sharp.jpg", custom_sharp)

# --- 3. Evaluation Test: Blur then Resharpen ---
ksize = 13
g1d = cv2.getGaussianKernel(ksize, 3.0)
G_blur = g1d @ g1d.T

# Artificially blur custom image
blurred = np.zeros_like(custom_arr)
for c in range(3):
    blurred[:, :, c] = convolve2d(custom_arr[:, :, c], G_blur, mode='same', boundary='symm')
blurred = np.clip(blurred, 0.0, 1.0)

# Re-sharpen the blurred image
resharpened = unsharp_mask(blurred, alpha=2.0, sigma=3.0)

plt.imsave("media/eval_original.jpg", custom_arr)
plt.imsave("media/eval_blurred.jpg", blurred)
plt.imsave("media/eval_resharpened.jpg", resharpened)