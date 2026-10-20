import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from scipy.signal import convolve2d
from filters_core import conv2d_4loops, conv2d_2loops


###################
### Section 1.1 ###
###################
# 1. Load selfie as grayscale float in [0, 1]
# Resolution moderate (~300x300) so 4 loops run in reasonable time
selfie = Image.open("media/selfie.jpg").convert("L")
img = np.array(selfie, dtype=np.float64) / 255.0

if img.shape[0] > 400 or img.shape[1] > 400:
    img = img[::2, ::2]  # Downsample if image exceeds 0.8 MB requirement

# 2. Define required kernels
box_9x9 = np.ones((9, 9), dtype=np.float64) / 81.0
Dx = np.array([[1, 0, -1]], dtype=np.float64)
Dy = np.array([[1], [0], [-1]], dtype=np.float64)

# 3. Benchmark Runtimes (Box Filter)
t0 = time.time()
res_4loop = conv2d_4loops(img, box_9x9)
t_4loop = time.time() - t0

t0 = time.time()
res_2loop = conv2d_2loops(img, box_9x9)
t_2loop = time.time() - t0

t0 = time.time()
res_scipy = convolve2d(img, box_9x9, mode='same', boundary='fill', fillvalue=0)
t_scipy = time.time() - t0

print(f"4-Loop Runtime : {t_4loop:.4f}s")
print(f"2-Loop Runtime : {t_2loop:.4f}s")
print(f"SciPy Runtime  : {t_scipy:.4f}s")

# Numerical Verification
max_diff = np.max(np.abs(res_2loop - res_scipy))
print(f"Max absolute difference (Scratch vs SciPy): {max_diff:.2e}")

# 4. Compute Finite Difference outputs on Selfie
selfie_Dx = conv2d_2loops(img, Dx)
selfie_Dy = conv2d_2loops(img, Dy)

# 5. Save outputs for webpage display
plt.imsave("media/selfie_box9x9.jpg", res_2loop, cmap="gray")
plt.imsave("media/selfie_Dx.jpg", selfie_Dx, cmap="gray")
plt.imsave("media/selfie_Dy.jpg", selfie_Dy, cmap="gray")


###################
### Section 1.2 ###
###################

# 1. Load cameraman image in grayscale float normalized to [0, 1]
cam_img = Image.open("media/cameraman.png").convert("L")
cam = np.array(cam_img, dtype=np.float64) / 255.0

# 2. Define finite difference operators
Dx = np.array([[1, 0, -1]], dtype=np.float64)
Dy = np.array([[1], [0], [-1]], dtype=np.float64)

# 3. Compute partial derivatives
Ix = convolve2d(cam, Dx, mode='same', boundary='fill', fillvalue=0)
Iy = convolve2d(cam, Dy, mode='same', boundary='fill', fillvalue=0)

# 4. Calculate gradient magnitude
grad_mag = np.sqrt(Ix**2 + Iy**2)

# 5. Threshold to create binary edge map (tune threshold empirically)
threshold = 0.25
edges_bin = (grad_mag > threshold).astype(np.float64)

# 6. Save visual deliverables to media/ folder
plt.imsave("media/cam_Ix.jpg", Ix, cmap="gray")
plt.imsave("media/cam_Iy.jpg", Iy, cmap="gray")
plt.imsave("media/cam_grad_mag.jpg", grad_mag, cmap="gray")
plt.imsave("media/cam_edges_bin.jpg", edges_bin, cmap="gray")

###################
### Section 1.3 ###
###################

# 1. Load cameraman
cam_img = Image.open("media/cameraman.png").convert("L")
cam = np.array(cam_img, dtype=np.float64) / 255.0

# Define finite differences
Dx = np.array([[1, 0, -1]], dtype=np.float64)
Dy = np.array([[1], [0], [-1]], dtype=np.float64)

# 2. Create 2D Gaussian Kernel (e.g., ksize=25, sigma=3)
g1d = cv2.getGaussianKernel(25, 3)
G = g1d @ g1d.T  # Outer product creates 2D Gaussian

# --- METHOD 1: Two-Step Filtering ---
cam_smooth = convolve2d(cam, G, mode='same', boundary='fill', fillvalue=0)
Ix_m1 = convolve2d(cam_smooth, Dx, mode='same', boundary='fill', fillvalue=0)
Iy_m1 = convolve2d(cam_smooth, Dy, mode='same', boundary='fill', fillvalue=0)
grad_mag_m1 = np.sqrt(Ix_m1**2 + Iy_m1**2)
edges_m1 = (grad_mag_m1 > 0.08).astype(np.float64)  # Tuned threshold

# --- METHOD 2: DoG Filters (Single Step) ---
DoG_x = convolve2d(G, Dx, mode='same', boundary='fill', fillvalue=0)
DoG_y = convolve2d(G, Dy, mode='same', boundary='fill', fillvalue=0)

Ix_m2 = convolve2d(cam, DoG_x, mode='same', boundary='fill', fillvalue=0)
Iy_m2 = convolve2d(cam, DoG_y, mode='same', boundary='fill', fillvalue=0)
grad_mag_m2 = np.sqrt(Ix_m2**2 + Iy_m2**2)
edges_m2 = (grad_mag_m2 > 0.08).astype(np.float64)

# Compare Method 1 vs Method 2 difference
diff = np.max(np.abs(grad_mag_m1 - grad_mag_m2))
print(f"Max difference between Method 1 and Method 2: {diff:.2e}")

# Save deliverables to media/
plt.imsave("media/cam_smooth.jpg", cam_smooth, cmap="gray")
plt.imsave("media/cam_dog_grad_mag.jpg", grad_mag_m1, cmap="gray")
plt.imsave("media/cam_dog_edges_bin.jpg", edges_m1, cmap="gray")
plt.imsave("media/DoG_x.jpg", DoG_x, cmap="gray")
plt.imsave("media/DoG_y.jpg", DoG_y, cmap="gray")

###################
### Section 1.4 ###
###################

# 1. Use Ix and Iy computed from DoG filters in Part 1.3
theta = np.arctan2(Iy_m1, Ix_m1)  # Angle in radians [-pi, pi]

# 2. Normalize components for HSV
H = (theta + np.pi) / (2.0 * np.pi)  # Map [-pi, pi] -> [0, 1]
S = np.ones_like(H)                   # Full saturation

# Normalize gradient magnitude to [0, 1] for Value channel
grad_norm = grad_mag_m1 / np.max(grad_mag_m1)
V = grad_norm

# 3. Stack into HSV array and convert to RGB
hsv_img = np.dstack((H, S, V))
rgb_img = hsv_to_rgb(hsv_img)

# 4. Save visual output for webpage
plt.imsave("media/cam_gradient_orientation_hsv.jpg", rgb_img)