import time
import numpy as np
import matplotlib.pyplot as plt
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