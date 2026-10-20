import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d
from filters_core import (
    load_image_normalized,
    get_gaussian_kernel2d,
    get_derivative_kernels,
    conv2d_4loops,
    conv2d_2loops,
)

os.makedirs("media", exist_ok=True)


def main():
    # --- 1.1 Convolution Benchmark ---
    img = load_image_normalized("data/selfie.jpg", mode="L")
    if img.shape[0] > 400 or img.shape[1] > 400:
        img = img[::2, ::2]

    box_9x9 = np.ones((9, 9), dtype=np.float64) / 81.0
    Dx, Dy = get_derivative_kernels()

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
    print(f"Max absolute diff: {np.max(np.abs(res_2loop - res_scipy)):.2e}")

    plt.imsave("media/selfie_box9x9.jpg", res_2loop, cmap="gray")
    plt.imsave("media/selfie_Dx.jpg", conv2d_2loops(img, Dx), cmap="gray")
    plt.imsave("media/selfie_Dy.jpg", conv2d_2loops(img, Dy), cmap="gray")

    # --- 1.2 Finite Difference Gradient Magnitude ---
    cam = load_image_normalized("data/cameraman.png", mode="L")
    Ix = convolve2d(cam, Dx, mode='same', boundary='fill', fillvalue=0)
    Iy = convolve2d(cam, Dy, mode='same', boundary='fill', fillvalue=0)
    grad_mag = np.sqrt(Ix**2 + Iy**2)
    edges_bin = (grad_mag > 0.25).astype(np.float64)

    plt.imsave("media/cam_Ix.jpg", Ix, cmap="gray")
    plt.imsave("media/cam_Iy.jpg", Iy, cmap="gray")
    plt.imsave("media/cam_grad_mag.jpg", grad_mag, cmap="gray")
    plt.imsave("media/cam_edges_bin.jpg", edges_bin, cmap="gray")

    # --- 1.3 Derivative of Gaussian (DoG) Filters ---
    G = get_gaussian_kernel2d(ksize=25, sigma=3)

    cam_smooth = convolve2d(cam, G, mode='same', boundary='fill', fillvalue=0)
    Ix_m1 = convolve2d(cam_smooth, Dx, mode='same', boundary='fill', fillvalue=0)
    Iy_m1 = convolve2d(cam_smooth, Dy, mode='same', boundary='fill', fillvalue=0)
    grad_mag_m1 = np.sqrt(Ix_m1**2 + Iy_m1**2)

    DoG_x = convolve2d(G, Dx, mode='same', boundary='fill', fillvalue=0)
    DoG_y = convolve2d(G, Dy, mode='same', boundary='fill', fillvalue=0)
    Ix_m2 = convolve2d(cam, DoG_x, mode='same', boundary='fill', fillvalue=0)
    Iy_m2 = convolve2d(cam, DoG_y, mode='same', boundary='fill', fillvalue=0)
    grad_mag_m2 = np.sqrt(Ix_m2**2 + Iy_m2**2)

    print(f"Max difference (Two-step vs DoG): {np.max(np.abs(grad_mag_m1 - grad_mag_m2)):.2e}")

    plt.imsave("media/cam_smooth.jpg", cam_smooth, cmap="gray")
    plt.imsave("media/cam_dog_grad_mag.jpg", grad_mag_m1, cmap="gray")
    plt.imsave("media/cam_dog_edges_bin.jpg", (grad_mag_m1 > 0.08).astype(np.float64), cmap="gray")
    plt.imsave("media/DoG_x.jpg", DoG_x, cmap="gray")
    plt.imsave("media/DoG_y.jpg", DoG_y, cmap="gray")

    # --- 1.4 Gradient Orientation ---
    theta = np.arctan2(Iy_m1, Ix_m1)
    H = (theta + np.pi) / (2.0 * np.pi)
    S = np.ones_like(H)
    V = grad_mag_m1 / np.max(grad_mag_m1)

    hsv_img = np.dstack((H, S, V))
    plt.imsave("media/cam_gradient_orientation_hsv.jpg", hsv_to_rgb(hsv_img))


if __name__ == "__main__":
    main()