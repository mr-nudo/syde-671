import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from filters_core import load_image_normalized, get_gaussian_kernel2d, unsharp_mask, align_images

os.makedirs("media", exist_ok=True)


def hybrid_image(im1, im2, cutoff_low, cutoff_high):
    """Low-pass filters im1 and high-pass filters im2 to construct a hybrid image."""
    G_low = get_gaussian_kernel2d(int(6 * cutoff_low + 1) | 1, cutoff_low)
    G_high = get_gaussian_kernel2d(int(6 * cutoff_high + 1) | 1, cutoff_high)

    low_pass = convolve2d(im1, G_low, mode='same', boundary='symm')
    low_pass_im2 = convolve2d(im2, G_high, mode='same', boundary='symm')
    high_pass = im2 - low_pass_im2

    hybrid = (low_pass + high_pass) / 2.0
    return np.clip(hybrid, 0.0, 1.0), low_pass, high_pass


def get_fft_spectrum(img_gray):
    """Computes log magnitude 2D FFT spectrum."""
    fshift = np.fft.fftshift(np.fft.fft2(img_gray))
    return np.log(np.abs(fshift) + 1e-8)


def process_hybrid_pair(path_a, path_b, cutoff_low, cutoff_high, output_path):
    """Pipeline for processing and saving an aligned hybrid image pair."""
    imA = load_image_normalized(path_a, mode="L")
    imB = load_image_normalized(path_b, mode="L")
    imA_aligned, imB_aligned = align_images(imA, imB)
    hybrid, _, _ = hybrid_image(imA_aligned, imB_aligned, cutoff_low, cutoff_high)
    plt.imsave(output_path, hybrid, cmap="gray")
    return imA_aligned, imB_aligned, hybrid


def main():
    # --- 2.1 Unsharp Masking ---
    taj = load_image_normalized("data/taj.jpg", mode="RGB")
    plt.imsave("media/taj_sharp.jpg", unsharp_mask(taj, alpha=1.5, sigma=2.0))

    custom = load_image_normalized("data/custom_building.jpg", mode="RGB")
    plt.imsave("media/custom_sharp.jpg", unsharp_mask(custom, alpha=2.0, sigma=2.0))

    G_blur = get_gaussian_kernel2d(11, 2.5)
    blurred = np.stack([convolve2d(custom[:, :, c], G_blur, mode='same', boundary='symm') for c in range(3)], axis=-1)
    blurred = np.clip(blurred, 0.0, 1.0)
    resharpened = unsharp_mask(blurred, alpha=2.5, sigma=2.5)

    plt.imsave("media/eval_original.jpg", custom)
    plt.imsave("media/eval_blurred.jpg", blurred)
    plt.imsave("media/eval_resharpened.jpg", resharpened)

    # --- 2.2 Hybrid Images & FFT Analysis ---
    im1_aligned, im2_aligned, im12 = process_hybrid_pair(
        "data/DerekPicture.jpg", "data/nutmeg.jpg", 12.0, 5.0, "media/hybrid_derek_nutmeg.jpg"
    )

    _, low_pass, high_pass = hybrid_image(im1_aligned, im2_aligned, 12.0, 5.0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    spectra = [
        ("Input A (Derek)", im1_aligned),
        ("Low-pass A", low_pass),
        ("Hybrid Image", im12),
        ("High-pass B", high_pass),
        ("Input B (Nutmeg)", im2_aligned),
    ]

    for ax, (title, img) in zip(axes, spectra):
        ax.imshow(get_fft_spectrum(img), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("media/hybrid_fft_analysis.jpg", dpi=150)
    plt.close()

    # Additional Hybrid Examples
    process_hybrid_pair("data/Obama-headshot.jpg", "data/Trump-headshot-2.jpg", 5.0, 3.5, "media/hybrid_trump_obama.jpg")
    process_hybrid_pair("data/tiger-front.jpg", "data/dog-front.jpg", 1.0, 20.0, "media/hybrid_tiger_dog.jpg")


if __name__ == "__main__":
    main()