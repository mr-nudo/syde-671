import cv2
import numpy as np
import matplotlib.pyplot as plt
from filters_core import load_image_normalized, get_gaussian_kernel2d


def build_gaussian_stack(img, levels=5, sigma_0=2.0, scale=2.0):
    """Builds a Gaussian pyramid stack."""
    g_stack = [img.copy()]
    curr_sigma = sigma_0
    for _ in range(1, levels):
        ksize = int(6 * curr_sigma + 1) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), curr_sigma)
        g_stack.append(blurred)
        curr_sigma *= scale
    return g_stack


def build_laplacian_stack(g_stack):
    """Builds a Laplacian stack from a Gaussian stack."""
    l_stack = [g_stack[k] - g_stack[k + 1] for k in range(len(g_stack) - 1)]
    l_stack.append(g_stack[-1])
    return l_stack


def multiresolution_blend(imgA, imgB, mask, levels=5):
    """Multiresolution spline blending (Burt & Adelson)."""
    g_A = build_gaussian_stack(imgA, levels=levels)
    g_B = build_gaussian_stack(imgB, levels=levels)
    g_M = build_gaussian_stack(mask, levels=levels)

    l_A = build_laplacian_stack(g_A)
    l_B = build_laplacian_stack(g_B)

    l_blended = [g_M[k] * l_A[k] + (1.0 - g_M[k]) * l_B[k] for k in range(levels)]
    blended_img = np.sum(l_blended, axis=0)

    return np.clip(blended_img, 0.0, 1.0), l_A, l_B, l_blended


def multiresolution_blend_color(imgA, imgB, mask, levels=5):
    """Multiresolution spline blending for RGB images across channels."""
    blended_channels = [
        multiresolution_blend(imgA[:, :, c], imgB[:, :, c], mask[:, :, c], levels=levels)[0]
        for c in range(3)
    ]
    return np.clip(np.stack(blended_channels, axis=-1), 0.0, 1.0)


def visualize_laplacian_breakdown(l_A, l_B, l_blended, levels, output_path):
    """Generates and saves the visual stack breakdown grid."""
    fig, axes = plt.subplots(levels, 3, figsize=(10, 2 * levels))

    for k in range(levels):
        for idx, (stack, title) in enumerate([(l_A, "Apple ($L_A$)"), (l_B, "Orange ($L_B$)"), (l_blended, "Blended ($L_C$)")]):
            norm_lap = (stack[k] - stack[k].min()) / (stack[k].max() - stack[k].min() + 1e-8)
            axes[k, idx].imshow(norm_lap)
            axes[k, idx].axis('off')
            if k == 0:
                axes[0, idx].set_title(title)
        axes[k, 0].set_ylabel(f"Level {k}")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    # --- Oraple Blending ---
    apple = load_image_normalized("data/burt_apple.png", mode="RGB")
    orange = load_image_normalized("data/burt_orange.png", mode="RGB")

    h, w, c = apple.shape
    orange = cv2.resize(orange, (w, h))

    mask = np.zeros((h, w, c), dtype=np.float64)
    mask[:, :w // 2, :] = 1.0

    num_levels = 5
    oraple, l_A, l_B, l_blended = multiresolution_blend(apple, orange, mask, levels=num_levels)
    oraple2 = multiresolution_blend_color(apple, orange, mask, levels=num_levels)

    plt.imsave("media/oraple.jpg", oraple)
    plt.imsave("media/oraple2.jpg", oraple2)

    visualize_laplacian_breakdown(l_A, l_B, l_blended, num_levels, "media/oraple_laplacian_stack.png")

    # --- Batman & Superman Blend ---
    batman = load_image_normalized("data/batman-logo.jpg", mode="RGB")
    superman = load_image_normalized("data/superman-logo.jpg", mode="RGB")

    h_bat, w_bat, _ = batman.shape
    superman_resized = cv2.resize(superman, (int(w_bat * 0.95), int(h_bat * 0.82)))
    h_sup, w_sup, _ = superman_resized.shape

    superman_aligned = np.ones((h_bat, w_bat, 3), dtype=np.float64)
    y_offset, x_offset = (h_bat - h_sup) // 2, (w_bat - w_sup) // 2
    superman_aligned[y_offset:y_offset + h_sup, x_offset:x_offset + w_sup] = superman_resized

    mask_split = np.zeros((h_bat, w_bat, 3), dtype=np.float64)
    mask_split[:, :w_bat // 2, :] = 1.0

    batman_superman_blend = multiresolution_blend_color(batman, superman_aligned, mask_split, levels=num_levels)
    plt.imsave("media/batman_superman_aligned_blend.jpg", batman_superman_blend)


if __name__ == "__main__":
    main()