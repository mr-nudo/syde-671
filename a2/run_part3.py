import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

def build_gaussian_stack(img, levels=5, sigma_0=2.0, scale=2.0):
    g_stack = [img.copy()]
    curr_sigma = sigma_0
    for _ in range(1, levels):
        ksize = int(6 * curr_sigma + 1) | 1
        blurred = cv2.GaussianBlur(img, (ksize, ksize), curr_sigma)
        g_stack.append(blurred)
        curr_sigma *= scale
    return g_stack

def build_laplacian_stack(g_stack):
    l_stack = []
    for k in range(len(g_stack) - 1):
        l_stack.append(g_stack[k] - g_stack[k + 1])
    l_stack.append(g_stack[-1])  # Top level remains the blurred low-pass
    return l_stack

def multiresolution_blend(imgA, imgB, mask, levels=5):
    """
    Multiresolution spline blending (Burt & Adelson).
    imgA, imgB, and mask must have the exact same shape and range [0, 1].
    """
    # 1. Build stacks
    g_A = build_gaussian_stack(imgA, levels=levels)
    g_B = build_gaussian_stack(imgB, levels=levels)
    g_M = build_gaussian_stack(mask, levels=levels)
    
    l_A = build_laplacian_stack(g_A)
    l_B = build_laplacian_stack(g_B)
    
    # 2. Blend each Laplacian level using the Gaussian mask stack
    l_blended = []
    for k in range(levels):
        l_k = g_M[k] * l_A[k] + (1.0 - g_M[k]) * l_B[k]
        l_blended.append(l_k)
        
    # 3. Reconstruct final image
    blended_img = np.sum(l_blended, axis=0)
    return np.clip(blended_img, 0.0, 1.0), l_A, l_B, l_blended

def multiresolution_blend_color(imgA, imgB, mask, levels=5):
    """
    Multiresolution spline blending for RGB images.
    imgA, imgB, mask: (H, W, 3) arrays normalized in [0.0, 1.0]
    """
    blended_channels = []
    for c in range(3):
        blended_c, _, _, _ = multiresolution_blend(
            imgA[:, :, c], imgB[:, :, c], mask[:, :, c], levels=levels
        )
        blended_channels.append(blended_c)
        
    blended_rgb = np.stack(blended_channels, axis=-1)
    return np.clip(blended_rgb, 0.0, 1.0)

#########################
### Section 2.3 & 2.4 ###
#########################

# 1. Load and normalize input images to [0, 1]
apple = np.array(Image.open("data/burt_apple.png").convert("RGB"), dtype=np.float64) / 255.0
orange = np.array(Image.open("data/burt_orange.png").convert("RGB"), dtype=np.float64) / 255.0

# Ensure images are the exact same shape
h, w, c = apple.shape
orange = cv2.resize(orange, (w, h))

# 2. Create vertical split mask (1.0 on left for Apple, 0.0 on right for Orange)
mask = np.zeros((h, w, c), dtype=np.float64)
mask[:, :w // 2, :] = 1.0

# 3. Perform Multiresolution Spline Blending (e.g., 5 levels)
num_levels = 5
oraple, l_A, l_B, l_blended = multiresolution_blend(apple, orange, mask, levels=num_levels)
oraple2 = multiresolution_blend_color(apple, orange, mask, levels=num_levels)

# Save final Oraple result
plt.imsave("media/oraple.jpg", oraple)
plt.imsave("media/oraple2.jpg", oraple2)

################################################
### Visualization: Laplacian Stack Breakdown ###
################################################

fig, axes = plt.subplots(num_levels, 3, figsize=(10, 2 * num_levels))

for k in range(num_levels):
    # Normalize Laplacian levels to [0, 1] for visual display
    lap_apple = (l_A[k] - l_A[k].min()) / (l_A[k].max() - l_A[k].min() + 1e-8)
    lap_orange = (l_B[k] - l_B[k].min()) / (l_B[k].max() - l_B[k].min() + 1e-8)
    lap_blend = (l_blended[k] - l_blended[k].min()) / (l_blended[k].max() - l_blended[k].min() + 1e-8)
    
    axes[k, 0].imshow(lap_apple)
    axes[k, 0].set_ylabel(f"Level {k}")
    axes[k, 0].axis('off')
    
    axes[k, 1].imshow(lap_orange)
    axes[k, 1].axis('off')
    
    axes[k, 2].imshow(lap_blend)
    axes[k, 2].axis('off')

axes[0, 0].set_title("Apple Stack ($L_A$)")
axes[0, 1].set_title("Orange Stack ($L_B$)")
axes[0, 2].set_title("Blended Stack ($L_C$)")

plt.tight_layout()
plt.savefig("media/oraple_laplacian_stack.png", bbox_inches='tight', dpi=300)
plt.close()

# =========================================================================
# Custom Multiresolution Blend: Batman & Superman Logos (RGB + Custom Mask)
# =========================================================================

# 1. Load color images
batman = np.array(Image.open("data/batman-logo.jpg").convert("RGB"), dtype=np.float64) / 255.0
superman = np.array(Image.open("data/superman-logo.jpg").convert("RGB"), dtype=np.float64) / 255.0

# 2. Resize and shift Superman to match Batman's vertical border span
h_bat, w_bat, _ = batman.shape

# Scale Superman slightly smaller (~82% height) and pad/crop to center the shield borders
superman_resized = cv2.resize(superman, (int(w_bat * 0.95), int(h_bat * 0.82)))
h_sup, w_sup, _ = superman_resized.shape

# Place resized Superman into a canvas matching Batman's dimensions
superman_aligned = np.ones((h_bat, w_bat, 3), dtype=np.float64)  # White canvas
y_offset = (h_bat - h_sup) // 2
x_offset = (w_bat - w_sup) // 2
superman_aligned[y_offset:y_offset + h_sup, x_offset:x_offset + w_sup] = superman_resized

# 3. Create vertical split mask
mask_split = np.zeros((h_bat, w_bat, 3), dtype=np.float64)
mask_split[:, :w_bat // 2, :] = 1.0

# 4. Perform Color Multiresolution Blend
num_levels = 5
batman_superman_blend = multiresolution_blend_color(
    batman, superman_aligned, mask_split, levels=num_levels
)

plt.imsave("media/batman_superman_aligned_blend.jpg", batman_superman_blend)