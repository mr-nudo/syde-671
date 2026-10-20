import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend for eye-point selection
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d

os.makedirs("media", exist_ok=True)

########################
### Helper Functions ###
########################

def unsharp_mask(image, alpha=2.0, sigma=2.0):
    """
    Sharpens an image (2D grayscale or 3D RGB) using Unsharp Masking.
    """
    ksize = int(6 * sigma + 1) | 1
    g1d = cv2.getGaussianKernel(ksize, sigma)
    G = g1d @ g1d.T
    
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

def align_images(im1, im2):
    """
    1. Pre-rotates BOTH im1 (Derek) and im2 (Nutmeg) independently so eye lines are horizontal (0 deg).
    2. Translates both eye midpoints to canvas centers using mean-tone padding (prevents reflected ears).
    3. Rescales feature distances to match.
    4. Symmetrically crops off empty boundary margins so only ONE clean image remains.
    """
    print("Click TWO reference points on Image 1 (Left Eye -> Right Eye)...")
    fig1, ax1 = plt.subplots()
    ax1.imshow(im1, cmap='gray' if im1.ndim == 2 else None)
    ax1.set_title("Select 2 points on Image 1 (Left Eye -> Right Eye)")
    pts1 = plt.ginput(2, timeout=0)
    plt.close(fig1)

    print("Click TWO corresponding points on Image 2 (Left Eye -> Right Eye)...")
    fig2, ax2 = plt.subplots()
    ax2.imshow(im2, cmap='gray' if im2.ndim == 2 else None)
    ax2.set_title("Select 2 points on Image 2 (Left Eye -> Right Eye)")
    pts2 = plt.ginput(2, timeout=0)
    plt.close(fig2)

    if len(pts1) < 2 or len(pts2) < 2:
        raise ValueError("Error: You must click 2 points on BOTH images.")

    x1, y1 = np.array([p[0] for p in pts1]), np.array([p[1] for p in pts1])
    x2, y2 = np.array([p[0] for p in pts2]), np.array([p[1] for p in pts2])

    cx1, cy1 = np.mean(x1), np.mean(y1)
    cx2, cy2 = np.mean(x2), np.mean(y2)

    # -------------------------------------------------------------------------
    # STEP 1: Level both images to 0 degrees using mean-tone edge padding
    # -------------------------------------------------------------------------
    angle1 = np.degrees(np.arctan2(y1[1] - y1[0], x1[1] - x1[0]))
    angle2 = np.degrees(np.arctan2(y2[1] - y2[0], x2[1] - x2[0]))

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    mean1 = float(np.mean(im1))
    mean2 = float(np.mean(im2))

    M_rot1 = cv2.getRotationMatrix2D((cx1, cy1), angle1, 1.0)
    M_rot2 = cv2.getRotationMatrix2D((cx2, cy2), angle2, 1.0)

    im1 = cv2.warpAffine(im1, M_rot1, (w1, h1), borderMode=cv2.BORDER_CONSTANT, borderValue=mean1)
    im2 = cv2.warpAffine(im2, M_rot2, (w2, h2), borderMode=cv2.BORDER_CONSTANT, borderValue=mean2)

    # Update eye coordinates post-rotation
    pts1_homo = np.hstack([np.column_stack([x1, y1]), np.ones((2, 1))])
    pts2_homo = np.hstack([np.column_stack([x2, y2]), np.ones((2, 1))])

    pts1_rot = (M_rot1 @ pts1_homo.T).T
    pts2_rot = (M_rot2 @ pts2_homo.T).T

    x1, y1 = pts1_rot[:, 0], pts1_rot[:, 1]
    x2, y2 = pts2_rot[:, 0], pts2_rot[:, 1]

    cx1, cy1 = np.mean(x1), np.mean(y1)
    cx2, cy2 = np.mean(x2), np.mean(y2)

    # -------------------------------------------------------------------------
    # STEP 2: Center eye midpoints onto image centers without mirror reflections
    # -------------------------------------------------------------------------
    tx1, ty1 = (w1 / 2.0 - cx1), (h1 / 2.0 - cy1)
    tx2, ty2 = (w2 / 2.0 - cx2), (h2 / 2.0 - cy2)

    M_trans1 = np.float32([[1, 0, tx1], [0, 1, ty1]])
    M_trans2 = np.float32([[1, 0, tx2], [0, 1, ty2]])

    im1 = cv2.warpAffine(im1, M_trans1, (w1, h1), borderMode=cv2.BORDER_CONSTANT, borderValue=mean1)
    im2 = cv2.warpAffine(im2, M_trans2, (w2, h2), borderMode=cv2.BORDER_CONSTANT, borderValue=mean2)

    x1, y1 = x1 + tx1, y1 + ty1
    x2, y2 = x2 + tx2, y2 + ty2

    # -------------------------------------------------------------------------
    # STEP 3: Scale distance between eyes to match
    # -------------------------------------------------------------------------
    len1 = np.sqrt((y1[1] - y1[0])**2 + (x1[1] - x1[0])**2)
    len2 = np.sqrt((y2[1] - y2[0])**2 + (x2[1] - x2[0])**2)
    dscale = len2 / len1

    if dscale < 1:
        print("Image 1 is larger: downscaling...")
        im1 = cv2.resize(im1, (0, 0), fx=dscale, fy=dscale, interpolation=cv2.INTER_LINEAR)
    else:
        print("Image 2 is larger: downscaling...")
        im2 = cv2.resize(im2, (0, 0), fx=1.0/dscale, fy=1.0/dscale, interpolation=cv2.INTER_LINEAR)

    # -------------------------------------------------------------------------
    # STEP 4: Tight crop around center to remove padded borders completely
    # -------------------------------------------------------------------------
    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    min_h = int(min(h1, h2) * 0.85)  # 15% inner crop to discard boundary artifacts
    min_w = int(min(w1, w2) * 0.85)

    cy1_c, cx1_c = h1 // 2, w1 // 2
    cy2_c, cx2_c = h2 // 2, w2 // 2

    im1_cropped = im1[cy1_c - min_h//2 : cy1_c + min_h//2, cx1_c - min_w//2 : cx1_c + min_w//2]
    im2_cropped = im2[cy2_c - min_h//2 : cy2_c + min_h//2, cx2_c - min_w//2 : cx2_c + min_w//2]

    return im1_cropped, im2_cropped

def hybrid_image(im1, im2, cutoff_low, cutoff_high):
    """
    Low-pass filters im1 (Gaussian) and high-pass filters im2 (Impulse - Gaussian).
    Returns (hybrid_image, low_pass, high_pass).
    """
    ksize_low = int(6 * cutoff_low + 1) | 1
    g1d_low = cv2.getGaussianKernel(ksize_low, cutoff_low)
    G_low = g1d_low @ g1d_low.T

    ksize_high = int(6 * cutoff_high + 1) | 1
    g1d_high = cv2.getGaussianKernel(ksize_high, cutoff_high)
    G_high = g1d_high @ g1d_high.T

    # Low-pass filter im1
    low_pass = convolve2d(im1, G_low, mode='same', boundary='symm')

    # High-pass filter im2 (Original minus Low-pass)
    low_pass_im2 = convolve2d(im2, G_high, mode='same', boundary='symm')
    high_pass = im2 - low_pass_im2

    # Average low and high frequency components
    hybrid = (low_pass + high_pass) / 2.0
    return np.clip(hybrid, 0.0, 1.0), low_pass, high_pass

def get_fft_spectrum(img_gray):
    """
    Computes log magnitude 2D FFT spectrum using the prompt's exact equation:
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray_image)))))
    """
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    return np.log(np.abs(fshift) + 1e-8)

###################
### Section 2.1 ###
###################

# --- 1. Process Taj Mahal ---
taj_arr = np.array(Image.open("data/taj.jpg"), dtype=np.float64) / 255.0
plt.imsave("media/taj_sharp.jpg", unsharp_mask(taj_arr, alpha=1.5, sigma=2.0))

# --- 2. Process Custom Image ---
custom_arr = np.array(Image.open("data/custom_building.jpg"), dtype=np.float64) / 255.0
plt.imsave("media/custom_sharp.jpg", unsharp_mask(custom_arr, alpha=2.0, sigma=2.0))

# --- 3. Evaluation Test: Blur then Resharpen ---
g1d_blur = cv2.getGaussianKernel(11, 2.5)
G_blur = g1d_blur @ g1d_blur.T

blurred = np.zeros_like(custom_arr)
for c in range(3):
    blurred[:, :, c] = convolve2d(custom_arr[:, :, c], G_blur, mode='same', boundary='symm')
blurred = np.clip(blurred, 0.0, 1.0)

resharpened = unsharp_mask(blurred, alpha=2.5, sigma=2.5)

plt.imsave("media/eval_original.jpg", custom_arr)
plt.imsave("media/eval_blurred.jpg", blurred)
plt.imsave("media/eval_resharpened.jpg", resharpened)

###################
### Section 2.2 ###
###################

# --- 1. Load & Align Derek & Nutmeg ---
im1 = np.array(Image.open("data/DerekPicture.jpg").convert("L"), dtype=np.float64) / 255.0
im2 = np.array(Image.open("data/nutmeg.jpg").convert("L"), dtype=np.float64) / 255.0

# Click points in order: LEFT EYE first, then RIGHT EYE on both images
im1_aligned, im2_aligned = align_images(im1, im2)

# --- 2. Generate Favorite Hybrid Image ---
cutoff_low, cutoff_high = 12.0, 5.0
im12, low_pass, high_pass = hybrid_image(im1_aligned, im2_aligned, cutoff_low, cutoff_high)

plt.imsave("media/hybrid_derek_nutmeg.jpg", im12, cmap="gray")

# --- 3. 2D FFT Frequency Domain Analysis Grid (Step 2 Deliverable) ---
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(get_fft_spectrum(im1_aligned), cmap="gray"); axes[0].set_title("Input A (Derek)"); axes[0].axis("off")
axes[1].imshow(get_fft_spectrum(low_pass), cmap="gray"); axes[1].set_title("Low-pass A"); axes[1].axis("off")
axes[2].imshow(get_fft_spectrum(im12), cmap="gray"); axes[2].set_title("Hybrid Image"); axes[2].axis("off")
axes[3].imshow(get_fft_spectrum(high_pass), cmap="gray"); axes[3].set_title("High-pass B"); axes[3].axis("off")
axes[4].imshow(get_fft_spectrum(im2_aligned), cmap="gray"); axes[4].set_title("Input B (Nutmeg)"); axes[4].axis("off")

plt.tight_layout()
plt.savefig("media/hybrid_fft_analysis.jpg", dpi=150)
plt.close()


### Additional Presidential Image ###

imA = np.array(Image.open("data/Obama-headshot.jpg").convert("L"), dtype=np.float64) / 255.0
imB = np.array(Image.open("data/Trump-headshot-2.jpg").convert("L"), dtype=np.float64) / 255.0

imA_aligned, imB_aligned = align_images(imA, imB)

# Tune cutoffs per image pair (Low-pass imA, High-pass imB)
hybrid_custom, _, _ = hybrid_image(imA_aligned, imB_aligned, cutoff_low=5.0, cutoff_high=3.5)

plt.imsave("media/hybrid_trump_obama.jpg", hybrid_custom, cmap="gray")


### Additional Animal Image ###

imA = np.array(Image.open("data/tiger-front.jpg").convert("L"), dtype=np.float64) / 255.0
imB = np.array(Image.open("data/dog-front.jpg").convert("L"), dtype=np.float64) / 255.0


imA_aligned, imB_aligned = align_images(imA, imB)

# Tune cutoffs per image pair (Low-pass imA, High-pass imB)
hybrid_custom, _, _ = hybrid_image(imA_aligned, imB_aligned, cutoff_low=1.0, cutoff_high=20.0)

plt.imsave("media/hybrid_tiger_dog.jpg", hybrid_custom, cmap="gray")