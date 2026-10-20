import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.signal import convolve2d


def load_image_normalized(filepath, mode="RGB"):
    """Loads an image file as a float64 array normalized to [0.0, 1.0]."""
    img = Image.open(filepath).convert(mode)
    return np.array(img, dtype=np.float64) / 255.0


def get_gaussian_kernel2d(ksize, sigma):
    """Generates a normalized 2D Gaussian kernel."""
    g1d = cv2.getGaussianKernel(ksize, sigma)
    return g1d @ g1d.T


def get_derivative_kernels():
    """Returns standard horizontal (Dx) and vertical (Dy) finite difference kernels."""
    Dx = np.array([[1, 0, -1]], dtype=np.float64)
    Dy = np.array([[1], [0], [-1]], dtype=np.float64)
    return Dx, Dy


def pad_image_zeros(img, pad_h, pad_w):
    """Pads 2D grayscale image borders with zeros."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)


def conv2d_4loops(img, kernel):
    """2D Convolution using 4 nested loops."""
    k_flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    H_in, W_in = img.shape
    H_k, W_k = kernel.shape
    pad_h, pad_w = H_k // 2, W_k // 2

    padded = pad_image_zeros(img, pad_h, pad_w)
    output = np.zeros_like(img, dtype=np.float64)

    for i in range(H_in):
        for j in range(W_in):
            acc = 0.0
            for m in range(H_k):
                for n in range(W_k):
                    acc += padded[i + m, j + n] * k_flipped[m, n]
            output[i, j] = acc

    return output


def conv2d_2loops(img, kernel):
    """2D Convolution using 2 nested loops with vectorized slice multiplication."""
    k_flipped = np.flip(np.flip(kernel, axis=0), axis=1)
    H_in, W_in = img.shape
    H_k, W_k = kernel.shape
    pad_h, pad_w = H_k // 2, W_k // 2

    padded = pad_image_zeros(img, pad_h, pad_w)
    output = np.zeros_like(img, dtype=np.float64)

    for i in range(H_in):
        for j in range(W_in):
            patch = padded[i:i + H_k, j:j + W_k]
            output[i, j] = np.sum(patch * k_flipped)

    return output


def unsharp_mask(image, alpha=2.0, sigma=2.0):
    """Sharpens an image using Unsharp Masking with Gaussian blurring."""
    ksize = int(6 * sigma + 1) | 1
    G = get_gaussian_kernel2d(ksize, sigma)

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
    """Aligns two images interactively based on user-selected eye positions."""
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

    angle1 = np.degrees(np.arctan2(y1[1] - y1[0], x1[1] - x1[0]))
    angle2 = np.degrees(np.arctan2(y2[1] - y2[0], x2[1] - x2[0]))

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    mean1, mean2 = float(np.mean(im1)), float(np.mean(im2))

    M_rot1 = cv2.getRotationMatrix2D((cx1, cy1), angle1, 1.0)
    M_rot2 = cv2.getRotationMatrix2D((cx2, cy2), angle2, 1.0)

    im1 = cv2.warpAffine(im1, M_rot1, (w1, h1), borderMode=cv2.BORDER_CONSTANT, borderValue=mean1)
    im2 = cv2.warpAffine(im2, M_rot2, (w2, h2), borderMode=cv2.BORDER_CONSTANT, borderValue=mean2)

    pts1_homo = np.hstack([np.column_stack([x1, y1]), np.ones((2, 1))])
    pts2_homo = np.hstack([np.column_stack([x2, y2]), np.ones((2, 1))])

    pts1_rot = (M_rot1 @ pts1_homo.T).T
    pts2_rot = (M_rot2 @ pts2_homo.T).T

    x1, y1 = pts1_rot[:, 0], pts1_rot[:, 1]
    x2, y2 = pts2_rot[:, 0], pts2_rot[:, 1]

    cx1, cy1 = np.mean(x1), np.mean(y1)
    cx2, cy2 = np.mean(x2), np.mean(y2)

    tx1, ty1 = (w1 / 2.0 - cx1), (h1 / 2.0 - cy1)
    tx2, ty2 = (w2 / 2.0 - cx2), (h2 / 2.0 - cy2)

    M_trans1 = np.float32([[1, 0, tx1], [0, 1, ty1]])
    M_trans2 = np.float32([[1, 0, tx2], [0, 1, ty2]])

    im1 = cv2.warpAffine(im1, M_trans1, (w1, h1), borderMode=cv2.BORDER_CONSTANT, borderValue=mean1)
    im2 = cv2.warpAffine(im2, M_trans2, (w2, h2), borderMode=cv2.BORDER_CONSTANT, borderValue=mean2)

    x1, y1 = x1 + tx1, y1 + ty1
    x2, y2 = x2 + tx2, y2 + ty2

    len1 = np.sqrt((y1[1] - y1[0])**2 + (x1[1] - x1[0])**2)
    len2 = np.sqrt((y2[1] - y2[0])**2 + (x2[1] - x2[0])**2)
    dscale = len2 / len1

    if dscale < 1:
        im1 = cv2.resize(im1, (0, 0), fx=dscale, fy=dscale, interpolation=cv2.INTER_LINEAR)
    else:
        im2 = cv2.resize(im2, (0, 0), fx=1.0 / dscale, fy=1.0 / dscale, interpolation=cv2.INTER_LINEAR)

    h1, w1 = im1.shape[:2]
    h2, w2 = im2.shape[:2]

    min_h = int(min(h1, h2) * 0.85)
    min_w = int(min(w1, w2) * 0.85)

    cy1_c, cx1_c = h1 // 2, w1 // 2
    cy2_c, cx2_c = h2 // 2, w2 // 2

    im1_cropped = im1[cy1_c - min_h // 2: cy1_c + min_h // 2, cx1_c - min_w // 2: cx1_c + min_w // 2]
    im2_cropped = im2[cy2_c - min_h // 2: cy2_c + min_h // 2, cx2_c - min_w // 2: cx2_c + min_w // 2]

    return im1_cropped, im2_cropped