"""
align_core.py
Core image alignment algorithms and metrics for Prokudin-Gorskii glass plate colorization.
"""

import numpy as np
from skimage import io, img_as_float
from skimage.filters import sobel
from skimage.transform import rescale


def load_and_split(image_path: str):
    """Loads an image, normalizes to float [0, 1], and splits vertically into B, G, R channels."""
    img = img_as_float(io.imread(image_path))
    if img.ndim == 3:
        img = img[:, :, 0]
    height = img.shape[0] // 3
    return img[0:height, :], img[height : 2 * height, :], img[2 * height : 3 * height, :]


def compute_metric(im1: np.ndarray, im2: np.ndarray, metric: str = "ncc", border_crop: float = 0.15) -> float:
    """Computes similarity score between two image channels using L2 or NCC with border cropping."""
    h, w = im1.shape
    crop_h, crop_w = int(h * border_crop), int(w * border_crop)
    i1 = im1[crop_h : h - crop_h, crop_w : w - crop_w]
    i2 = im2[crop_h : h - crop_h, crop_w : w - crop_w]

    if metric == "l2":
        return -float(np.sum((i1 - i2) ** 2))
    elif metric == "ncc":
        norm1, norm2 = np.linalg.norm(i1), np.linalg.norm(i2)
        if norm1 > 0 and norm2 > 0:
            return float(np.sum((i1 / norm1) * (i2 / norm2)))
        return -np.inf
    raise ValueError(f"Unsupported metric: {metric}")


def apply_shift(img: np.ndarray, shift: tuple[int, int]) -> np.ndarray:
    """Applies a 2D integer displacement shift (dx, dy) to an image array."""
    return np.roll(img, (shift[1], shift[0]), axis=(0, 1))


def align_single_scale(
    ref: np.ndarray,
    target: np.ndarray,
    search_window: tuple[int, int] = (-15, 15),
    metric: str = "ncc",
) -> tuple[int, int]:
    """Exhaustive search alignment over specified search window [-radius, radius]."""
    best_score, best_shift = -np.inf, (0, 0)
    min_w, max_w = search_window
    for dy in range(min_w, max_w + 1):
        for dx in range(min_w, max_w + 1):
            shifted_target = apply_shift(target, (dx, dy))
            score = compute_metric(ref, shifted_target, metric=metric)
            if score > best_score:
                best_score, best_shift = score, (dx, dy)
    return best_shift


def align_pyramid_robust(
    ref: np.ndarray,
    target: np.ndarray,
    max_depth: int = 4,
    metric: str = "ncc",
    use_edges: bool = True,
) -> tuple[int, int]:
    """Coarse-to-fine multi-scale image pyramid alignment using recursive downsampling."""
    h, w = ref.shape
    ch, cw = int(h * 0.10), int(w * 0.10)

    r_clean = ref[ch : h - ch, cw : w - cw]
    t_clean = target[ch : h - ch, cw : w - cw]

    r_feat = sobel(r_clean) if use_edges else r_clean
    t_feat = sobel(t_clean) if use_edges else t_clean

    def _recursive_align(r: np.ndarray, t: np.ndarray, depth: int) -> tuple[int, int]:
        if depth == 0 or min(r.shape) < 128:
            return align_single_scale(r, t, search_window=(-15, 15), metric=metric)

        r_small = rescale(r, 0.5, anti_aliasing=True)
        t_small = rescale(t, 0.5, anti_aliasing=True)

        coarse_shift = _recursive_align(r_small, t_small, depth - 1)
        scaled_shift = (coarse_shift[0] * 2, coarse_shift[1] * 2)

        shifted_t = apply_shift(t, scaled_shift)
        fine_shift = align_single_scale(r, shifted_t, search_window=(-7, 7), metric=metric)

        return (scaled_shift[0] + fine_shift[0], scaled_shift[1] + fine_shift[1])

    return _recursive_align(r_feat, t_feat, max_depth)