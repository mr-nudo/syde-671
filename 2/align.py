import os
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.transform import rescale
from skimage.filters import sobel


def load_and_split(image_path):
    img = img_as_float(io.imread(image_path))
    if img.ndim == 3:
        img = img[:, :, 0]
    height = img.shape[0] // 3
    return img[0:height, :], img[height : 2 * height, :], img[2 * height : 3 * height, :]


def compute_metric(im1, im2, metric="ncc", border_crop=0.15):
    h, w = im1.shape
    crop_h, crop_w = int(h * border_crop), int(w * border_crop)
    i1 = im1[crop_h : h - crop_h, crop_w : w - crop_w]
    i2 = im2[crop_h : h - crop_h, crop_w : w - crop_w]

    if metric == "l2":
        return -np.sum((i1 - i2) ** 2)
    elif metric == "ncc":
        norm1, norm2 = np.linalg.norm(i1), np.linalg.norm(i2)
        return np.sum((i1 / norm1) * (i2 / norm2)) if norm1 > 0 and norm2 > 0 else -np.inf
    raise ValueError(f"Unsupported metric: {metric}")


def align_single_scale(ref, target, search_window=(-15, 15), metric="ncc"):
    best_score, best_shift = -np.inf, (0, 0)
    for dy in range(search_window[0], search_window[1] + 1):
        for dx in range(search_window[0], search_window[1] + 1):
            score = compute_metric(ref, np.roll(target, (dy, dx), axis=(0, 1)), metric=metric)
            if score > best_score:
                best_score, best_shift = score, (dx, dy)
    return best_shift


def align_pyramid_robust(ref, target, max_depth=4, metric="ncc", use_edges=True):
    h, w = ref.shape
    ch, cw = int(h * 0.10), int(w * 0.10)

    r_clean = ref[ch : h - ch, cw : w - cw]
    t_clean = target[ch : h - ch, cw : w - cw]

    r_feat = sobel(r_clean) if use_edges else r_clean
    t_feat = sobel(t_clean) if use_edges else t_clean

    def _recursive_align(r, t, depth):
        if depth == 0 or min(r.shape) < 128:
            return align_single_scale(r, t, search_window=(-15, 15), metric=metric)

        r_small = rescale(r, 0.5, anti_aliasing=True)
        t_small = rescale(t, 0.5, anti_aliasing=True)

        coarse_shift = _recursive_align(r_small, t_small, depth - 1)
        scaled_shift = (coarse_shift[0] * 2, coarse_shift[1] * 2)

        shifted_t = np.roll(t, (scaled_shift[1], scaled_shift[0]), axis=(0, 1))
        fine_shift = align_single_scale(r, shifted_t, search_window=(-7, 7), metric=metric)

        return (scaled_shift[0] + fine_shift[0], scaled_shift[1] + fine_shift[1])

    return _recursive_align(r_feat, t_feat, max_depth)


def apply_shift(img, shift):
    return np.roll(img, (shift[1], shift[0]), axis=(0, 1))


def format_time(seconds):
    mins, secs = divmod(seconds, 60)
    if mins > 0:
        return f"{int(mins)}m {int(secs)}s"
    return f"{secs:.2f}s"


def generate_intermediate_artifacts(sample_path, output_dir):
    b, g, r = load_and_split(sample_path)

    plt.imsave(os.path.join(output_dir, "intermediate_01_unaligned.jpg"), np.dstack((r, g, b)))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    channels = [("Blue Channel (B)", b), ("Green Channel (G)", g), ("Red Channel (R)", r)]

    for col, (title, ch) in enumerate(channels):
        axes[col].imshow(ch, cmap="gray")
        axes[col].set_title(title)
        axes[col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "intermediate_02_split_channels.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    input_dir = "data"
    output_dir = "media"
    os.makedirs(output_dir, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Error: No images found in '{input_dir}/'.")
        exit()

    cathedral_path = next(
        (p for p in image_paths if "cathedral" in os.path.basename(p).lower()), sorted(image_paths)[0]
    )
    print(f"Generating intermediate process artifacts using: {os.path.basename(cathedral_path)}")
    generate_intermediate_artifacts(cathedral_path, output_dir)

    results = []

    for path in sorted(image_paths):
        filename = os.path.basename(path)
        name, ext = os.path.splitext(filename)
        is_high_res = ext.lower() in (".tif", ".tiff")
        category = "LoC Custom" if filename.startswith("loc-") else "Provided"

        print(f"Processing ({category}): {filename}...")
        img_start_time = time.time()
        b, g, r = load_and_split(path)

        # 1. Single-Scale Execution & Timing
        if not is_high_res:
            t_single_start = time.time()
            g_single = align_single_scale(b, g, search_window=(-15, 15), metric="ncc")
            r_single = align_single_scale(b, r, search_window=(-15, 15), metric="ncc")
            rgb_single = np.dstack((apply_shift(r, r_single), apply_shift(g, g_single), b))
            plt.imsave(os.path.join(output_dir, f"{name}_single.jpg"), rgb_single)
            single_time_str = format_time(time.time() - t_single_start)
        else:
            g_single, r_single = ("N/A", "N/A")
            single_time_str = "N/A"

        # 2. Pyramid Execution & Timing
        t_pyr_start = time.time()
        g_pyr = align_pyramid_robust(b, g, max_depth=4, metric="ncc", use_edges=True)
        r_pyr = align_pyramid_robust(b, r, max_depth=4, metric="ncc", use_edges=True)
        rgb_pyr = np.dstack((apply_shift(r, r_pyr), apply_shift(g, g_pyr), b))
        plt.imsave(os.path.join(output_dir, f"{name}_pyramid.jpg"), rgb_pyr)
        pyr_time_str = format_time(time.time() - t_pyr_start)

        # 3. Total Image Execution Time
        total_time_str = format_time(time.time() - img_start_time)

        print(f"  -> Single-scale time : {single_time_str}")
        print(f"  -> Pyramid time      : {pyr_time_str}")
        print(f"  -> Total image time  : {total_time_str}\n")

        results.append({
            "name": filename,
            "category": category,
            "g_single": str(g_single),
            "r_single": str(r_single),
            "single_time": single_time_str,
            "g_pyr": str(g_pyr),
            "r_pyr": str(r_pyr),
            "pyr_time": pyr_time_str,
            "total_time": total_time_str
        })

    print("\n" + "=" * 135)
    print(f"{'Filename':<20} | {'Category':<10} | {'Single G,R (x,y)':<18} | {'Single Time':<12} | {'Pyramid G,R (x,y)':<18} | {'Pyr Time':<10} | {'Total Time':<10}")
    print("-" * 135)
    for r in results:
        single_shifts = f"{r['g_single']} / {r['r_single']}"
        pyr_shifts = f"{r['g_pyr']} / {r['r_pyr']}"
        print(f"{r['name']:<20} | {r['category']:<10} | {single_shifts:<18} | {r['single_time']:<12} | {pyr_shifts:<18} | {r['pyr_time']:<10} | {r['total_time']:<10}")
    print("=" * 135)