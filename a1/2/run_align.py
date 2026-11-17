"""
run_align.py
Batch processing, timing, artifact generation, and reporting pipeline.
"""

import glob
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# Import core modular functions
from align_core import (
    align_pyramid_robust,
    align_single_scale,
    apply_shift,
    load_and_split,
)


def format_time(seconds: float) -> str:
    """Formats execution time in seconds or minutes/seconds."""
    mins, secs = divmod(seconds, 60)
    if mins > 0:
        return f"{int(mins)}m {int(secs)}s"
    return f"{secs:.2f}s"


def generate_intermediate_artifacts(sample_path: str, output_dir: str) -> None:
    """Saves raw unaligned composite and split channel figures for post documentation."""
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


def print_summary_table(results: list[dict]) -> None:
    """Prints the final formatted benchmark summary table to console."""
    divider = "=" * 135
    header = (
        f"{'Filename':<20} | {'Category':<10} | {'Single G,R (x,y)':<18} | "
        f"{'Single Time':<12} | {'Pyramid G,R (x,y)':<18} | {'Pyr Time':<10} | {'Total Time':<10}"
    )
    print("\n" + divider)
    print(header)
    print("-" * 135)

    for r in results:
        single_shifts = f"{r['g_single']} / {r['r_single']}"
        pyr_shifts = f"{r['g_pyr']} / {r['r_pyr']}"
        print(
            f"{r['name']:<20} | {r['category']:<10} | {single_shifts:<18} | "
            f"{r['single_time']:<12} | {pyr_shifts:<18} | {r['pyr_time']:<10} | {r['total_time']:<10}"
        )
    print(divider)


def process_batch(input_dir: str = "data", output_dir: str = "media") -> None:
    """Runs single-scale and pyramid alignment across all dataset images."""
    os.makedirs(output_dir, exist_ok=True)

    extensions = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff")
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"Error: No images found in '{input_dir}/'.")
        return

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

        # 1. Single-Scale Alignment & Timing
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

        # 2. Multi-Scale Pyramid Alignment & Timing
        t_pyr_start = time.time()
        g_pyr = align_pyramid_robust(b, g, max_depth=4, metric="ncc", use_edges=True)
        r_pyr = align_pyramid_robust(b, r, max_depth=4, metric="ncc", use_edges=True)
        rgb_pyr = np.dstack((apply_shift(r, r_pyr), apply_shift(g, g_pyr), b))
        plt.imsave(os.path.join(output_dir, f"{name}_pyramid.jpg"), rgb_pyr)
        pyr_time_str = format_time(time.time() - t_pyr_start)

        # 3. Total Timing Accumulation
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
            "total_time": total_time_str,
        })

    print_summary_table(results)


if __name__ == "__main__":
    process_batch(input_dir="data", output_dir="media")