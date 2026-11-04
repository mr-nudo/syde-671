import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mosaic_core import (
    computeH,
    warpImageNearestNeighbor,
    warpImageBilinear,
    get_bounding_box
)

def interactive_click_points(img, num_pts, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    plt.tight_layout()
    plt.show(block=False)
    
    print(f"\n-> Click {num_pts} points on the image window.")
    pts = []
    for i in range(num_pts):
        plt.sca(ax)
        p = plt.ginput(1, timeout=-1)
        if not p:
            break
        pts.append(p[0])
        ax.plot(p[0][0], p[0][1], 'ro', markersize=6)
        ax.text(p[0][0]+5, p[0][1]+5, str(i+1), color='yellow', fontsize=12, weight='bold')
        fig.canvas.draw()
        
    plt.close(fig)
    return np.array(pts, dtype=np.float32)

def interactive_click_pairs_alternating(img1, img2, num_pts=6):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Image 1: Reference (Left)")
    
    # Updated title as requested
    axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Image 2: Target (Right)")
    
    # On-screen instruction banner displayed directly on the window
    fig.suptitle("Instructions: Click Image 1 first, then click corresponding point on Image 2", 
                 fontsize=13, fontweight='bold', color='darkblue')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Make room for suptitle banner
    plt.show(block=False)
    
    pts1, pts2 = [], []
    print(f"\n[Interactive Point Selection - Alternating Mode]")
    print(f"-> Select {num_pts} pairs one by one. Click corresponding structural features!")
    
    for i in range(num_pts):
        print(f"\n--- Selecting Pair {i+1} of {num_pts} ---")
        
        # 1. Click on Image 1
        plt.sca(axes[0])
        print("-> Click point on Image 1...")
        p1 = plt.ginput(1, timeout=-1)
        if not p1: break
        pts1.append(p1[0])
        axes[0].plot(p1[0][0], p1[0][1], 'ro', markersize=6)
        axes[0].text(p1[0][0]+5, p1[0][1]+5, str(i+1), color='yellow', fontsize=12, weight='bold')
        fig.canvas.draw()
        
        # 2. Click on Image 2
        plt.sca(axes[1])
        print("-> Click the matching point on Image 2...")
        p2 = plt.ginput(1, timeout=-1)
        if not p2: break
        pts2.append(p2[0])
        axes[1].plot(p2[0][0], p2[0][1], 'ro', markersize=6)
        axes[1].text(p2[0][0]+5, p2[0][1]+5, str(i+1), color='yellow', fontsize=12, weight='bold')
        fig.canvas.draw()
        
    plt.close(fig)
    return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)

def run_rectification():
    print("--- Running Rectification (Part A.3) ---")
    data_dir = "data"
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)
    
    rect_path = os.path.join(data_dir, "rectification_test.jpg")
    if not os.path.exists(rect_path):
        raise FileNotFoundError(f"Could not find {rect_path}.")
        
    im = cv2.imread(rect_path)
    
    # Updated text for clarity
    rect_title = "Click 4 corners (Top Left -> Top Right -> Bottom Right -> Bottom Left)"
    print(f"\n[Interactive Rectification]\n-> {rect_title}")
    
    src_pts = interactive_click_points(im, 4, rect_title)
    
    box_size = 400
    dst_pts = np.array([[0, 0], [box_size, 0], [box_size, box_size], [0, box_size]], dtype=np.float32)
    
    H = computeH(src_pts, dst_pts)
    nn_warped, _, _ = warpImageNearestNeighbor(im, H)
    bil_warped, _, _ = warpImageBilinear(im, H)
    
    cv2.imwrite(os.path.join(media_dir, "rectified_nn.jpg"), nn_warped)
    cv2.imwrite(os.path.join(media_dir, "rectified_bilinear.jpg"), bil_warped)
    print("-> Saved rectification outputs.")

def run_manual_mosaics():
    print("\n--- Running Manual Mosaics (Part A.4 - Outdoor Set) ---")
    data_dir = "data"
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)
    
    img_left_orig = cv2.imread(os.path.join(data_dir, "outdoor_left.jpg"))
    img_center_orig = cv2.imread(os.path.join(data_dir, "outdoor_center.jpg"))
    
    if img_left_orig is None or img_center_orig is None:
        print("-> Error: Could not load outdoor_left.jpg or outdoor_center.jpg from data/")
        return
        
    # --- 1. SAFEGUARD: Downscale for smooth, lag-free GUI clicking ---
    WORK_SCALE = 0.5
    img_left_small = cv2.resize(img_left_orig, (0, 0), fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    img_center_small = cv2.resize(img_center_orig, (0, 0), fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    
    print("-> Please select corresponding point pairs in the interactive window...")
    pts_left_small, pts_center_small = interactive_click_pairs_alternating(img_left_small, img_center_small, num_pts=6)
    if len(pts_left_small) < 4:
        print("-> Error: Not enough points selected.")
        return
        
    # --- 2. SCALE POINTS BACK UP TO HIGH-RES COORDINATES ---
    pts_left = pts_left_small / WORK_SCALE
    pts_center = pts_center_small / WORK_SCALE
    
    H = computeH(pts_left, pts_center)
    print("-> Computed Homography matrix H successfully!")
    
    min_x, max_x, min_y, max_y = get_bounding_box(img_left_orig, H)
    print(f"-> Predicted Bounding Box: x[{min_x}, {max_x}], y[{min_y}, {max_y}]")
    
    if (max_x - min_x > 15000) or (max_y - min_y > 15000):
        print("-> ABORT: Bounding box is excessively large! Check your clicked point order.")
        return
        
    warped_left, mask_left, (min_x, min_y) = warpImageBilinear(img_left_orig, H)
    
    h_c, w_c = img_center_orig.shape[:2]
    
    # Global canvas bounds
    all_min_x = min(0, min_x)
    all_max_x = max(w_c, min_x + warped_left.shape[1])
    all_min_y = min(0, min_y)
    all_max_y = max(h_c, min_y + warped_left.shape[0])
    
    canvas_w = all_max_x - all_min_x + 1
    canvas_h = all_max_y - all_min_y + 1
    
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight_sum = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)
    
    # Offsets
    c_x_offset = -all_min_x
    c_y_offset = -all_min_y
    
    w_y_offset = min_y - all_min_y
    w_x_offset = min_x - all_min_x
    
    # Helper to add an image with a smooth feather mask
    def add_to_canvas(img, mask, y_off, x_off):
        h, w = img.shape[:2]
        c_y1 = max(0, y_off)
        c_x1 = max(0, x_off)
        c_y2 = min(canvas_h, y_off + h)
        c_x2 = min(canvas_w, x_off + w)
        
        i_y1 = max(0, -y_off)
        i_x1 = max(0, -x_off)
        i_y2 = i_y1 + (c_y2 - c_y1)
        i_x2 = i_x1 + (c_x2 - c_x1)
        
        if c_y2 > c_y1 and c_x2 > c_x1:
            sub_img = img[i_y1:i_y2, i_x1:i_x2].astype(np.float32)
            sub_mask = (mask[i_y1:i_y2, i_x1:i_x2] > 0).astype(np.float32)
            
            if np.sum(sub_mask) > 0:
                # Create a smooth distance feather mask for this specific image chunk
                dist = cv2.distanceTransform((sub_mask * 255).astype(np.uint8), cv2.DIST_L2, 3)
                if dist.max() > 0:
                    dist = dist / dist.max()  # Normalize to [0, 1]
                
                # Ensure minimum weight so pixels never vanish
                alpha = np.expand_dims(np.maximum(dist, 0.05) * sub_mask, axis=-1)
                
                canvas[c_y1:c_y2, c_x1:c_x2] += sub_img * alpha
                weight_sum[c_y1:c_y2, c_x1:c_x2] += alpha

    # --- 3. ADD CENTER IMAGE ---
    center_mask = np.ones((h_c, w_c), dtype=np.uint8) * 255
    add_to_canvas(img_center_orig, center_mask, c_y_offset, c_x_offset)
    
    # --- 4. ADD WARPED LEFT IMAGE ---
    add_to_canvas(warped_left, mask_left, w_y_offset, w_x_offset)
    
    # Normalize by accumulated weights to achieve seamless blending
    weight_sum[weight_sum == 0] = 1.0
    final_canvas = canvas / weight_sum
    final_output = np.clip(final_canvas, 0, 255).astype(np.uint8)
    
    output_path = os.path.join(media_dir, "manual_mosaic_outdoor.jpg")
    cv2.imwrite(output_path, final_output)
    print(f"-> Saved properly blended manual mosaic output to {output_path}")

if __name__ == "__main__":
    run_rectification()
    run_manual_mosaics()