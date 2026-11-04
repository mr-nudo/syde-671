import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from autostitch_core import (
    detect_harris_corners,
    adaptive_non_maximal_suppression,
    extract_descriptors,
    match_descriptors,
    ransac_homography
)
from mosaic_core import warpImageBilinear

def run_autostitch_pipeline():
    print("--- Running 3-Image Autostitch Pipeline (Part 2 - Indoor Set) ---")
    data_dir = "data"
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)
    
    # Load all 3 indoor images
    img_left = cv2.imread(os.path.join(data_dir, "indoor_left.jpg"))
    img_center = cv2.imread(os.path.join(data_dir, "indoor_center.jpg"))
    img_right = cv2.imread(os.path.join(data_dir, "indoor_right.jpg"))
    
    if img_left is None or img_center is None or img_right is None:
        print("-> Error: One or more indoor images missing in data/ folder.")
        return
        
    # --- SAFEGUARD: Downscale working images to prevent RAM crashes ---
    WORK_SCALE = 0.5 
    img_left = cv2.resize(img_left, (0, 0), fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    img_center = cv2.resize(img_center, (0, 0), fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
    img_right = cv2.resize(img_right, (0, 0), fx=WORK_SCALE, fy=WORK_SCALE, interpolation=cv2.INTER_AREA)
        
    # Helper function for matching pair pipeline & saving diagnostic figures
    def process_pair(im1, im2, pair_name, save_diagnostics=False):
        pts1, str1 = detect_harris_corners(im1, scale_factor=0.25)
        pts2, str2 = detect_harris_corners(im2, scale_factor=0.25)
        
        anms1 = adaptive_non_maximal_suppression(pts1, str1, num_to_keep=300)
        anms2 = adaptive_non_maximal_suppression(pts2, str2, num_to_keep=300)
        
        if save_diagnostics:
            # 1. Raw Harris Corners (Without ANMS)
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
            if len(pts1) > 0:
                # pts1 format handling (assuming [y, x] or [x, y])
                plt.scatter(pts1[:, 1], pts1[:, 0], s=2, c='red', alpha=0.6)
            plt.title("Raw Harris Corners (Without ANMS)")
            plt.axis('off')
            plt.savefig(os.path.join(media_dir, "harris_raw.jpg"), bbox_inches='tight', dpi=200)
            plt.close()

            # 2. ANMS Suppressed Corners
            plt.figure(figsize=(8, 6))
            plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))
            if len(anms1) > 0:
                plt.scatter(anms1[:, 1], anms1[:, 0], s=10, c='cyan', marker='x')
            plt.title("Harris Corners with ANMS Suppression")
            plt.axis('off')
            plt.savefig(os.path.join(media_dir, "harris_anms.jpg"), bbox_inches='tight', dpi=200)
            plt.close()

        desc1, v1 = extract_descriptors(im1, anms1)
        desc2, v2 = extract_descriptors(im2, anms2)
        
        m1, m2 = match_descriptors(desc1, desc2, v1, v2)
        print(f"[{pair_name}] Found {len(m1)} raw matches after ratio test.")
        
        if save_diagnostics and len(m1) > 0:
            h1, w1 = im1.shape[:2]
            h2, w2 = im2.shape[:2]
            match_vis = np.hstack((im1, im2))
            
            plt.figure(figsize=(12, 6))
            plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB))
            for pt1, pt2 in zip(m1[:50], m2[:50]): # Plot first 50 matches for clarity
                plt.plot([pt1[1], pt2[1] + w1], [pt1[0], pt2[0]], color='yellow', linewidth=0.8, alpha=0.7)
                plt.scatter([pt1[1], pt2[1] + w1], [pt1[0], pt2[0]], s=15, c='lime', marker='o')
            plt.title("Lowe's Ratio Feature Matches Between Image Pair")
            plt.axis('off')
            plt.savefig(os.path.join(media_dir, "feature_matches.jpg"), bbox_inches='tight', dpi=200)
            plt.close()

        if len(m1) >= 4:
            H, inliers = ransac_homography(m1, m2, num_iterations=1000, threshold=3.0)
            print(f"[{pair_name}] RANSAC kept {np.sum(inliers)} inliers.")
            return H, inliers
        return None, None

    # 1. Compute H: Left -> Center (with diagnostic figure saving enabled)
    H_left_center, _ = process_pair(img_left, img_center, "Left->Center", save_diagnostics=True)
    
    # 2. Compute H: Right -> Center
    H_right_center, _ = process_pair(img_right, img_center, "Right->Center", save_diagnostics=False)
    
    if H_left_center is None or H_right_center is None:
        print("-> Error: Homography estimation failed for one or more pairs.")
        return

    print("-> Homographies computed successfully. Warping onto multi-image canvas...")

    # 3. Warp Left and Right images into Center's reference frame
    warped_left, mask_left, (min_x_l, min_y_l) = warpImageBilinear(img_left, H_left_center)
    warped_right, mask_right, (min_x_r, min_y_r) = warpImageBilinear(img_right, H_right_center)
    
    h_c, w_c = img_center.shape[:2]
    
    # Determine global canvas bounds encompassing all three images
    all_min_x = min(0, min_x_l, min_x_r)
    all_max_x = max(w_c, min_x_l + warped_left.shape[1], min_x_r + warped_right.shape[1])
    all_min_y = min(0, min_y_l, min_y_r)
    all_max_y = max(h_c, min_y_l + warped_left.shape[0], min_y_r + warped_right.shape[0])
    
    canvas_w = all_max_x - all_min_x + 1
    canvas_h = all_max_y - all_min_y + 1
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Offsets for global origin alignment
    c_x_offset = -all_min_x
    c_y_offset = -all_min_y
    
    # Place Center image
    canvas[c_y_offset:c_y_offset+h_c, c_x_offset:c_x_offset+w_c] = img_center
    
    # Vectorized canvas pasting helper
    def paste_warped(canvas, warped, mask, min_x, min_y, all_min_x, all_min_y):
        y_off = min_y - all_min_y
        x_off = min_x - all_min_x
        h_w, w_w = warped.shape[:2]
        
        c_y1 = max(0, y_off)
        c_x1 = max(0, x_off)
        c_y2 = min(canvas.shape[0], y_off + h_w)
        c_x2 = min(canvas.shape[1], x_off + w_w)
        
        w_y1 = max(0, -y_off)
        w_x1 = max(0, -x_off)
        w_y2 = w_y1 + (c_y2 - c_y1)
        w_x2 = w_x1 + (c_x2 - c_x1)
        
        if c_y2 > c_y1 and c_x2 > c_x1:
            sub_mask = mask[w_y1:w_y2, w_x1:w_x2] > 0
            canvas_region = canvas[c_y1:c_y2, c_x1:c_x2]
            warped_region = warped[w_y1:w_y2, w_x1:w_x2]
            canvas_region[sub_mask] = warped_region[sub_mask]

    # Place Warped Left and Right images onto canvas
    paste_warped(canvas, warped_left, mask_left, min_x_l, min_y_l, all_min_x, all_min_y)
    paste_warped(canvas, warped_right, mask_right, min_x_r, min_y_r, all_min_x, all_min_y)
            
    output_path = os.path.join(media_dir, "autostitch_indoor.jpg")
    cv2.imwrite(output_path, canvas)
    print(f"-> Saved 3-image autostitch mosaic output to {output_path}")
    print("-> All diagnostic figures (harris_raw.jpg, harris_anms.jpg, feature_matches.jpg) successfully exported to media/!")

if __name__ == "__main__":
    run_autostitch_pipeline()