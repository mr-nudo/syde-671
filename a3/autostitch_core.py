import numpy as np
import cv2
from mosaic_core import computeH


def detect_harris_corners(im, k=0.04, threshold_factor=0.01, scale_factor=0.25):
    """
    Detects Harris corners in a grayscale image.
    Optionally downscales the image first for massive speedups on high-res photos.
    Returns (N, 2) array of (x, y) coordinates scaled back to original resolution.
    """
    if im.ndim == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im.copy()
        
    # Optional downscaling for speed
    if scale_factor < 1.0:
        h_orig, w_orig = gray.shape[:2]
        gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        
    gray = gray.astype(np.float32) / 255.0
    
    # Compute image gradients
    Ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy
    
    # Apply Gaussian smoothing to derivative products
    Sx2 = cv2.GaussianBlur(Ixx, (3, 3), 1.0)
    Sy2 = cv2.GaussianBlur(Iyy, (3, 3), 1.0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.0)
    
    # Compute Harris response R = det(M) - k * trace(M)^2
    detM = (Sx2 * Sy2) - (Sxy ** 2)
    traceM = Sx2 + Sy2
    R = detM - k * (traceM ** 2)
    
    # Thresholding & local maxima extraction
    threshold = threshold_factor * np.max(R)
    r_mask = (R > threshold)
    local_max = cv2.dilate(R, np.ones((3, 3))) == R
    corner_mask = r_mask & local_max
    
    y_idxs, x_idxs = np.where(corner_mask)
    strengths = R[y_idxs, x_idxs]
    
    points = np.stack([x_idxs, y_idxs], axis=-1)
    
    # Rescale points back to original image coordinates if downscaled
    if scale_factor < 1.0 and len(points) > 0:
        points = points / scale_factor
        
    return points.astype(np.float32), strengths


def adaptive_non_maximal_suppression(pts, strengths, num_to_keep=500, c_robust=0.9):
    """
    Implements Adaptive Non-Maximal Suppression (ANMS) from Brown et al.
    Includes a corner-capping safeguard to prevent O(N^2) loop freezes.
    """
    if len(pts) <= num_to_keep:
        return pts
        
    # Safeguard: Keep only the top 1500 strongest corners to prevent N^2 explosion
    if len(pts) > 1500:
        top_indices = np.argsort(strengths)[::-1][:1500]
        pts = pts[top_indices]
        strengths = strengths[top_indices]
        
    N = len(pts)
    radii = np.inf * np.ones(N, dtype=np.float64)
    
    # Sort points by strength descending
    sorted_indices = np.argsort(strengths)[::-1]
    pts_sorted = pts[sorted_indices]
    strengths_sorted = strengths[sorted_indices]
    
    for i in range(N):
        xi, yi = pts_sorted[i]
        si = strengths_sorted[i]
        for j in range(i):
            sj = strengths_sorted[j]
            if sj > c_robust * si:
                xj, yj = pts_sorted[j]
                dist = (xi - xj) ** 2 + (yi - yj) ** 2
                if dist < radii[i]:
                    radii[i] = dist
                    
    # Sort by radius descending
    radius_order = np.argsort(radii)[::-1]
    selected_indices = sorted_indices[radius_order[:num_to_keep]]
    return pts[selected_indices]


def extract_descriptors(im, pts):
    """
    Extracts bias- and gain-normalized 8x8 feature descriptors sampled from 40x40 patches.
    """
    if im.ndim == 3:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        gray = im.copy()
        
    gray = gray.astype(np.float32) / 255.0
    h, w = gray.shape
    
    descriptors = []
    valid_pts = []
    
    for pt in pts:
        x, y = int(pt[0]), int(pt[1])
        
        if x < 20 or x >= w - 20 or y < 20 or y >= h - 20:
            continue
            
        patch_40 = gray[y - 20:y + 20, x - 20:x + 20]
        blurred = cv2.GaussianBlur(patch_40, (5, 5), 1.0)
        patch_8 = cv2.resize(blurred, (8, 8), interpolation=cv2.INTER_AREA)
        
        mean = np.mean(patch_8)
        std = np.std(patch_8)
        if std > 1e-5:
            norm_patch = (patch_8 - mean) / std
        else:
            norm_patch = patch_8 - mean
            
        descriptors.append(norm_patch.flatten())
        valid_pts.append(pt)
        
    return np.array(descriptors), np.array(valid_pts)


def match_descriptors(desc1, desc2, pts1, pts2, ratio_thresh=0.7):
    """
    Matches feature descriptors using Lowe's Ratio Test.
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2)), np.empty((0, 2))
        
    dists = np.sqrt(
        np.sum(desc1**2, axis=1, keepdims=True) +
        np.sum(desc2**2, axis=1) -
        2.0 * np.dot(desc1, desc2.T)
    )
    
    matches1, matches2 = [], []
    for i in range(dists.shape[0]):
        sorted_indices = np.argsort(dists[i])
        if len(sorted_indices) < 2:
            continue
        nn1 = sorted_indices[0]
        nn2 = sorted_indices[1]
        
        if dists[i, nn1] < ratio_thresh * dists[i, nn2]:
            matches1.append(pts1[i])
            matches2.append(pts2[nn1])
            
    return np.array(matches1), np.array(matches2)


def ransac_homography(pts1, pts2, num_iterations=1000, threshold=3.0):
    """
    Computes a robust homography matrix using 4-point RANSAC from scratch.
    """
    num_pts = len(pts1)
    if num_pts < 4:
        raise ValueError("At least 4 point correspondences are needed for RANSAC.")
        
    best_H = None
    max_inliers = -1
    best_inlier_mask = None
    
    pts1_homog = np.hstack([pts1, np.ones((num_pts, 1))])
    
    for _ in range(num_iterations):
        indices = np.random.choice(num_pts, 4, replace=False)
        p1_sample = pts1[indices]
        p2_sample = pts2[indices]
        
        try:
            H_candidate = computeH(p1_sample, p2_sample)
        except np.linalg.LinAlgError:
            continue
            
        projected = (H_candidate @ pts1_homog.T).T
        projected /= projected[:, 2:]
        projected_pts = projected[:, :2]
        
        errors = np.sqrt(np.sum((pts2 - projected_pts) ** 2, axis=1))
        inlier_mask = errors < threshold
        num_inliers = np.sum(inlier_mask)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inlier_mask
            best_H = H_candidate
            
    if max_inliers >= 4:
        best_H = computeH(pts1[best_inlier_mask], pts2[best_inlier_mask])
        
    return best_H, best_inlier_mask