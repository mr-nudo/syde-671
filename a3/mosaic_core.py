import numpy as np
import cv2


def computeH(im1_pts, im2_pts):
    """
    Computes the 3x3 homography matrix H such that im2_pts ~ H * im1_pts.
    
    Parameters:
        im1_pts: (N, 2) array of (x, y) coordinates in Image 1.
        im2_pts: (N, 2) array of (x', y') coordinates in Image 2.
        
    Returns:
        H: 3x3 Homography matrix normalized so H[2, 2] = 1.0.
    """
    im1_pts = np.asarray(im1_pts, dtype=np.float64)
    im2_pts = np.asarray(im2_pts, dtype=np.float64)
    
    num_pts = im1_pts.shape[0]
    if num_pts < 4:
        raise ValueError("At least 4 point correspondences are required to compute H.")
        
    A = []
    for i in range(num_pts):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i]
        
        # Row 1: [x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp]
        # Row 2: [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp]
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -xp * x, -xp * y, -xp])
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -yp * x, -yp * y, -yp])
        
    A = np.array(A, dtype=np.float64)
    
    # Solve Ah = 0 using SVD (Singular Value Decomposition)
    _, _, Vh = np.linalg.svd(A)
    h = Vh[-1, :]  # Right-singular vector for the smallest singular value
    
    H = h.reshape((3, 3))
    H = H / H[2, 2]  # Normalize so bottom-right element is 1.0
    return H


def get_bounding_box(im, H):
    """
    Predicts the output canvas bounding box when image 'im' is transformed by H.
    """
    h, w = im.shape[:2]
    # Corners in (x, y) format: top-left, top-right, bottom-right, bottom-left
    corners = np.array([
        [0, 0, 1.0],
        [w - 1, 0, 1.0],
        [w - 1, h - 1, 1.0],
        [0, h - 1, 1.0]
    ]).T
    
    warped_corners = H @ corners
    warped_corners /= warped_corners[2, :]  # Normalize by homogeneous coordinate
    
    x_coords = warped_corners[0, :]
    y_coords = warped_corners[1, :]
    
    min_x, max_x = np.floor(np.min(x_coords)).astype(int), np.ceil(np.max(x_coords)).astype(int)
    min_y, max_y = np.floor(np.min(y_coords)).astype(int), np.ceil(np.max(y_coords)).astype(int)
    
    return min_x, max_x, min_y, max_y


def warpImageNearestNeighbor(im, H):
    """
    Warps image 'im' using Homography H and Nearest-Neighbor interpolation (Inverse Warping).
    """
    min_x, max_x, min_y, max_y = get_bounding_box(im, H)
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    
    # Create target canvas grid coordinates
    grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    ones = np.ones_like(grid_x)
    target_pts = np.stack([grid_x.ravel(), grid_y.ravel(), ones.ravel()], axis=0)
    
    # Inverse warp: map target coordinates back to source image
    H_inv = np.linalg.inv(H)
    src_pts = H_inv @ target_pts
    src_pts /= src_pts[2, :]  # Normalize
    
    src_x = np.round(src_pts[0, :]).astype(int)
    src_y = np.round(src_pts[1, :]).astype(int)
    
    h_src, w_src = im.shape[:2]
    num_channels = im.shape[2] if im.ndim == 3 else 1
    
    # Check valid source coordinate mask
    valid_mask = (src_x >= 0) & (src_x < w_src) & (src_y >= 0) & (src_y < h_src)
    
    warped_im = np.zeros((out_h, out_w, num_channels) if im.ndim == 3 else (out_h, out_w), dtype=im.dtype)
    mask = np.zeros((out_h, out_w), dtype=bool)
    
    # Map valid pixels
    out_y_indices = (grid_y.ravel()[valid_mask] - min_y)
    out_x_indices = (grid_x.ravel()[valid_mask] - min_x)
    
    warped_im[out_y_indices, out_x_indices] = im[src_y[valid_mask], src_x[valid_mask]]
    mask[out_y_indices, out_x_indices] = True
    
    return warped_im, mask, (min_x, min_y)


def warpImageBilinear(im, H):
    """
    Warps image 'im' using Homography H and Bilinear interpolation (Inverse Warping).
    """
    min_x, max_x, min_y, max_y = get_bounding_box(im, H)
    out_w = max_x - min_x + 1
    out_h = max_y - min_y + 1
    
    grid_x, grid_y = np.meshgrid(np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1))
    ones = np.ones_like(grid_x)
    target_pts = np.stack([grid_x.ravel(), grid_y.ravel(), ones.ravel()], axis=0)
    
    H_inv = np.linalg.inv(H)
    src_pts = H_inv @ target_pts
    src_pts /= src_pts[2, :]
    
    src_x = src_pts[0, :]
    src_y = src_pts[1, :]
    
    h_src, w_src = im.shape[:2]
    num_channels = im.shape[2] if im.ndim == 3 else 1
    
    # Determine four surrounding pixel coordinates
    x0 = np.floor(src_x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(src_y).astype(int)
    y1 = y0 + 1
    
    valid_mask = (x0 >= 0) & (x1 < w_src) & (y0 >= 0) & (y1 < h_src)
    
    warped_im = np.zeros((out_h, out_w, num_channels) if im.ndim == 3 else (out_h, out_w), dtype=np.float32)
    mask = np.zeros((out_h, out_w), dtype=bool)
    
    if not np.any(valid_mask):
        return warped_im.astype(im.dtype), mask, (min_x, min_y)
        
    # Interpolation weights
    wx1 = src_x[valid_mask] - x0[valid_mask]
    wx0 = 1.0 - wx1
    wy1 = src_y[valid_mask] - y0[valid_mask]
    wy0 = 1.0 - wy1
    
    out_y = grid_y.ravel()[valid_mask] - min_y
    out_x = grid_x.ravel()[valid_mask] - min_x
    
    v_x0, v_x1 = x0[valid_mask], x1[valid_mask]
    v_y0, v_y1 = y0[valid_mask], y1[valid_mask]
    
    if im.ndim == 3:
        for c in range(num_channels):
            I00 = im[v_y0, v_x0, c]
            I10 = im[v_y0, v_x1, c]
            I01 = im[v_y1, v_x0, c]
            I11 = im[v_y1, v_x1, c]
            
            interp_val = wy0 * (wx0 * I00 + wx1 * I10) + wy1 * (wx0 * I01 + wx1 * I11)
            warped_im[out_y, out_x, c] = interp_val
    else:
        I00 = im[v_y0, v_x0]
        I10 = im[v_y0, v_x1]
        I01 = im[v_y1, v_x0]
        I11 = im[v_y1, v_x1]
        
        interp_val = wy0 * (wx0 * I00 + wx1 * I10) + wy1 * (wx0 * I01 + wx1 * I11)
        warped_im[out_y, out_x] = interp_val
        
    mask[out_y, out_x] = True
    return warped_im.astype(im.dtype), mask, (min_x, min_y)


def create_alpha_weight_map(h, w):
    """
    Generates a 2D distance transform alpha mask where weight falls off from center to 0 at borders.
    """
    mask = np.ones((h, w), dtype=np.uint8)
    mask[0, :] = 0
    mask[-1, :] = 0
    mask[:, 0] = 0
    mask[:, -1] = 0
    
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_val = np.max(dist_map)
    if max_val > 0:
        dist_map = dist_map / max_val
    return dist_map