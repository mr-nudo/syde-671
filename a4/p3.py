import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

def factorization_method(points_im1, points_im2):
    """
    FACTORIZATION_METHOD The Tomasi and Kanade Factorization Method to determine
    the 3D structure of the scene and the motion of the cameras.
    
    Arguments:
        points_im1 - N x 3 homogeneous points in the first image (x, y, 1)
        points_im2 - N x 3 homogeneous points in the second image (x, y, 1)
    Returns:
        structure - 3 x N matrix of 3D points
        motion - 4 x 3 motion matrix
    """
    # Extract x and y coordinates (drop the homogeneous '1' column)
    pts1 = points_im1[:, :2].T  # 2 x N
    pts2 = points_im2[:, :2].T  # 2 x N
    
    # 1. Center the measurements (subtract mean for each coordinate stream)
    pts1_centered = pts1 - np.mean(pts1, axis=1, keepdims=True)
    pts2_centered = pts2 - np.mean(pts2, axis=1, keepdims=True)
    
    # Construct 4 x N measurement matrix W
    W = np.vstack((pts1_centered, pts2_centered))
    
    # 2. SVD of W
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    
    # Keep top 3 singular values/vectors
    U3 = U[:, :3]
    S3 = np.diag(np.sqrt(S[:3]))
    V3 = Vt[:3, :]
    
    # Initial affine motion and structure
    M = U3 @ S3  # 4 x 3
    S_3d = S3 @ V3  # 3 x N
    
    # 3. Metric constraint optimization to resolve affine ambiguity (Q matrix)
    A = np.zeros((2, 6))
    b = np.zeros((2, 1))
    
    for i in range(2):
        r1 = M[2*i, :]
        r2 = M[2*i+1, :]
        
        A[i, 0] = r1[0]*r1[0] - r2[0]*r2[0]
        A[i, 1] = 2.0 * (r1[0]*r1[1] - r2[0]*r2[1])
        A[i, 2] = r1[1]*r1[1] - r2[1]*r2[1]
        A[i, 3] = 2.0 * (r1[0]*r1[2] - r2[0]*r2[2])
        A[i, 4] = 2.0 * (r1[1]*r1[2] - r2[1]*r2[2])
        A[i, 5] = r1[2]*r1[2] - r2[2]*r2[2]
        
        b[i, 0] = 1.0 if i == 0 else 0.0
        
    # Solve for symmetric matrix L = Q * Q^T
    L_vec, _, _, _ = np.linalg.lstsq(A, np.array([1.0, 0.0]), rcond=None)
    
    L = np.array([
        [L_vec[0], L_vec[1], L_vec[3]],
        [L_vec[1], L_vec[2], L_vec[4]],
        [L_vec[3], L_vec[4], L_vec[5]]
    ])
    
    # Ensure L is positive semi-definite, then compute Q via Cholesky decomposition
    try:
        L = 0.5 * (L + L.T)
        vals, vecs = np.linalg.eigh(L)
        vals[vals < 0] = 1e-6
        L_fixed = vecs @ np.diag(vals) @ vecs.T
        Q = np.linalg.cholesky(L_fixed)
    except np.linalg.LinAlgError:
        Q = np.eye(3)
        
    # Apply metric upgrade
    motion = M @ Q
    structure = np.linalg.inv(Q) @ S_3d
    
    return structure, motion

if __name__ == '__main__':
    os.makedirs('media', exist_ok=True)
    
    for im_set in ['data/set1', 'data/set1_subset']:
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points_im1 = get_data_from_txt_file(im_set + '/pt_2D_1.txt')
        points_im2 = get_data_from_txt_file(im_set + '/pt_2D_2.txt')
        points_3d = get_data_from_txt_file(im_set + '/pt_3D.txt')
        assert (points_im1.shape == points_im2.shape)

        structure, motion = factorization_method(points_im1, points_im2)

        # Scale normalization to map reconstructed magnitude back to ground truth range
        recon_spread = np.std(structure)
        gt_spread = np.std(points_3d)
        if recon_spread > 0:
            structure = structure * (gt_spread / recon_spread)

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(121, projection='3d')
        scatter_3D_axis_equal(structure[0,:], structure[1,:], structure[2,:], ax)
        ax.set_title('Factorization Method')
        
        ax = fig.add_subplot(122, projection='3d')
        scatter_3D_axis_equal(points_3d[:,0], points_3d[:,1], points_3d[:,2], ax)
        ax.set_title('Ground Truth')

        suffix = im_set.split('/')[-1]
        plt.savefig(f'media/p3_factorization_{suffix}.png', bbox_inches='tight')
        plt.close()