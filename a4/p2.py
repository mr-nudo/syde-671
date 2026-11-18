import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from epipolar_utils import *
from p1 import normalized_eight_point_alg, plot_epipolar_lines_on_images

'''
COMPUTE_EPIPOLE computes the epipole e in homogenous coordinates
given the fundamental matrix
Arguments:
    F - the Fundamental matrix solved for with normalized_eight_point_alg(points1, points2)

Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(F):
    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1, :]
    return e / e[2]

'''
COMPUTE_H computes a homography to map an epipole to infinity along the horizontal axis 
Arguments:
    e - the epipole
    im - the image
Returns:
    H - homography matrix
'''
def compute_H(e, im):
    height, width = im.shape[:2]
    
    # 1. Translation matrix moving center to (0, 0, 1)
    T = np.array([
        [1.0, 0.0, -width / 2.0],
        [0.0, 1.0, -height / 2.0],
        [0.0, 0.0, 1.0]
    ])
    
    Te = T @ e
    e1, e2 = Te[0], Te[1]
    
    # 2. Rotation matrix to place epipole on the horizontal axis
    alpha = 1.0 if e1 >= 0 else -1.0
    norm_e = np.sqrt(e1**2 + e2**2)
    
    if norm_e < 1e-8:
        R = np.eye(3)
    else:
        R = np.array([
            [alpha * e1 / norm_e, alpha * e2 / norm_e, 0.0],
            [-alpha * e2 / norm_e, alpha * e1 / norm_e, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
    # 3. Matrix G mapping (f, 0, 1) to infinity (f, 0, 0)
    f = max(width, height)
    G = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0 / f, 0.0, 1.0]
    ])
    
    return np.linalg.inv(T) @ G @ R @ T

'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    H2 = compute_H(e2, im2)
    
    e_x = np.array([
        [0.0, -e2[2], e2[1]],
        [e2[2], 0.0, -e2[0]],
        [-e2[1], e2[0], 0.0]
    ])
    
    v = np.array([1.0, 1.0, 1.0])
    M = e_x @ F + np.outer(e2, v)
    
    num_pts = points1.shape[0]
    
    transformed_pts = (H2 @ M @ points1.T).T
    x_hat = transformed_pts[:, 0] / transformed_pts[:, 2]
    y_hat = transformed_pts[:, 1] / transformed_pts[:, 2]
    
    W = np.column_stack([x_hat, y_hat, np.ones(num_pts)])
    h2_pts2 = (H2 @ points2.T).T
    b = h2_pts2[:, 0] / h2_pts2[:, 2]
    
    # Use regularized least-squares to prevent ill-conditioned scaling blowouts
    a, _, _, _ = np.linalg.lstsq(W, b, rcond=1e-3)
    
    Ha = np.array([
        [a[0], a[1], a[2]],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    Ha = Ha / Ha[2, 2]
    H1 = Ha @ H2 @ M
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    # F is such that (points2)^T * F * points1 = 0, so e1 is e' and e2 is e
    e1 = compute_epipole(F.T)
    e2 = compute_epipole(F)
    print("e1", e1)
    print("e2", e2)

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print('')

    # Normalize overall homographies
    H1 = H1 / H1[2, 2]
    H2 = H2 / H2[2, 2]

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    
    # Normalize figure sizing and aspect ratio to prevent subplot compression
    fig = plt.gcf()
    fig.set_size_inches(12, 6)
    plt.tight_layout()
    
    plt.savefig('media/p2_rectified.png', bbox_inches='tight', dpi=300)
    plt.show()