import os
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from epipolar_utils import *

def lls_eight_point_alg(points1, points2):
    """
    LLS_EIGHT_POINT_ALG computes the fundamental matrix from matching points using 
    linear least squares eight point algorithm.
    
    Arguments:
        points1 - N x 3 homogeneous points in the first image (x, y, 1)
        points2 - N x 3 homogeneous points in the second image (x, y, 1)
    Returns:
        F - the 3 x 3 fundamental matrix such that (points2)^T * F * points1 = 0
    """
    num_pts = points1.shape[0]
    A = np.zeros((num_pts, 9))
    
    for i in range(num_pts):
        x1, y1, _ = points1[i]
        x2, y2, _ = points2[i]
        A[i] = [x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1.0]
        
    # Solve Af = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    f = Vt[-1, :]
    F = f.reshape(3, 3)
    
    # Enforce rank-2 constraint via SVD
    U_f, S_f, Vt_f = np.linalg.svd(F)
    S_f[2] = 0.0
    F = U_f @ np.diag(S_f) @ Vt_f
    
    return F


def normalized_eight_point_alg(points1, points2):
    """
    NORMALIZED_EIGHT_POINT_ALG computes the fundamental matrix from matching points
    using the normalized eight point algorithm.
    """
    def get_normalization_matrix(points):
        pts_xy = points[:, :2]
        centroid = np.mean(pts_xy, axis=0)
        shifted = pts_xy - centroid
        mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
        scale = np.sqrt(2.0) / mean_dist
        
        T = np.array([
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0]
        ])
        return T

    T1 = get_normalization_matrix(points1)
    T2 = get_normalization_matrix(points2)
    
    norm_points1 = (T1 @ points1.T).T
    norm_points2 = (T2 @ points2.T).T
    
    # Compute F on normalized coordinates
    F_norm = lls_eight_point_alg(norm_points1, norm_points2)
    
    # Denormalize F: F = T2^T * F_norm * T1
    F = T2.T @ F_norm @ T1
    
    # Re-enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    F = U @ np.diag(S) @ Vt
    
    return F / F[2, 2]


def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    def plot_epipolar_lines_on_image(points1, points2, im, F):
        im_height = im.shape[0]
        im_width = im.shape[1]
        lines = F.T.dot(points2.T)
        plt.imshow(im, cmap='gray')
        for line in lines.T:
            a, b, c = line
            xs = [1, im.shape[1]-1]
            ys = [(-c - a * x) / b for x in xs]
            plt.plot(xs, ys, 'r')
        for i in range(points1.shape[0]):
            x, y, _ = points1[i]
            plt.plot(x, y, '*b')
        plt.axis([0, im_width, im_height, 0])

    new_figsize = (8 * (float(max(im1.shape[1], im2.shape[1])) / min(im1.shape[1], im2.shape[1]))**2, 6)
    fig = plt.figure(figsize=new_figsize)
    plt.subplot(121)
    plot_epipolar_lines_on_image(points1, points2, im1, F)
    plt.axis('off')
    plt.subplot(122)
    plot_epipolar_lines_on_image(points2, points1, im2, F.T)
    plt.axis('off')


def compute_distance_to_epipolar_lines(points1, points2, F):
    """
    COMPUTE_DISTANCE_TO_EPIPOLAR_LINES computes the average distance of a set of 
    points1 to their corresponding epipolar lines derived from points2 and F.
    """
    lines = F.T.dot(points2.T) # shape: 3 x N
    numer = np.abs(np.sum(lines.T * points1, axis=1))
    denom = np.sqrt(lines[0, :] ** 2 + lines[1, :] ** 2)
    
    distances = numer / denom
    return np.mean(distances)


if __name__ == '__main__':
    os.makedirs('media', exist_ok=True)
    
    for im_set in ['data/set1', 'data/set2']:
        print('-'*80)
        print("Set:", im_set)
        print('-'*80)

        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        F_lls = lls_eight_point_alg(points1, points2)
        print("Fundamental Matrix from LLS 8-point algorithm:\n", F_lls)
        print("Distance to lines in image 1 for LLS:", compute_distance_to_epipolar_lines(points1, points2, F_lls))
        print("Distance to lines in image 2 for LLS:", compute_distance_to_epipolar_lines(points2, points1, F_lls.T))

        F_normalized = normalized_eight_point_alg(points1, points2)
        pFp = [points2[i].dot(F_normalized.dot(points1[i])) for i in range(points1.shape[0])]
        print("p'^T F p =", np.abs(pFp).max())
        print("Fundamental Matrix from normalized 8-point algorithm:\n", F_normalized)
        print("Distance to lines in image 1 for normalized:", compute_distance_to_epipolar_lines(points1, points2, F_normalized))
        print("Distance to lines in image 2 for normalized:", compute_distance_to_epipolar_lines(points2, points1, F_normalized.T))

        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plt.savefig(f'media/p1_lls_{im_set.split("/")[-1]}.png', bbox_inches='tight')
        plt.close()

        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)
        plt.savefig(f'media/p1_norm_{im_set.split("/")[-1]}.png', bbox_inches='tight')
        plt.close()