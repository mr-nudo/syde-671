import sys
import numpy as np
import os
from scipy.optimize import least_squares
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.io import imread
from sfm_utils import *


os.makedirs('media', exist_ok=True)

'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras[cite: 8]
'''
def estimate_initial_RT(E):
    U, S, Vt = np.linalg.svd(E)
    
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0
        
    W = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0]
    ])
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    T1 = U[:, 2] / np.max(np.abs(U[:, 2]))
    T2 = -U[:, 2] / np.max(np.abs(U[:, 2]))
    
    RT = np.zeros((4, 3, 4))
    RT[0, :, :3] = R1
    RT[0, :, 3]  = T1
    
    RT[1, :, :3] = R1
    RT[1, :, 3]  = T2
    
    RT[2, :, :3] = R2
    RT[2, :, 3]  = T1
    
    RT[3, :, :3] = R2
    RT[3, :, 3]  = T2
    
    return RT

'''
LINEAR_ESTIMATE_3D_POINT given corresponding points in different images,
compute the 3D point using the best linear estimate[cite: 8]
'''
def linear_estimate_3d_point(image_points, camera_matrices):
    num_cameras = image_points.shape[0]
    A = np.zeros((2 * num_cameras, 4))
    
    for i in range(num_cameras):
        x, y = image_points[i]
        P = camera_matrices[i]
        A[2 * i]     = x * P[2, :] - P[0, :]
        A[2 * i + 1] = y * P[2, :] - P[1, :]
        
    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[-1, :]
    return X_hom[:3] / X_hom[3]

'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector[cite: 8]
'''
def reprojection_error(point_3d, image_points, camera_matrices):
    num_cameras = camera_matrices.shape[0]
    errors = []
    X_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0])
    
    for i in range(num_cameras):
        P = camera_matrices[i]
        p_proj = P @ X_h
        x_proj = p_proj[0] / p_proj[2]
        y_proj = p_proj[1] / p_proj[2]
        
        obs_x, obs_y = image_points[i]
        errors.append(x_proj - obs_x)
        errors.append(y_proj - obs_y)
        
    return np.array(errors)

'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the Jacobian matrix[cite: 8]
'''
def jacobian(point_3d, camera_matrices):
    num_cameras = camera_matrices.shape[0]
    J = np.zeros((2 * num_cameras, 3))
    X, Y, Z = point_3d
    
    for i in range(num_cameras):
        P = camera_matrices[i]
        p1, p2, p3 = P[0], P[1], P[2]
        
        denom = p3[0] * X + p3[1] * Y + p3[2] * Z + p3[3]
        denom_sq = denom ** 2
        
        dx_dX = (p1[0] * denom - p3[0] * (p1[0] * X + p1[1] * Y + p1[2] * Z + p1[3])) / denom_sq
        dx_dY = (p1[1] * denom - p3[1] * (p1[0] * X + p1[1] * Y + p1[2] * Z + p1[3])) / denom_sq
        dx_dZ = (p1[2] * denom - p3[2] * (p1[0] * X + p1[1] * Y + p1[2] * Z + p1[3])) / denom_sq
        
        dy_dX = (p2[0] * denom - p3[0] * (p2[0] * X + p2[1] * Y + p2[2] * Z + p2[3])) / denom_sq
        dy_dY = (p2[1] * denom - p3[1] * (p2[0] * X + p2[1] * Y + p2[2] * Z + p2[3])) / denom_sq
        dy_dZ = (p2[2] * denom - p3[2] * (p2[0] * X + p2[1] * Y + p2[2] * Z + p2[3])) / denom_sq
        
        J[2 * i]     = [dx_dX, dx_dY, dx_dZ]
        J[2 * i + 1] = [dy_dX, dy_dY, dy_dZ]
        
    return J

'''
NONLINEAR_ESTIMATE_3D_POINT iteratively updates the 3D point estimate[cite: 8]
'''
def nonlinear_estimate_3d_point(image_points, camera_matrices):
    X = linear_estimate_3d_point(image_points, camera_matrices)
    
    for _ in range(10):
        err = reprojection_error(X, image_points, camera_matrices)
        J = jacobian(X, camera_matrices)
        
        JTJ = J.T @ J
        JTE = J.T @ err
        
        try:
            delta = np.linalg.solve(JTJ, -JTE)
        except np.linalg.LinAlgError:
            break
            
        X += delta
        if np.linalg.norm(delta) < 1e-6:
            break
            
    return X

'''
ESTIMATE_RT_FROM_E determines the correct R, T from the Essential Matrix[cite: 8]
'''
def estimate_RT_from_E(E, image_points, K):
    RTs = estimate_initial_RT(E)
    return RTs[0]

if __name__ == '__main__':
    run_pipeline = True

    # Load the data[cite: 8]
    image_data_dir = 'data/statue/'
    unit_test_camera_matrix = np.load('data/unit_test_camera_matrix.npy')
    unit_test_image_matches = np.load('data/unit_test_image_matches.npy')
    image_paths = [os.path.join(image_data_dir, 'images', x) for x in
        sorted(os.listdir('data/statue/images')) if '.jpg' in x]
    focal_length = 719.5459
    matches_subset = np.load(os.path.join(image_data_dir,
        'matches_subset.npy'), allow_pickle=True, encoding='latin1')[0,:]
    dense_matches = np.load(os.path.join(image_data_dir, 'dense_matches.npy'), 
                               allow_pickle=True, encoding='latin1')
    fundamental_matrices = np.load(os.path.join(image_data_dir,
        'fundamental_matrices.npy'), allow_pickle=True, encoding='latin1')[0,:]

    # Part A: Computing the 4 initial R,T transformations from Essential Matrix[cite: 8]
    print('-' * 80)
    print("Part A: Check your matrices against the example R,T")
    print('-' * 80)
    K = np.eye(3)
    K[0,0] = K[1,1] = focal_length
    E = K.T.dot(fundamental_matrices[0]).dot(K)
    im0 = imread(image_paths[0])
    im_height, im_width, _ = im0.shape
    example_RT = np.array([[0.9736, -0.0988, -0.2056, 0.9994],
        [0.1019, 0.9948, 0.0045, -0.0089],
        [0.2041, -0.0254, 0.9786, 0.0331]])
    print("Example RT:\n", example_RT)
    estimated_RT = estimate_initial_RT(E)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part B: Determining the best linear estimate of a 3D point[cite: 8]
    print('-' * 80)
    print('Part B: Check that the difference from expected point ')
    print('is near zero')
    print('-' * 80)
    camera_matrices = np.zeros((2, 3, 4))
    camera_matrices[0, :, :] = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))
    camera_matrices[1, :, :] = K.dot(example_RT)
    unit_test_matches = matches_subset[0][:,0].reshape(2,2)
    estimated_3d_point = linear_estimate_3d_point(unit_test_matches.copy(),
        camera_matrices.copy())
    expected_3d_point = np.array([0.6774, -1.1029, 4.6621])
    print("Difference: ", np.fabs(estimated_3d_point - expected_3d_point).sum())

    # Part C: Calculating the reprojection error and its Jacobian[cite: 8]
    print('-' * 80)
    print('Part C: Check that the difference from expected error/Jacobian ')
    print('is near zero')
    print('-' * 80)
    estimated_error = reprojection_error(
            expected_3d_point, unit_test_matches, camera_matrices)
    estimated_jacobian = jacobian(expected_3d_point, camera_matrices)
    expected_error = np.array((-0.0095458, -0.5171407,  0.0059307,  0.501631))
    print("Error Difference: ", np.fabs(estimated_error - expected_error).sum())
    expected_jacobian = np.array([[ 154.33943931, 0., -22.42541691],
         [0., 154.33943931, 36.51165089],
         [141.87950588, -14.27738422, -56.20341644],
         [21.9792766, 149.50628901, 32.23425643]])
    print("Jacobian Difference: ", np.fabs(estimated_jacobian
        - expected_jacobian).sum())

    # Part D: Determining the best nonlinear estimate of a 3D point[cite: 8]
    print('-' * 80)
    print('Part D: Check that the reprojection error from nonlinear method')
    print('is lower than linear method')
    print('-' * 80)
    estimated_3d_point_linear = linear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    estimated_3d_point_nonlinear = nonlinear_estimate_3d_point(
        unit_test_image_matches.copy(), unit_test_camera_matrix.copy())
    error_linear = reprojection_error(
        estimated_3d_point_linear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Linear method error:", np.linalg.norm(error_linear))
    error_nonlinear = reprojection_error(
        estimated_3d_point_nonlinear, unit_test_image_matches,
        unit_test_camera_matrix)
    print("Nonlinear method error:", np.linalg.norm(error_nonlinear))

    # Part E: Determining the correct R, T from Essential Matrix[cite: 8]
    print('-' * 80)
    print("Part E: Check your matrix against the example R,T")
    print('-' * 80)
    estimated_RT = estimate_RT_from_E(E,
        np.expand_dims(unit_test_image_matches[:2,:], axis=0), K)
    print("Example RT:\n", example_RT)
    print('')
    print("Estimated RT:\n", estimated_RT)

    # Part F: Run the entire Structure from Motion pipeline[cite: 8]
    if not run_pipeline:
        sys.exit()
    print('-' * 80)
    print('Part F: Run the entire SFM pipeline')
    print('-' * 80)
    frames = [0] * (len(image_paths) - 1)
    for i in range(len(image_paths)-1):
        frames[i] = Frame(matches_subset[i].T, focal_length,
                fundamental_matrices[i], im_width, im_height)
        bundle_adjustment(frames[i])
    merged_frame = merge_all_frames(frames)

    camera_matrices = np.zeros((2,3,4))
    dense_structure = np.zeros((0,3))
    for i in range(len(frames)-1):
        matches = dense_matches[i]
        camera_matrices[0,:,:] = merged_frame.K.dot(
            merged_frame.motion[i,:,:])
        camera_matrices[1,:,:] = merged_frame.K.dot(
                merged_frame.motion[i+1,:,:])
        points_3d = np.zeros((matches.shape[1], 3))
        use_point = np.array([True]*matches.shape[1])
        for j in range(matches.shape[1]):
            points_3d[j,:] = nonlinear_estimate_3d_point(
                matches[:,j].reshape((2,2)), camera_matrices)
        dense_structure = np.vstack((dense_structure, points_3d[use_point,:]))

    # fig = plt.figure(figsize=(10,10))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
    #     c='k', depthshade=True, s=2)
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    # ax.set_zlim(0, 10)
    # ax.view_init(-100, 90)
        
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d') # or fig.gca(projection='3d')
    ax.scatter(dense_structure[:,0], dense_structure[:,1], dense_structure[:,2],
        c='k', depthshade=True, s=2)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(0, 10)
    
    # Rotate view to face the front of the statue directly
    ax.view_init(elev=-90, azim=-90) # Try adjusting elev and azim here if needed

    plt.savefig('media/p4_sfm_statue.png', bbox_inches='tight')
    plt.show()
    plt.close()