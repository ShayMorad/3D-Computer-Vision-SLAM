"""
Geometric Vision Math Library
=============================

Serves as the core mathematical and spatial computation library for the SLAM pipeline.
This module encapsulates pure geometric transformations, multi-view triangulations,
and projection mathematics. It operates strictly on numerical coordinate data, decoupled
from image processing or visual feature extraction.

Key Responsibilities:
- Manual Direct Linear Transformation (DLT) using Singular Value Decomposition (SVD).
- Highly optimized, vectorized multi-point triangulation via OpenCV.
- Euclidean reprojection error calculation for 3D-to-2D geometric verification.
"""

import cv2
import numpy as np

RANSAC_PROBABILITY = 0.999
DEFAULT_RANSAC_ITERATIONS = 1000


def triangulate_points_dlt(pts1, pts2, P1, P2):
    """
    Manual Direct Linear Triangulation.
    pts1, pts2: Nx2 arrays of matched 2D pixel coordinates.
    P1, P2: 3x4 projection matrices.
    Returns Nx3 array of 3D points.
    """
    num_points = pts1.shape[0]
    points_3d = np.zeros((num_points, 3))

    for i in range(num_points):
        u1, v1 = pts1[i, 0], pts1[i, 1]
        u2, v2 = pts2[i, 0], pts2[i, 1]

        # Construct the 4x4 A matrix
        A = np.array([
            u1 * P1[2, :] - P1[0, :],
            v1 * P1[2, :] - P1[1, :],
            u2 * P2[2, :] - P2[0, :],
            v2 * P2[2, :] - P2[1, :]
        ])

        # Solve AX = 0 using SVD
        U, S, Vh = np.linalg.svd(A)

        # The solution is the last row of Vh (which is V transpose)
        X_homogeneous = Vh[-1, :]

        # Normalize by dividing by the 4th coordinate (w)
        X = X_homogeneous[:3] / X_homogeneous[3]
        points_3d[i, :] = X

    return points_3d


def triangulate_points_cv2(pts1, pts2, P1, P2):
    """
    Vectorized Linear Triangulation using OpenCV.
    Replaces the slow manual SVD loop for instant execution.
    """
    # cv2.triangulatePoints requires 2xN arrays
    pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    # Normalize by the 4th homogeneous coordinate
    pts4D /= pts4D[3]
    # Return as Nx3 array
    return pts4D[:3].T


def compute_reprojection_error(pt3D, P, pt2D_actual):
    """
    Projects a 3D point through a camera matrix and calculates the
    Euclidean distance (error) to the actual 2D matched pixel.
    """
    # 1. Convert 3D point to homogeneous coordinates [X, Y, Z, 1]
    pt3D_hom = np.append(pt3D, 1.0)

    # 2. Project into the 2D image plane using the camera matrix P
    pt2D_proj_hom = P @ pt3D_hom

    # 3. Normalize by Z (the 3rd coordinate) to get Cartesian [u, v] pixels
    # Avoid division by zero
    if pt2D_proj_hom[2] == 0:
        return float('inf')

    u_proj = pt2D_proj_hom[0] / pt2D_proj_hom[2]
    v_proj = pt2D_proj_hom[1] / pt2D_proj_hom[2]

    # 4. Calculate Euclidean distance between projected and actual pixel
    u_actual, v_actual = pt2D_actual
    error = np.sqrt((u_proj - u_actual) ** 2 + (v_proj - v_actual) ** 2)

    return error


def run_custom_ransac(obj_pts_3D, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1, P2, M2,
                      num_iterations=DEFAULT_RANSAC_ITERATIONS, threshold=2.0, should_print=True):
    """
    Vectorized RANSAC implementation using NumPy Matrix Multiplication.
    """
    best_supporters_count = 0
    best_mask = None
    total_points = len(obj_pts_3D)

    # 1. Convert all 3D points to homogeneous coordinates (4 x N matrix)
    pts3D_hom = np.hstack((obj_pts_3D, np.ones((total_points, 1)))).T

    # 2. Extract all 2D pixel coordinates into (N x 2) arrays
    actual_l0 = np.array([kp0_l[t[0]].pt for t in track_indices])
    actual_r0 = np.array([kp0_r[t[1]].pt for t in track_indices])
    actual_l1 = np.array([kp1_l[t[2]].pt for t in track_indices])
    actual_r1 = np.array([kp1_r[t[3]].pt for t in track_indices])

    # 3. Pre-calculate Frame 0 projections (These never change during the loop)
    proj_l0 = P1 @ pts3D_hom
    proj_r0 = P2 @ pts3D_hom

    # Fast Z-normalization (with division-by-zero protection)
    z_l0 = np.where(proj_l0[2] == 0, 1e-6, proj_l0[2])
    z_r0 = np.where(proj_r0[2] == 0, 1e-6, proj_r0[2])

    err_l0 = np.sqrt(((proj_l0[0] / z_l0) - actual_l0[:, 0]) ** 2 + ((proj_l0[1] / z_l0) - actual_l0[:, 1]) ** 2)
    err_r0 = np.sqrt(((proj_r0[0] / z_r0) - actual_r0[:, 0]) ** 2 + ((proj_r0[1] / z_r0) - actual_r0[:, 1]) ** 2)

    valid_frame0 = (err_l0 <= threshold) & (err_r0 <= threshold)
    t_stereo = M2[:, 3].reshape(3, 1)

    for i in range(num_iterations):
        sample_indices = np.random.choice(total_points, 4, replace=False)
        sample_3D = obj_pts_3D[sample_indices]
        sample_2D = actual_l1[sample_indices]

        success, rvec, tvec = cv2.solvePnP(sample_3D, sample_2D, K, None, flags=cv2.SOLVEPNP_P3P)
        if not success:
            continue

        R, _ = cv2.Rodrigues(rvec)

        # Build Frame 1 Projection Matrices
        P_l1 = K @ np.hstack((R, tvec))
        P_r1 = K @ np.hstack((R, tvec + t_stereo))

        # Project ALL points instantly via Matrix Multiplication
        proj_l1 = P_l1 @ pts3D_hom
        proj_r1 = P_r1 @ pts3D_hom

        z_l1 = np.where(proj_l1[2] == 0, 1e-6, proj_l1[2])
        z_r1 = np.where(proj_r1[2] == 0, 1e-6, proj_r1[2])

        err_l1 = np.sqrt(((proj_l1[0] / z_l1) - actual_l1[:, 0]) ** 2 + ((proj_l1[1] / z_l1) - actual_l1[:, 1]) ** 2)
        err_r1 = np.sqrt(((proj_r1[0] / z_r1) - actual_r1[:, 0]) ** 2 + ((proj_r1[1] / z_r1) - actual_r1[:, 1]) ** 2)

        # Vectorized Consensus Check
        current_mask = valid_frame0 & (err_l1 <= threshold) & (err_r1 <= threshold)

        supporters_count = np.sum(current_mask)
        if supporters_count > best_supporters_count:
            best_supporters_count = supporters_count
            best_mask = current_mask

            # --- DYNAMIC EARLY STOPPING ---
            w = best_supporters_count / total_points
            # Prevent division by zero if w is perfectly 1.0
            if w > 0.99:
                break

            p = RANSAC_PROBABILITY  # 99.9% confidence
            s = 4  # Sample size for P3P

            # Calculate dynamic N
            numerator = np.log(1 - p)
            denominator = np.log(1 - (w ** s))
            dynamic_N = int(numerator / denominator) + 1

            # If our current iteration `i` has surpassed the required dynamic_N, we can safely exit!
            if i > dynamic_N:
                break

    if should_print:
        print(f"RANSAC found {best_supporters_count} inliers out of {total_points} after {num_iterations} iterations.")

    # Refinement using all identified inliers
    inlier_3D = obj_pts_3D[best_mask]
    inlier_2D = actual_l1[best_mask]

    success, refined_rvec, refined_tvec = cv2.solvePnP(inlier_3D, inlier_2D, K, None)
    refined_R, _ = cv2.Rodrigues(refined_rvec)

    return refined_R, refined_tvec, best_mask.tolist()
