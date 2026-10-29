"""
Loop Closure Detection & Verification Engine
==========================================

Implements the online loop closure detection and validation pipeline for the SLAM system.
To eliminate accumulated dead-reckoning drift, this module actively searches for spatial
loop closures by evaluating the statistical proximity of historical keyframes. It ensures
extreme robustness by fusing statistical covariance thresholds with strict multi-view
visual geometry checks before committing new constraints to the global graph.

Key Responsibilities:
- Accumulating global spatial uncertainty (covariances) over the pose graph via Dijkstra's
  shortest path algorithm.
- Statistically filtering loop candidates using 6-DOF Mahalanobis distance and 99% Chi-Square
  confidence thresholds.
- Validating geometric consensus through robust 4-way stereo feature matching and custom
  RANSAC P3P filtering.
- Formulating localized 2-frame mini-Bundle Adjustments to extract highly accurate relative
  poses and conditional covariances for global graph updates.
"""

import cv2
import gtsam
import numpy as np
import heapq
from collections import defaultdict

from feature_tracking import match_features_ratio_test, process_stereo_frame, prepare_4way_data
from geometry import run_custom_ransac


def build_initial_adjacency_list(pose_graph):
    """Builds the adjacency list ONCE from the base Pose Graph."""
    adj_list = defaultdict(list)
    for i in range(pose_graph.size()):
        factor = pose_graph.at(i)
        if isinstance(factor, gtsam.BetweenFactorPose3):
            keys = factor.keys()
            k1, k2 = keys[0], keys[1]
            edge_cov = factor.noiseModel().covariance()
            adj_list[k1].append((k2, edge_cov))
            adj_list[k2].append((k1, edge_cov))
    return adj_list


def get_shortest_paths_covariances(adj_list, start_key):
    """
    Implements Dijkstra's algorithm using a pre-built adjacency list.
    """
    best_covariances = {}
    min_dets = defaultdict(lambda: float('inf'))
    pq = []

    zero_cov = np.zeros((6, 6))
    best_covariances[start_key] = zero_cov
    min_dets[start_key] = 0.0
    heapq.heappush(pq, (0.0, start_key))

    while pq:
        current_det, u = heapq.heappop(pq)

        if current_det > min_dets[u]:
            continue

        cov_u = best_covariances[u]

        for v, edge_cov in adj_list[u]:
            new_cov = cov_u + edge_cov
            new_det = np.linalg.det(new_cov)

            if new_det < min_dets[v]:
                min_dets[v] = new_det
                best_covariances[v] = new_cov
                heapq.heappush(pq, (new_det, v))

    return best_covariances


def filter_candidates_by_mahalanobis(optimized_values, current_key, accumulated_covs,
                                     mahalanobis_thresh=16.81, min_frame_gap=50):
    """
    Filters loop closure candidates using the Mahalanobis distance.

    :param optimized_values: gtsam.Values containing the current global poses.
    :param current_key: The gtsam.symbol key of the current frame c_n.
    :param accumulated_covs: Dictionary of accumulated covariances from Dijkstra.
    :param mahalanobis_thresh: Chi-Square threshold for 6-DOF (16.81 = 99% confidence).
    :param min_frame_gap: Minimum index difference to avoid matching immediate previous frames.
    :return: A sorted list of tuples (candidate_key, mahalanobis_distance).
    """
    valid_candidates = []

    # Extract the current pose c_n
    pose_n = optimized_values.atPose3(current_key)
    current_idx = gtsam.symbolIndex(current_key)

    for candidate_key, cov_matrix in accumulated_covs.items():
        candidate_idx = gtsam.symbolIndex(candidate_key)

        # 1. Enforce temporal gap (We don't want to loop closure with frame t-1)
        if (current_idx - candidate_idx) < min_frame_gap:
            continue

        # 2. Extract the candidate pose c_i
        if not optimized_values.exists(candidate_key):
            continue
        pose_i = optimized_values.atPose3(candidate_key)

        # 3. Compute the relative error vector in tangent space (6x1 vector)
        # c_i.between(pose_n) calculates c_i^-1 * c_n
        relative_pose = pose_i.between(pose_n)
        error_vector = gtsam.Pose3.Logmap(relative_pose)

        # 4. Invert the accumulated covariance matrix (Information Matrix)
        # Use pseudo-inverse to ensure stability if the matrix is ill-conditioned
        info_matrix = np.linalg.pinv(cov_matrix)

        # 5. Calculate squared Mahalanobis distance: v^T * Sigma^-1 * v
        md_squared = error_vector.T @ info_matrix @ error_vector

        # 6. Apply statistical threshold
        if md_squared < mahalanobis_thresh:
            valid_candidates.append((candidate_key, md_squared))

    # Sort candidates so the most statistically probable (lowest MD) are first
    valid_candidates.sort(key=lambda x: x[1])

    return valid_candidates


def validate_and_estimate_loop(data_path, K, P1, P2, M2, f_i, f_n, min_inliers=30):
    """
    Section 7.2: Consensus Matching using existing 4-way Stereo Pipeline.
    Returns the initial relative pose guess and tracking data for the Bundle.
    """
    # 1. Process both frames using exact Ex 3/4 pipeline
    kp_i_l, des_i_l, kp_i_r, des_i_r, inliers_i, pts3D_i = process_stereo_frame(data_path, f_i, "AKAZE", P1, P2)
    kp_n_l, des_n_l, kp_n_r, des_n_r, inliers_n, pts3D_n = process_stereo_frame(data_path, f_n, "AKAZE", P1, P2)

    # 2. Match temporally between the left images
    matches_temporal, _ = match_features_ratio_test(des_i_l, des_n_l, "AKAZE", ratio=0.75)

    # 3. Use exact 4-way preparation logic
    pts3D_i_synced, pts2D_n_synced, track_indices = prepare_4way_data(
        inliers_i, inliers_n, matches_temporal, kp_i_l, kp_n_l, pts3D_i
    )

    if len(pts3D_i_synced) < min_inliers:
        return False, None

    # 4. Use custom vectorized P3P RANSAC
    R, tvec, mask_list = run_custom_ransac(
        pts3D_i_synced, track_indices, kp_i_l, kp_i_r, kp_n_l, kp_n_r, K, P1, P2, M2,
        threshold=2.0, should_print=False
    )

    if sum(mask_list) < min_inliers:
        return False, None

    # 5. Convert PnP output to GTSAM Pose3 for the initial guess
    initial_pose_guess = gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(tvec.flatten()))

    # Package the surviving inliers for the mini-Bundle Adjustment in 7.3
    mask = np.array(mask_list, dtype=bool)

    # Extract the actual 2D pixels using the same logic from geometry.py
    actual_l0 = np.array([kp_i_l[t[0]].pt for t in track_indices])[mask]
    actual_r0 = np.array([kp_i_r[t[1]].pt for t in track_indices])[mask]
    actual_l1 = np.array([kp_n_l[t[2]].pt for t in track_indices])[mask]
    actual_r1 = np.array([kp_n_r[t[3]].pt for t in track_indices])[mask]

    matches_4way = [cv2.DMatch(_queryIdx=t[0], _trainIdx=t[2], _distance=1.0) for t in track_indices]

    loop_data = {
        'initial_pose': initial_pose_guess,
        'pts3D_i': pts3D_i_synced[mask],
        'pts2D_i_L': actual_l0,
        'pts2D_i_R': actual_r0,
        'pts2D_n_L': actual_l1,
        'pts2D_n_R': actual_r1,
        'matches_4way': matches_4way,
        'mask': mask_list,
        'kp_i_l': kp_i_l,
        'kp_n_l': kp_n_l
    }

    return True, loop_data


def calculate_loop_closure_bundle(gtsam_calib, loop_data, f_i, f_n, pixel_sigma=1.0):
    """
    Section 7.3: Mini 2-Frame Bundle Optimization.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()
    key_i = gtsam.symbol('c', f_i)
    key_n = gtsam.symbol('c', f_n)

    # 1. Anchor Frame i (Required by GTSAM to prevent under-constrained gauge freedom)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
    graph.add(gtsam.PriorFactorPose3(key_i, gtsam.Pose3(), prior_noise))
    initial_values.insert(key_i, gtsam.Pose3())

    # 2. Insert Initial Guess for Frame n
    initial_values.insert(key_n, loop_data['initial_pose'])

    # 3. Add Landmarks and Stereo Factors (with Robust Noise)
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, pixel_sigma)
    robust_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(2.0), measurement_noise
    )

    for idx in range(len(loop_data['pts3D_i'])):
        q_key = gtsam.symbol('q', idx)
        pt3d = loop_data['pts3D_i'][idx]
        initial_values.insert(q_key, gtsam.Point3(pt3d))

        uL_i, v_i = loop_data['pts2D_i_L'][idx]
        uR_i = loop_data['pts2D_i_R'][idx][0]
        graph.add(gtsam.GenericStereoFactor3D(
            gtsam.StereoPoint2(uL_i, uR_i, v_i), robust_noise, key_i, q_key, gtsam_calib))

        uL_n, v_n = loop_data['pts2D_n_L'][idx]
        uR_n = loop_data['pts2D_n_R'][idx][0]
        graph.add(gtsam.GenericStereoFactor3D(
            gtsam.StereoPoint2(uL_n, uR_n, v_n), robust_noise, key_n, q_key, gtsam_calib))

    # 4. Optimize the Mini-Bundle
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    optimized_values = optimizer.optimize()
    marginals = gtsam.Marginals(graph, optimized_values)

    # Return everything needed for Ex 6's Schur Complement extraction
    return optimized_values, marginals, key_i, key_n
