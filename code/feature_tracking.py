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
from geometry import triangulate_points_cv2
from dataset_utils import read_stereo_pair

from tracking_database import TrackingDB
MAX_HEIGHT = 50
MIN_HEIGHT = -10
MAX_DEPTH = 150
MIN_DEPTH = 0
AKAZE_THRESHOLD = 0.0007
DEFAULT_FEATURES = 2000


def extract_features(img, algorithm="AKAZE", nfeatures=DEFAULT_FEATURES):
    """
    Factory function to detect key-points and compute descriptors.
    """
    algorithm = algorithm.upper()

    if algorithm == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
    elif algorithm == "SIFT":
        detector = cv2.SIFT_create(nfeatures=nfeatures)
    elif algorithm == "AKAZE":
        # AKAZE uses a threshold rather than a strict feature limit by default.
        detector = cv2.AKAZE_create(threshold=AKAZE_THRESHOLD)
    else:
        raise ValueError(f"Unsupported feature extraction algorithm: {algorithm}")

    kp, des = detector.detectAndCompute(img, None)
    return kp, des


def match_features(des1, des2, algorithm="AKAZE"):
    """
    Factory function to match features using the correct distance norm.
    Automatically handles float (L2) vs. binary (Hamming) descriptors.
    """
    algorithm = algorithm.upper()

    # Gradient-based (float) descriptors need L2 norm
    if algorithm == "SIFT":
        norm = cv2.NORM_L2
    # Binary descriptors need Hamming norm
    elif algorithm in ["ORB", "AKAZE"]:
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported matching algorithm: {algorithm}")

    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(des1, des2)

    return sorted(matches, key=lambda x: x.distance)


def match_features_ratio_test(des1, des2, algorithm="AKAZE", ratio=0.75):
    """
    Matches features and applies Lowe's distance ratio test.
    Automatically handles the correct norm based on the algorithm.
    """
    algorithm = algorithm.upper()

    if algorithm == "SIFT":
        norm = cv2.NORM_L2
    elif algorithm in ["ORB", "AKAZE"]:
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unsupported matching algorithm: {algorithm}")

    # crossCheck must be False for knnMatch to return multiple neighbors
    bf = cv2.BFMatcher(norm, crossCheck=False)
    knn_matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    rejected_matches = []

    for match_pair in knn_matches:
        # Ensure we actually have 2 neighbors to compare
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)
            else:
                rejected_matches.append(m)

    return good_matches, rejected_matches


def process_stereo_frame(data_path, frame_idx, algo, P1, P2):
    """
    Complete pipeline for a single stereo pair: Extract, Match, Filter, Triangulate.
    Returns synced 2D keypoints, descriptors, filtered matches, and filtered 3D points.
    """
    img_l, img_r = read_stereo_pair(data_path, frame_idx)

    kp_l, des_l = extract_features(img_l, algo)
    kp_r, des_r = extract_features(img_r, algo)
    matches = match_features(des_l, des_r, algo)

    # Geometric Filter (deviations <= 2px)
    deviations = np.array([abs(kp_l[m.queryIdx].pt[1] - kp_r[m.trainIdx].pt[1]) for m in matches])
    inliers = [m for m, dev in zip(matches, deviations) if dev <= 2]

    # Triangulate
    pts_l = np.array([kp_l[m.queryIdx].pt for m in inliers])
    pts_r = np.array([kp_r[m.trainIdx].pt for m in inliers])
    pts3D = triangulate_points_cv2(pts_l, pts_r, P1, P2)

    # Logical Bounding Box Filter
    X = pts3D[:, 0]
    Y = pts3D[:, 1]
    Z = pts3D[:, 2]

    valid_mask = (np.isfinite(pts3D).all(axis=1)) & (Z > MIN_DEPTH) & (Z < MAX_DEPTH) & (Y > MIN_HEIGHT) & (
            Y < MAX_HEIGHT)

    clean_pts3D = pts3D[valid_mask]
    clean_inliers = [inliers[i] for i, is_valid in enumerate(valid_mask) if is_valid]

    return kp_l, des_l, kp_r, des_r, clean_inliers, clean_pts3D


def track_features_4way(matches0, matches1, matches_temporal):
    """
    Finds feature indices that successfully matched across all 4 images.
    Returns a list of tuples: (idx_left0, idx_right0, idx_left1, idx_right1)
    """
    # Create fast lookup dictionaries for the stereo pairs
    # Key = Left Image Index, Value = Right Image Index
    dict0 = {m.queryIdx: m.trainIdx for m in matches0}
    dict1 = {m.queryIdx: m.trainIdx for m in matches1}

    four_way_indices = []

    # matches_temporal: query is left0, train is left1
    for m in matches_temporal:
        idx_l0 = m.queryIdx
        idx_l1 = m.trainIdx

        # If this temporal feature ALSO has a valid stereo match in BOTH frames...
        if idx_l0 in dict0 and idx_l1 in dict1:
            idx_r0 = dict0[idx_l0]
            idx_r1 = dict1[idx_l1]
            four_way_indices.append((idx_l0, idx_r0, idx_l1, idx_r1))

    return four_way_indices


def prepare_4way_data(matches0, matches1, matches_t, kp0_l, kp1_l, pts3D_0):
    """
    Cross-references the matches to find features present in all 4 images.
    Extracts the synced 3D points (from Frame 0) and 2D pixels (from Frame 1).
    """
    track_indices = track_features_4way(matches0, matches1, matches_t)

    valid_3d_dict = {m.queryIdx: pts3D_0[i] for i, m in enumerate(matches0)}

    final_3D_pts = []
    final_2D_pts = []
    final_indices = []

    for idx_tuple in track_indices:
        idx_l0 = idx_tuple[0]
        idx_l1 = idx_tuple[2]

        if idx_l0 in valid_3d_dict:
            final_3D_pts.append(valid_3d_dict[idx_l0])
            final_2D_pts.append(kp1_l[idx_l1].pt)
            final_indices.append(idx_tuple)

    return np.array(final_3D_pts), np.array(final_2D_pts), final_indices


def generate_database_data(track_indices, ransac_mask, inliers_stereo0, kp0_l, inliers_stereo1, kp1_l):
    """
    Maps raw keypoint indices to the filtered database indices and constructs
    the exact dense match array required by TrackingDB.add_frame().
    """
    # 1. Build Raw -> Filtered index mappings for Frame 0
    is_valid_0 = [False] * len(kp0_l)
    for m in inliers_stereo0:
        is_valid_0[m.queryIdx] = True

    raw_to_filt_0 = {}
    filt_idx = 0
    for raw_idx, valid in enumerate(is_valid_0):
        if valid:
            raw_to_filt_0[raw_idx] = filt_idx
            filt_idx += 1

    # 2. Build Raw -> Filtered index mappings for Frame 1
    is_valid_1 = [False] * len(kp1_l)
    for m in inliers_stereo1:
        is_valid_1[m.queryIdx] = True

    raw_to_filt_1 = {}
    filt_idx = 0
    for raw_idx, valid in enumerate(is_valid_1):
        if valid:
            raw_to_filt_1[raw_idx] = filt_idx
            filt_idx += 1

    # 3. Initialize the dense arrays exactly as TrackingDB expects
    num_filtered_0 = len(raw_to_filt_0)
    db_matches = [cv2.DMatch() for _ in range(num_filtered_0)]  # Array of dummy matches
    db_inliers = [False] * num_filtered_0  # All False by default

    # 4. Populate only the features that survived 4-Way RANSAC
    for i, (idx_l0, idx_r0, idx_l1, idx_r1) in enumerate(track_indices):
        if ransac_mask[i]:
            # Translate raw indices to the database's filtered index space
            filt_0 = raw_to_filt_0[idx_l0]
            filt_1 = raw_to_filt_1[idx_l1]

            # Construct a match object that TrackingDB can safely read
            match = cv2.DMatch()
            match.queryIdx = filt_0
            match.trainIdx = filt_1
            match.distance = 1.0

            db_matches[filt_0] = match
            db_inliers[filt_0] = True

    return db_matches, db_inliers


def extract_safe_database_links(des, kp_l, kp_r, inliers_stereo):
    """
    Safely wraps TrackingDB.create_links by strictly enforcing index sorting.
    Prevents database coordinate desynchronization from distance-sorted matches.
    """
    # Force index sorting to satisfy TrackingDB's internal array assumptions
    sorted_inliers = sorted(inliers_stereo, key=lambda m: m.queryIdx)

    # Safely call the class method
    return TrackingDB.create_links(des, kp_l, kp_r, sorted_inliers)
