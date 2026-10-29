"""
Local Sliding-Window Back-End (Bundle Adjustment)
=================================================

Manages the local optimization back-end of the SLAM system.
To contain computational complexity and floating-point errors, this module dynamically
extracts local spatial windows (keyframes). Utilizing the GTSAM library, it constructs
local non-linear factor graphs and performs joint Bundle Adjustment on both camera
poses and 3D spatial landmarks to mitigate local tracking drift.

Key Responsibilities:
- Distance-based dynamic keyframe selection to guarantee sufficient stereo parallax.
- GTSAM Factor Graph formulation using Stereo Point Multi-view factors.
- Strict Prior anchoring to eliminate Gauge Freedom in local coordinate windows.
- Levenberg-Marquardt optimization of highly coupled, relative coordinate systems.
"""

import cv2
import gtsam
import numpy as np
from geometry import compute_reprojection_error


def calculate_backward_reprojection(db, track_id, frames, poses_4x4, P1_local, P2_local):
    """
    Triangulates from the last frame and projects backward through time
    using a pre-loaded array of 4x4 ground truth matrices.
    """
    last_frame = frames[-1]

    # 1. Triangulate from the LAST frame
    T_w_c_last = poses_4x4[last_frame]
    P_left_last = P1_local @ T_w_c_last
    P_right_last = P2_local @ T_w_c_last

    link_last = db.link(last_frame, track_id)
    pt1 = np.array([[link_last.x_left, link_last.y]], dtype=np.float32)
    pt2 = np.array([[link_last.x_right, link_last.y]], dtype=np.float32)

    pt4D = cv2.triangulatePoints(P_left_last, P_right_last, pt1.T, pt2.T)
    pt3D_world = (pt4D[:3] / pt4D[3]).flatten()

    # 2. Project backward through time
    errors_left, errors_right, distances = [], [], []

    for i, fId in enumerate(reversed(frames)):
        # Grab the matrix instantly from the pre-loaded array
        T_w_c_i = poses_4x4[fId]
        P_left_i = P1_local @ T_w_c_i
        P_right_i = P2_local @ T_w_c_i

        link_i = db.link(fId, track_id)

        err_l = compute_reprojection_error(pt3D_world, P_left_i, (link_i.x_left, link_i.y))
        err_r = compute_reprojection_error(pt3D_world, P_right_i, (link_i.x_right, link_i.y))

        errors_left.append(err_l)
        errors_right.append(err_r)
        distances.append(i)

    return distances, errors_left, errors_right


def create_gtsam_pose3(T_w_c):
    T_c_w = np.linalg.inv(T_w_c)
    R = T_c_w[:3, :3]
    t = T_c_w[:3, 3]
    return gtsam.Pose3(gtsam.Rot3(R), gtsam.Point3(t))


def create_gtsam_calibration(K, M2):
    """Constructs calibration cleanly matching the class example."""
    baseline = -M2[0, 3]
    return gtsam.Cal3_S2Stereo(K[0, 0], K[1, 1], K[0, 1], K[0, 2], K[1, 2], baseline)


def select_keyframes(poses_4x4, dist_threshold=10.0, max_frames=15):
    """
    Dynamically selects keyframes based on Euclidean distance travelled.
    A new keyframe is created when the vehicle moves > dist_threshold,
    OR as a failsafe if max_frames have elapsed to ensure the window
    doesn't get too large.
    """
    keyframes = [0]  # Frame 0 is always the first keyframe
    last_kf_pose = poses_4x4[0]

    for i in range(1, len(poses_4x4)):
        current_pose = poses_4x4[i]

        # Calculate Euclidean distance between current frame and last keyframe
        dx = current_pose[0, 3] - last_kf_pose[0, 3]
        dy = current_pose[1, 3] - last_kf_pose[1, 3]
        dz = current_pose[2, 3] - last_kf_pose[2, 3]
        dist = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        frames_elapsed = i - keyframes[-1]

        if dist >= dist_threshold or frames_elapsed >= max_frames:
            keyframes.append(i)
            last_kf_pose = current_pose

    return keyframes


def run_bundle_adjustment(db, bundle_frames, poses_4x4, gtsam_calib, pixel_sigma=1.0, make_relative=True):
    """
    Constructs and optimizes a GTSAM Factor Graph for an arbitrary window of frames.
    If make_relative=True, transforms all initial estimates into the local coordinate
    frame of the first bundle frame (Anchor).
    """
    graph = gtsam.NonlinearFactorGraph()
    initial_values = gtsam.Values()

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(3, pixel_sigma)
    # The Prior anchors the first frame to prevent Gauge Freedom explosion
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))

    # Identify valid tracks (length >= 2 in window)
    tracks_in_window = {}
    for fId in bundle_frames:
        for tId in db.tracks(fId):
            tracks_in_window.setdefault(tId, []).append(fId)
    valid_tracks = {tId: fIds for tId, fIds in tracks_in_window.items() if len(fIds) >= 2}

    frame_keys = []
    factor_metadata = []

    # Extract the absolute global pose of the Anchor frame
    anchor_global_pose = create_gtsam_pose3(poses_4x4[bundle_frames[0]])

    # 1. Instantiate Camera Variables & Priors
    for fId in bundle_frames:
        c_key = gtsam.symbol('c', fId)
        frame_keys.append(c_key)

        pose_global = create_gtsam_pose3(poses_4x4[fId])

        if make_relative:
            # Transform global pose into the local coordinate system of the anchor
            pose_local = anchor_global_pose.between(pose_global)
        else:
            pose_local = pose_global

        initial_values.insert(c_key, pose_local)

        # Add the anchoring factor to the first frame
        if fId == bundle_frames[0]:
            prior_factor = gtsam.PriorFactorPose3(c_key, pose_local, prior_noise)
            graph.add(prior_factor)
            anchor_factor = prior_factor  # Save for error analysis in 5.4

    # 2. Instantiate Spatial Landmarks & Multi-view Stereo Factors
    for tId, fIds in valid_tracks.items():
        q_key = gtsam.symbol('q', tId)

        last_fId = fIds[-1]
        pose_last_global = create_gtsam_pose3(poses_4x4[last_fId])

        if make_relative:
            pose_last = anchor_global_pose.between(pose_last_global)
        else:
            pose_last = pose_last_global

        stereo_cam = gtsam.StereoCamera(pose_last, gtsam_calib)
        link = db.link(last_fId, tId)
        meas = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)
        pt3d = stereo_cam.backproject(meas)
        initial_values.insert(q_key, pt3d)

        for fId in fIds:
            c_key = gtsam.symbol('c', fId)
            lk = db.link(fId, tId)
            meas_pt2d = gtsam.StereoPoint2(lk.x_left, lk.x_right, lk.y)

            factor = gtsam.GenericStereoFactor3D(meas_pt2d, measurement_noise, c_key, q_key, gtsam_calib)
            graph.add(factor)
            factor_metadata.append({
                'factor': factor, 'c_key': c_key, 'q_key': q_key,
                'meas': meas_pt2d, 'fId': fId, 'tId': tId
            })

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values)
    optimized_values = optimizer.optimize()

    return initial_values, optimized_values, frame_keys, graph, factor_metadata, anchor_factor
