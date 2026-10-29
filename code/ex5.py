"""
ex5.py
Main execution file for VAN Exercise 5: Bundle Adjustment
"""

import random
import numpy as np
import gtsam

from tracking_database import TrackingDB
from dataset_utils import read_ground_truth, read_stereo_pair, read_cameras
from bundle_adjustment import create_gtsam_pose3, create_gtsam_calibration, select_keyframes, run_bundle_adjustment
from visualization import plot_gtsam_factor_vs_reprojection, plot_stereo_projection, plot_bundle_results, \
    plot_global_trajectory, plot_localization_error

DATA_PATH = '../dataset/sequences/00/'
POSES_PATH = '../dataset/poses/00.txt'
DB_NAME = "kitti_00_database"


def q5_1(db, poses_4x4, gtsam_calib):
    print("--- Q 5.1: GTSAM Coordinate Inversion and Reprojection Analysis ---")

    # 1. Select a valid track
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) >= 10]
    track_id = random.choice(valid_tracks)
    frames = db.frames(track_id)
    print(f"Selected Track ID: {track_id} | Track Length: {len(frames)} frames\n")

    # 2. Setup Noise Model & Symbols
    # Using Isotropic 1.0 pixel noise (equivalent to Diagonal([1.0, 1.0, 1.0]))
    sigma = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)
    q_key = gtsam.symbol('q', track_id)

    # 3. Triangulate the 3D landmark from the LAST frame
    last_frame = frames[-1]
    pose_last = create_gtsam_pose3(poses_4x4[last_frame])
    stereo_cam_last = gtsam.StereoCamera(pose_last, gtsam_calib)

    link_last = db.link(last_frame, track_id)
    meas_last = gtsam.StereoPoint2(link_last.x_left, link_last.x_right, link_last.y)

    pt3d_global = stereo_cam_last.backproject(meas_last)

    # 4. Project backwards through time
    distances, l2_errors, factor_errors = [], [], []

    for dist, fId in enumerate(reversed(frames)):
        # Initialize the Camera for this specific frame
        pose = create_gtsam_pose3(poses_4x4[fId])
        stereo_cam = gtsam.StereoCamera(pose, gtsam_calib)
        c_key = gtsam.symbol('c', fId)

        # Retrieve the database observation
        link = db.link(fId, track_id)
        meas_pt2d = gtsam.StereoPoint2(link.x_left, link.x_right, link.y)

        # --- Evaluate GTSAM Factor Error (Mahalanobis) ---
        factor = gtsam.GenericStereoFactor3D(meas_pt2d, sigma, c_key, q_key, gtsam_calib)
        values = gtsam.Values()
        values.insert(c_key, pose)
        values.insert(q_key, pt3d_global)
        fac_err = factor.error(values)

        # --- Evaluate L2 Reprojection Error (Pixels) ---
        proj_pt2d = stereo_cam.project(pt3d_global)
        dx = proj_pt2d.uL() - meas_pt2d.uL()
        dy = proj_pt2d.uR() - meas_pt2d.uR()
        dz = proj_pt2d.v() - meas_pt2d.v()
        l2_err = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

        distances.append(dist)
        l2_errors.append(l2_err)
        factor_errors.append(fac_err)

    # 5. Plot the comparison
    plot_gtsam_factor_vs_reprojection(distances, l2_errors, factor_errors, track_id)


def q5_3(db, poses_4x4, gtsam_calib):
    """Answers Section 5.3: Bundle Adjustment Window with a clean structural engine."""
    print("--- Q 5.3: Bundle Adjustment Window ---")

    # Define the execution block window
    keyframes = select_keyframes(poses_4x4, dist_threshold=5.0, max_frames=15)
    bundle_frames = list(range(keyframes[0], keyframes[1] + 1))
    print(f"Dynamic Keyframe Criterion: Distance > 5.0m or Frames >= 15")
    print(f"Bundle Window Bounds: Frames {bundle_frames[0]} to {bundle_frames[-1]}")

    init_vals, opt_vals, frame_keys, graph, metadata, anchor_factor = run_bundle_adjustment(db, bundle_frames,
                                                                                            poses_4x4, gtsam_calib,
                                                                                            pixel_sigma=1.0)

    # Profile Initial Graph Violations
    init_err = graph.error(init_vals)
    num_factors = graph.size()
    print(f"\n--- BEFORE OPTIMIZATION ---")
    print(f"Total Factors: {num_factors}")
    print(f"Graph Error: {init_err:.2f} | Average Error: {(init_err / num_factors):.2f}")

    # Extract the absolute worst outlier factor before correction
    worst_factor_data = max(metadata, key=lambda meta: meta['factor'].error(init_vals))
    wf_fId, wf_tId = worst_factor_data['fId'], worst_factor_data['tId']
    meas = worst_factor_data['meas']

    print(f"\nMax Initial Error Factor: Frame c={wf_fId}, Track q={wf_tId}")
    print(f"Initial Factor Error: {worst_factor_data['factor'].error(init_vals):.2f}")

    # --- Calculate & Print Distances BEFORE BA ---
    init_cam = gtsam.StereoCamera(init_vals.atPose3(worst_factor_data['c_key']), gtsam_calib)
    init_proj = init_cam.project(init_vals.atPoint3(worst_factor_data['q_key']))

    dist_l_init = np.sqrt((init_proj.uL() - meas.uL()) ** 2 + (init_proj.v() - meas.v()) ** 2)
    dist_r_init = np.sqrt((init_proj.uR() - meas.uR()) ** 2 + (init_proj.v() - meas.v()) ** 2)
    print(f"Distance from Measurements -> Left: {dist_l_init:.2f}px, Right: {dist_r_init:.2f}px")

    img_l, img_r = read_stereo_pair(DATA_PATH, wf_fId)
    plot_stereo_projection(img_l, img_r, meas.uL(), meas.uR(), meas.v(), init_proj.uL(), init_proj.uR(), init_proj.v(),
                           f"BEFORE BA: Frame {wf_fId}, Track {wf_tId}")

    # Profile Post-Optimization Gains
    final_err = graph.error(opt_vals)
    print(f"\n--- AFTER OPTIMIZATION ---")
    print(f"Total Graph Error: {final_err:.2f} | Average Error: {(final_err / num_factors):.2f}")
    print(f"\nRe-evaluating Factor: Frame c={wf_fId}, Track q={wf_tId}")
    print(f"Corrected Target Factor Error: {worst_factor_data['factor'].error(opt_vals):.2f}")

    # ---Calculate & Print Distances AFTER BA ---
    opt_cam = gtsam.StereoCamera(opt_vals.atPose3(worst_factor_data['c_key']), gtsam_calib)
    opt_proj = opt_cam.project(opt_vals.atPoint3(worst_factor_data['q_key']))

    dist_l_opt = np.sqrt((opt_proj.uL() - meas.uL()) ** 2 + (opt_proj.v() - meas.v()) ** 2)
    dist_r_opt = np.sqrt((opt_proj.uR() - meas.uR()) ** 2 + (opt_proj.v() - meas.v()) ** 2)
    print(f"Distance from Measurements -> Left: {dist_l_opt:.2f}px, Right: {dist_r_opt:.2f}px\n")

    plot_stereo_projection(img_l, img_r, meas.uL(), meas.uR(), meas.v(), opt_proj.uL(), opt_proj.uR(), opt_proj.v(),
                           f"AFTER BA: Frame {wf_fId}, Track {wf_tId}")

    # Generate Marginal Covariances and Plot the clean, tightly bound figures
    marginals = gtsam.Marginals(graph, opt_vals)
    plot_bundle_results(init_vals, opt_vals, frame_keys, marginals)
    print("FINISHED: --- SECTION 5.3 ---\n")


def q5_4(db, poses_4x4, gtsam_calib):
    """Answers Section 5.4: Global Sliding Window and Map Chaining."""
    print("--- Q 5.4: Global Sliding Window Execution ---")

    # 1. Extract all Keyframes dynamically
    keyframes = select_keyframes(poses_4x4, dist_threshold=5.0, max_frames=15)
    print(f"Total Keyframes Extracted: {len(keyframes)}")

    # 2. Storage for the chained global trajectory
    # Anchor the first frame exactly to the Ground Truth origin
    global_poses = {keyframes[0]: create_gtsam_pose3(poses_4x4[keyframes[0]])}
    global_landmarks = []

    localization_errors = [0.0]
    processed_keyframes = [keyframes[0]]

    # 3. Iterate through every bundle window
    for i in range(len(keyframes) - 1):
        kf1 = keyframes[i]
        kf2 = keyframes[i + 1]
        bundle_frames = list(range(kf1, kf2 + 1))

        if i % 10 == 0:
            print(f"Optimizing Bundle {i + 1}/{(len(keyframes) - 1)} [Frames {kf1}-{kf2}]...")

        # Run relative Bundle Adjustment
        init_vals, opt_vals, _, graph, _, anchor_factor = run_bundle_adjustment(
            db, bundle_frames, poses_4x4, gtsam_calib, pixel_sigma=1.0, make_relative=True
        )

        # 4. Answer PDF specific questions on the LAST bundle
        if i == len(keyframes) - 2:
            print("\n--- LAST BUNDLE THEORETICAL ANALYSIS ---")

            # Print position of the first frame (Anchor) after optimization
            anchor_key = gtsam.symbol('c', kf1)
            optimized_anchor_pose = opt_vals.atPose3(anchor_key)
            print("Optimized Anchor Position (X, Y, Z):")
            print(optimized_anchor_pose.translation())

            # Print the anchoring factor final error
            anchor_err = anchor_factor.error(opt_vals)
            print(f"Anchoring Factor Final Error: {anchor_err}")
            print("----------------------------------------\n")

        # 5. Extract the relative transformation T_{kf1 -> kf2}
        relative_pose = opt_vals.atPose3(gtsam.symbol('c', kf2))

        # 6. Chain it to the Global Map: T_global_kf2 = T_global_kf1 * T_rel
        global_poses[kf2] = global_poses[kf1].compose(relative_pose)

        # Transform landmarks from local bundle coordinates to global coordinates
        for key in opt_vals.keys():
            if gtsam.symbolChr(key) == ord('q'):
                local_pt = opt_vals.atPoint3(key)
                global_pt = global_poses[kf1].transformFrom(local_pt)
                global_landmarks.append(global_pt)

        # Calculate Euclidean Localization Error vs Ground Truth
        gt_pose = create_gtsam_pose3(poses_4x4[kf2])
        gt_t = gt_pose.translation()
        est_t = global_poses[kf2].translation()
        error_m = np.sqrt((gt_t[0] - est_t[0]) ** 2 + (gt_t[1] - est_t[1]) ** 2 + (gt_t[2] - est_t[2]) ** 2)

        localization_errors.append(error_m)
        processed_keyframes.append(kf2)

    # 7. Plotting Final Results
    print("\nOptimization Complete. Generating Final Trajectory Maps...")
    gt_keyframe_poses = [poses_4x4[kf] for kf in processed_keyframes]
    plot_global_trajectory(gt_keyframe_poses, global_poses, global_landmarks)
    plot_localization_error(processed_keyframes, localization_errors)
    print("FINISHED: --- SECTION 5.4 ---")


def main():
    db = TrackingDB()
    db.load(DB_NAME)

    K, _, M2 = read_cameras(DATA_PATH)
    gtsam_calib = create_gtsam_calibration(K, M2)
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)

    # Q 5.1
    q5_1(db, poses_4x4, gtsam_calib)

    # Q 5.3
    q5_3(db, poses_4x4, gtsam_calib)

    # Q 5.4
    q5_4(db, poses_4x4, gtsam_calib)


if __name__ == "__main__":
    main()
