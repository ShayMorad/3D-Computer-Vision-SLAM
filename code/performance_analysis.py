"""
performance_analysis.py
Master Execution Pipeline for Performance Analysis & Benchmarking.

Stages:
1. Infrastructure & Database Initialization
2. Front-End Tracking Statistics & Connectivity Analysis
3. PnP Visual Odometry & Convergence Analysis
4. Local Bundle Adjustment & Sequential Pose Graph Optimization
5. Online Loop Closure Candidate Search & Global Optimization
6. Master Trajectory Plotting & Error Benchmarks
7. Uncertainty Trace Isolation & Loop Closure Statistics
"""
import matplotlib.pyplot as plt

from geometry import triangulate_points_cv2
from pose_graph import generate_sequential_measurements, build_pose_graph
from tracking_database import TrackingDB
from visualization import plot_sequence_stat, plot_histogram, format_plot
import os
import cv2
import numpy as np
import gtsam
from dataset_utils import read_ground_truth, read_cameras
from bundle_adjustment import create_gtsam_calibration, select_keyframes, run_bundle_adjustment
from loop_closure import build_initial_adjacency_list, get_shortest_paths_covariances, filter_candidates_by_mahalanobis
from ex7 import q7_2, q7_3, q7_4, REL_COV_SCALE, COV_SCALE

DATA_PATH = '../dataset/sequences/00/'
POSES_PATH = '../dataset/poses/00.txt'
DB_NAME = "kitti_00_database"


# --- CHUNK 1 TRACKING STATISTICS & FEATURE CONNECTIVITY ---

def analyze_tracking_statistics(db):
    """
    Addresses: General Tracking Stats, Matches/Inliers per frame, Connectivity, and Track Lengths.
    """
    print("--- 1. Tracking Statistics & Frame-Level Analysis ---")

    # 1. Filter out trivial tracks (length 1) based on Ex 4 logic
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) > 1]
    valid_tracks_set = set(valid_tracks)
    track_lengths = [len(db.frames(tId)) for tId in valid_tracks]

    # Calculate frame links (outgoing tracks)
    tracks_per_frame = []
    connectivity = []
    total_matches = []
    inlier_percentages = []
    inlier_counts = []

    for fId in range(db.frame_num()):
        frame_tracks = db.tracks(fId)
        valid_count = sum(1 for tId in frame_tracks if tId in valid_tracks_set)
        tracks_per_frame.append(valid_count)

        # Matches & Inliers Logic
        # db.features(fId) contains the raw matched features before rigorous RANSAC
        features = db.features(fId)
        num_raw_matches = features.shape[0] if features is not None else 0
        total_matches.append(num_raw_matches)

        if num_raw_matches > 0:
            # Inliers are features that were successfully assigned a Track ID (!= -1)
            num_inliers = sum(1 for tId in db.frameId_to_trackIds_list[fId] if tId != -1)
            inlier_counts.append(num_inliers)
            inlier_percentages.append((num_inliers / num_raw_matches) * 100)
        else:
            inlier_counts.append(0)
            inlier_percentages.append(0)

        # Connectivity: Intersection of tracks between frame i and frame i+1
        if fId < db.frame_num() - 1:
            tracks_current = set(db.tracks(fId))
            tracks_next = set(db.tracks(fId + 1))
            connectivity.append(len(tracks_current.intersection(tracks_next)))

    # --- Print Text Statistics ---
    print(f"Total number of tracks: {len(valid_tracks)}")
    print(f"Number of frames: {db.frame_num()}")
    print(f"Mean track length: {np.mean(track_lengths):.2f} frames")
    print(f"Mean number of frame links (tracks per image): {np.mean(tracks_per_frame):.2f}")

    # --- Generate Graphs ---
    frames_x = list(range(db.frame_num()))

    # Graph A: Matches and Inliers per frame (Overlaid for easy comparison)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames_x, total_matches, label='Total Matches (Post-Ratio Test)', color='slategray', alpha=0.8,
            linewidth=1.5)
    ax.plot(frames_x, inlier_counts, label='Inliers (Post-RANSAC)', color='dodgerblue', alpha=0.9, linewidth=1.5)
    format_plot(ax, 'Number of Matches and Inliers per Frame', 'Frame Index', 'Count')
    ax.legend(loc='upper right')
    ax.set_xlim(0, len(frames_x))
    plt.show()

    # Graph B: Percentage of Inliers
    plot_sequence_stat(frames_x, inlier_percentages, 'Percentage of Inliers per Frame', 'Frame Index', 'Inliers (%)',
                       mean_line=True)

    # Graph C: Connectivity
    frames_conn = list(range(len(connectivity)))
    plot_sequence_stat(frames_conn, connectivity, 'Connectivity (Outgoing Tracks per Frame)', 'Frame Index',
                       'Outgoing Tracks', mean_line=True)

    # Graph D: Track Length Histogram
    plot_histogram(track_lengths, bins=50, title='Track Length Histogram', xlabel='Track Length (Frames)',
                   ylabel='Number of Tracks', log_scale=True)


# --- CHUNK 2 TRAJECTORY ESTIMATION & OPTIMIZATION CONVERGENCE ---

def generate_pnp_trajectory_from_db(db, K, P1, P2):
    """
    Rapidly generates the sequential PnP trajectory by querying the TrackingDB,
    bypassing the need to re-run slow feature extraction.
    Returns a dictionary of 4x4 KITTI-formatted matrices (T_w_c).
    """
    print("Generating PnP Trajectory directly from TrackingDB...")
    pnp_poses = {0: np.eye(4)}
    current_pose_w_c = np.eye(4)

    for i in range(1, db.frame_num()):
        # Find tracks that exist in both frame i-1 and frame i
        tracks_prev = set(db.tracks(i - 1))
        tracks_curr = set(db.tracks(i))
        common_tracks = list(tracks_prev.intersection(tracks_curr))
        common_tracks = [t for t in common_tracks if t != -1]  # Remove NO_ID

        if len(common_tracks) < 6:
            pnp_poses[i] = current_pose_w_c
            continue

        # Extract 2D points from DB for frame i-1
        pts1 = np.array([[db.link(i - 1, t).x_left, db.link(i - 1, t).y] for t in common_tracks])
        pts2 = np.array([[db.link(i - 1, t).x_right, db.link(i - 1, t).y] for t in common_tracks])

        # Triangulate 3D points in the local coordinate frame of i-1
        pts3D_c_prev = triangulate_points_cv2(pts1, pts2, P1, P2)

        # Extract corresponding 2D points in frame i
        pts2D_curr = np.array([[db.link(i, t).x_left, db.link(i, t).y] for t in common_tracks])

        # Solve PnP using RANSAC for robust estimation
        success, rvec, tvec, _ = cv2.solvePnPRansac(pts3D_c_prev, pts2D_curr, K, None, flags=cv2.SOLVEPNP_P3P)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = tvec.flatten()

            # Kinematic Chaining: T_w_ci = T_ci_c(i-1) @ T_w_c(i-1)
            current_pose_w_c = T_rel @ current_pose_w_c

        pnp_poses[i] = current_pose_w_c

    return pnp_poses


def extract_trajectory_coords(poses_data, frames):
    """Helper to safely extract the global C = -R^T * t coordinates for plotting.
    Safely handles both lists (Ground Truth) and dictionaries (Estimated Poses)."""
    coords_x, coords_z = [], []
    for f in frames:
        try:
            pose = poses_data[f]
            C = -pose[:3, :3].T @ pose[:3, 3]
            coords_x.append(C[0])
            coords_z.append(C[2])
        except (KeyError, IndexError):
            pass
    return coords_x, coords_z


def plot_master_trajectory(poses_4x4, pnp_poses_dict, ba_poses_dict, pg_poses_dict, keyframes):
    """
    Addresses: The trajectory - bird eye view of PnP, Bundle, Pose Graph, and Ground Truth.
    """
    print("\n--- Generating Master Trajectory Plot ---")
    fig, ax = plt.subplots(figsize=(10, 10))

    gt_x, gt_z = extract_trajectory_coords(poses_4x4, keyframes)
    pnp_x, pnp_z = extract_trajectory_coords(pnp_poses_dict, keyframes)
    ba_x, ba_z = extract_trajectory_coords(ba_poses_dict, keyframes)
    pg_x, pg_z = extract_trajectory_coords(pg_poses_dict, keyframes)

    # Updated to distinct solid lines
    ax.plot(gt_x, gt_z, color='green', linestyle='-', linewidth=2.5, label='Ground Truth')
    ax.plot(pnp_x, pnp_z, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='PnP (Dead Reckoning)')
    ax.plot(ba_x, ba_z, color='red', linestyle='-', linewidth=1.5, label='Bundle Adjustment (Local Window)')

    if pg_x:  # Only plot if Pose Graph data is provided
        ax.plot(pg_x, pg_z, color='blue', linestyle='-', linewidth=2, label='Pose Graph (Global)')

    format_plot(ax, "Global Trajectory: Multi-Stage Estimation vs Ground Truth", "X (East) [m]", "Z (North) [m]",
                equal_axes=True)
    ax.legend(loc='best')
    plt.show()


def analyze_optimization_errors(db, poses_4x4, gtsam_calib):
    """
    Addresses: Optimization error (Mean Factor Error) & Median Projection Error per Bundle Window.
    """
    print("\n--- 2. Optimization Convergence Analysis ---")
    keyframes = select_keyframes(poses_4x4, dist_threshold=5.0, max_frames=15)

    mean_factor_init, mean_factor_opt = [], []
    median_proj_init, median_proj_opt = [], []
    kf_indices = []

    for i in range(len(keyframes) - 1):
        kf_0, kf_k = keyframes[i], keyframes[i + 1]
        bundle_frames = list(range(kf_0, kf_k + 1))

        # Run local BA (making it relative to the anchor frame)
        init_vals, opt_vals, _, graph, factor_metadata, _ = run_bundle_adjustment(
            db, bundle_frames, poses_4x4, gtsam_calib, pixel_sigma=1.0, make_relative=True
        )

        num_factors = graph.size()
        kf_indices.append(kf_0)

        # 1. Mean Factor Graph Error (Total Error / # Factors)
        mean_factor_init.append(graph.error(init_vals) / num_factors)
        mean_factor_opt.append(graph.error(opt_vals) / num_factors)

        # 2. Median Projection Error (Pixels)
        proj_errs_init = []
        proj_errs_opt = []

        for meta in factor_metadata:
            meas = meta['meas']

            # Initial Projection
            cam_init = gtsam.StereoCamera(init_vals.atPose3(meta['c_key']), gtsam_calib)
            proj_init = cam_init.project(init_vals.atPoint3(meta['q_key']))
            dist_init = np.sqrt((proj_init.uL() - meas.uL()) ** 2 + (proj_init.v() - meas.v()) ** 2)
            proj_errs_init.append(dist_init)

            # Optimized Projection
            cam_opt = gtsam.StereoCamera(opt_vals.atPose3(meta['c_key']), gtsam_calib)
            proj_opt = cam_opt.project(opt_vals.atPoint3(meta['q_key']))
            dist_opt = np.sqrt((proj_opt.uL() - meas.uL()) ** 2 + (proj_opt.v() - meas.v()) ** 2)
            proj_errs_opt.append(dist_opt)

        median_proj_init.append(np.median(proj_errs_init))
        median_proj_opt.append(np.median(proj_errs_opt))

    # --- Generate Graphs ---
    # Graph E: Mean Factor Error
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(kf_indices, mean_factor_init, 'r-', alpha=0.6, label='Initial Error (Pre-BA)')
    ax.plot(kf_indices, mean_factor_opt, 'b-', linewidth=2, label='Optimized Error (Post-BA)')
    format_plot(ax, 'Mean Factor Error Before and After Bundle Optimization', 'Bundle Starting at Frame Idx',
                'Mean Factor Error')
    ax.legend()
    plt.show()

    # Graph F: Median Projection Error
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(kf_indices, median_proj_init, 'r-', alpha=0.6, label='Initial Median Projection (Pixels)')
    ax.plot(kf_indices, median_proj_opt, 'b-', linewidth=2, label='Optimized Median Projection (Pixels)')
    format_plot(ax, 'Median Projection Error Before and After Bundle Optimization', 'Bundle Starting at Frame Idx',
                'Error (Pixels)')
    ax.legend()
    plt.show()


def analyze_reprojection_vs_distance(db, poses_4x4, P1, P2):
    """
    Addresses: Projection error vs track length for PnP estimation.
    We approximate this by sampling the longest 100 tracks to get a smooth average curve.
    """
    print("\n--- 3. Reprojection Error vs Distance (PnP) ---")
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) >= 20]

    # Select a representative subset to average the noise
    import random
    subset_tracks = random.sample(valid_tracks, min(100, len(valid_tracks)))

    avg_left_errs = np.zeros(50)
    avg_right_errs = np.zeros(50)
    counts = np.zeros(50)

    for tId in subset_tracks:
        frames = db.frames(tId)
        from bundle_adjustment import calculate_backward_reprojection
        distances, err_l, err_r = calculate_backward_reprojection(db, tId, frames, poses_4x4, P1, P2)

        for d, el, er in zip(distances, err_l, err_r):
            if d < 50:
                avg_left_errs[d] += el
                avg_right_errs[d] += er
                counts[d] += 1

    # Normalize by counts
    valid_mask = counts > 0
    avg_left_errs[valid_mask] /= counts[valid_mask]
    avg_right_errs[valid_mask] /= counts[valid_mask]

    # Graph G: Reprojection Error vs Track Length
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(np.arange(50)[valid_mask], avg_left_errs[valid_mask], label='Left', color='slategray')
    ax.plot(np.arange(50)[valid_mask], avg_right_errs[valid_mask], label='Right', color='goldenrod')
    format_plot(ax, 'PnP - Projection Error vs Track Length', 'Distance from Reference (Frames)',
                'Projection Error (Pixels)')
    ax.legend()
    plt.show()


# --- KITTI BENCHMARK MATH HELPERS (From Appendix) ---

def get_kitti_error(est_a, est_b, gt_a, gt_b):
    """
    Implements the strict KITTI relative error math from the Appendix.
    est_displacement(a,b) = est_a^-1 * est_b
    gt_displacement(a,b) = gt_a^-1 * gt_b
    error(a,b) = est_displacement^-1 * gt_displacement
    """
    est_disp = np.linalg.inv(est_a) @ est_b
    gt_disp = np.linalg.inv(gt_a) @ gt_b

    # E = Delta C(est_disp, gt_disp) = est_disp^-1 * gt_disp
    E = np.linalg.inv(est_disp) @ gt_disp

    # Location error norm
    loc_err = np.linalg.norm(E[:3, 3])

    # Angle error via Rodrigues
    rvec, _ = cv2.Rodrigues(E[:3, :3])
    ang_err = np.linalg.norm(rvec) * 180.0 / np.pi

    return loc_err, ang_err


def compute_gt_cumulative_distances(gt_poses):
    """Calculates the cumulative Ground Truth distance for the denominator."""
    cumulative = [0.0]
    total = 0.0
    for i in range(1, len(gt_poses)):
        gt_disp = np.linalg.inv(gt_poses[i - 1]) @ gt_poses[i]
        total += np.linalg.norm(gt_disp[:3, 3])
        cumulative.append(total)
    return np.array(cumulative)


# --- CHUNK 3 ANALYSIS FUNCTIONS ---

def analyze_absolute_errors(gt_poses, est_poses_dict, title_prefix):
    """
    Addresses: Absolute estimation error (X, Y, Z, Norm, Angle)
    Expects est_poses_dict to map frame_id -> 4x4 transformation matrix.
    """
    print(f"\n--- Calculating Absolute Errors: {title_prefix} ---")

    frames = sorted(list(est_poses_dict.keys()))
    x_err, y_err, z_err, norm_err, ang_err = [], [], [], [], []

    for f in frames:
        gt = gt_poses[f]
        est = est_poses_dict[f]

        # Absolute error = Delta C(est, gt) = est^-1 * gt
        E = np.linalg.inv(est) @ gt

        x_err.append(abs(E[0, 3]))
        y_err.append(abs(E[1, 3]))
        z_err.append(abs(E[2, 3]))
        norm_err.append(np.linalg.norm(E[:3, 3]))

        rvec, _ = cv2.Rodrigues(E[:3, :3])
        ang_err.append(np.linalg.norm(rvec) * 180.0 / np.pi)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Location Errors
    ax1.plot(frames, x_err, label='X Error', alpha=0.7)
    ax1.plot(frames, y_err, label='Y Error', alpha=0.7)
    ax1.plot(frames, z_err, label='Z Error', alpha=0.7)
    ax1.plot(frames, norm_err, label='Total Norm', color='k', linewidth=2)
    format_plot(ax1, f'{title_prefix}: Absolute Location Error', 'Frame', 'Error (m)')
    ax1.legend()

    # Angle Error
    ax2.plot(frames, ang_err, label='Angle Error', color='darkred', linewidth=2)
    format_plot(ax2, f'{title_prefix}: Absolute Angle Error', 'Frame', 'Error (deg)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def analyze_kitti_relative_errors(gt_poses, est_poses_dict, title_prefix):
    """
    Addresses: Relative estimation error over sub-sections (100, 400, 800)
    """
    print(f"\n--- Calculating KITTI Sub-Sequence Errors: {title_prefix} ---")

    cumulative_dists = compute_gt_cumulative_distances(gt_poses)
    frames = sorted(list(est_poses_dict.keys()))

    lengths = [100, 400, 800]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    for L in lengths:
        loc_errs_L = []
        ang_errs_L = []
        valid_frames = []

        for i in range(len(frames)):
            f_start = frames[i]
            f_end_target = f_start + L

            # Find the closest keyframe to the target length
            # (Requires exact frame matching or closest approximation if keyframes skip)
            future_frames = [f for f in frames if f >= f_end_target]
            if not future_frames:
                break

            f_end = future_frames[0]

            # Distance traveled in GT
            dist_traveled = cumulative_dists[f_end] - cumulative_dists[f_start]
            if dist_traveled < 1.0:  # Prevent division by zero
                continue

            loc_err, ang_err = get_kitti_error(
                est_poses_dict[f_start], est_poses_dict[f_end],
                gt_poses[f_start], gt_poses[f_end]
            )

            # Normalize by distance (Location as %, Angle as deg/m)
            loc_errs_L.append((loc_err / dist_traveled) * 100.0)
            ang_errs_L.append(ang_err / dist_traveled)
            valid_frames.append(f_start)

        if loc_errs_L:
            avg_loc = np.mean(loc_errs_L)
            avg_ang = np.mean(ang_errs_L)
            print(
                f"[{title_prefix}] Sequence Length {L} -> Avg Loc Err: {avg_loc:.2f}%, Avg Ang Err: {avg_ang:.4f} deg/m")

            ax1.plot(valid_frames, loc_errs_L, label=f'Length {L} (Avg: {avg_loc:.2f}%)')
            ax2.plot(valid_frames, ang_errs_L, label=f'Length {L} (Avg: {avg_ang:.4f} deg/m)')

    format_plot(ax1, f'{title_prefix}: Relative Location Error over Sub-sections', 'Frame Number',
                'Translation Error (%)')
    ax1.legend()

    format_plot(ax2, f'{title_prefix}: Relative Angle Error over Sub-sections', 'Frame Number',
                'Rotation Error (deg/m)')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def analyze_consecutive_relative_errors(gt_poses, est_poses_dict, title_prefix, keyframes):
    """
    Addresses: Relative error of every two consecutive keyframes (Norm and Angle).
    """
    print(f"\n--- Calculating Consecutive Relative Errors: {title_prefix} ---")

    loc_errs, ang_errs, frames_x = [], [], []

    for i in range(len(keyframes) - 1):
        f_a, f_b = keyframes[i], keyframes[i + 1]

        if f_a not in est_poses_dict or f_b not in est_poses_dict:
            continue

        loc_err, ang_err = get_kitti_error(
            est_poses_dict[f_a], est_poses_dict[f_b],
            gt_poses[f_a], gt_poses[f_b]
        )

        loc_errs.append(loc_err)
        ang_errs.append(ang_err)
        frames_x.append(f_b)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(frames_x, loc_errs, color='teal', linewidth=1.5)
    format_plot(ax1, f'{title_prefix}: Consecutive Relative Location Error', 'Frame', 'Error [m]')

    ax2.plot(frames_x, ang_errs, color='maroon', linewidth=1.5)
    format_plot(ax2, f'{title_prefix}: Consecutive Relative Angle Error', 'Frame', 'Error [deg]')

    plt.tight_layout()
    plt.show()


def analyze_loop_closure_stats(successful_loops):
    """
    Addresses: Number of matches and Inlier percentage per successful loop closure.
    """
    print("\n--- Loop Closure Match Statistics ---")
    for idx, loop in enumerate(successful_loops):
        f_current = loop['current_kf']
        f_candidate = loop['candidate_kf']

        total_matches = len(loop['match_data']['matches_4way'])
        inliers = sum(loop['match_data']['mask'])
        percentage = (inliers / total_matches) * 100 if total_matches > 0 else 0

        print(f"Loop {idx + 1} [Frame {f_current} -> {f_candidate}]:")
        print(f"  Total 4-Way Matches: {total_matches}")
        print(f"  Consensus Inliers:   {inliers}")
        print(f"  Inlier Percentage:   {percentage:.2f}%\n")


def plot_full_uncertainty_analysis(base_marginals, final_marginals, keyframes):
    """
    Addresses: Uncertainty size vs keyframe for Location AND Angle, with and without LC.
    """
    base_loc, base_ang = [], []
    final_loc, final_ang = [], []

    for kf in keyframes:
        key = gtsam.symbol('c', kf)

        # Base Graph (Without LC)
        if base_marginals:
            cov_base = base_marginals.marginalCovariance(key)
            base_ang.append(np.trace(cov_base[0:3, 0:3]))  # Rotation block
            base_loc.append(np.trace(cov_base[3:6, 3:6]))  # Translation block

        # Final Graph (With LC)
        if final_marginals:
            cov_final = final_marginals.marginalCovariance(key)
            final_ang.append(np.trace(cov_final[0:3, 0:3]))
            final_loc.append(np.trace(cov_final[3:6, 3:6]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Location Uncertainty
    ax1.plot(keyframes, base_loc, 'r--', label='Without Loop Closures')
    ax1.plot(keyframes, final_loc, 'b-', label='With Loop Closures')
    format_plot(ax1, "Location Uncertainty Size (Trace)", "Keyframe Index", "Trace [$m^2$]")
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, linestyle=':', which='both')

    # Angle Uncertainty
    ax2.plot(keyframes, base_ang, 'r--', label='Without Loop Closures')
    ax2.plot(keyframes, final_ang, 'b-', label='With Loop Closures')
    format_plot(ax2, "Angle Uncertainty Size (Trace)", "Keyframe Index", "Trace [$rad^2$]")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, linestyle=':', which='both')

    plt.tight_layout()
    plt.show()


def analyze_bundle_reprojection_vs_distance(db, poses_4x4, gtsam_calib):
    """
    Addresses: Projection error vs distance for Bundle estimation.
    Calculates the optimized projection error as a function of distance from
    the bundle's reference frame (the 1st frame of the window).
    """
    print("\n--- Calculating Bundle Reprojection Error vs Distance ---")
    keyframes = select_keyframes(poses_4x4, dist_threshold=5.0, max_frames=15)

    # Store errors grouped by distance from reference frame (0 to 15)
    left_errors_by_dist = {i: [] for i in range(20)}
    right_errors_by_dist = {i: [] for i in range(20)}

    for i in range(len(keyframes) - 1):
        kf_0, kf_k = keyframes[i], keyframes[i + 1]
        bundle_frames = list(range(kf_0, kf_k + 1))

        # Run local BA
        _, opt_vals, _, _, factor_metadata, _ = run_bundle_adjustment(
            db, bundle_frames, poses_4x4, gtsam_calib, pixel_sigma=1.0, make_relative=True
        )

        for meta in factor_metadata:
            meas = meta['meas']
            fId = meta['fId']
            dist_from_ref = fId - kf_0  # Distance from the 1st frame of this bundle

            if dist_from_ref >= 20: continue

            cam_opt = gtsam.StereoCamera(opt_vals.atPose3(meta['c_key']), gtsam_calib)
            proj_opt = cam_opt.project(opt_vals.atPoint3(meta['q_key']))

            err_l = np.sqrt((proj_opt.uL() - meas.uL()) ** 2 + (proj_opt.v() - meas.v()) ** 2)
            err_r = np.sqrt((proj_opt.uR() - meas.uR()) ** 2 + (proj_opt.v() - meas.v()) ** 2)

            left_errors_by_dist[dist_from_ref].append(err_l)
            right_errors_by_dist[dist_from_ref].append(err_r)

    # Calculate Medians
    distances = []
    med_left, med_right = [], []
    for d in range(20):
        if len(left_errors_by_dist[d]) > 0:
            distances.append(d)
            med_left.append(np.median(left_errors_by_dist[d]))
            med_right.append(np.median(right_errors_by_dist[d]))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(distances, med_left, label='Left Camera', color='darkcyan', marker='o', markersize=4)
    ax.plot(distances, med_right, label='Right Camera', color='darkred', marker='x', markersize=4)
    format_plot(ax, 'Bundle Adjustment - Projection Error vs Distance from Anchor', 'Distance from Reference Frame',
                'Median Projection Error (Pixels)')
    ax.legend()
    plt.show()


def analyze_mahalanobis_distance_spectrum(target_kf, optimized_values, adj_list, threshold=16.81):
    """
    Addresses: Visualizing the probabilistic candidate search space.
    Plots the Mahalanobis D^2 for all historical frames relative to a target loop frame.
    """
    print(f"\n--- Plotting Mahalanobis Distance Spectrum for Frame {target_kf} ---")

    target_key = gtsam.symbol('c', target_kf)
    accumulated_covs = get_shortest_paths_covariances(adj_list, target_key)

    frame_ids = []
    distances = []
    pose_n = optimized_values.atPose3(target_key)

    for candidate_key, cov_matrix in accumulated_covs.items():
        if not optimized_values.exists(candidate_key):
            continue

        pose_i = optimized_values.atPose3(candidate_key)
        candidate_idx = gtsam.symbolIndex(candidate_key)

        # Enforce a small temporal gap so we don't just match frame t to frame t-1
        if (target_kf - candidate_idx) < 100:
            continue

        # Mahalanobis Math
        relative_pose = pose_i.between(pose_n)
        error_vector = gtsam.Pose3.Logmap(relative_pose)
        info_matrix = np.linalg.pinv(cov_matrix)
        md_squared = error_vector.T @ info_matrix @ error_vector

        frame_ids.append(candidate_idx)
        distances.append(md_squared)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame_ids, distances, 'b.', alpha=0.6, label="Mahalanobis $D^2$")
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f"$\chi^2$ Threshold ({threshold})")

    ax.set_yscale('log')
    format_plot(ax, f"Mahalanobis Distance Spectrum from Loop Frame (Keyframe {target_kf})",
                "Candidate Historical Frame ID", "Squared Mahalanobis Distance ($D^2$)")
    ax.legend(loc='upper right')
    ax.grid(True, which="both", ls=":", alpha=0.6)
    plt.show()


def main():
    print("==========================================================")
    print("   INITIALIZING SLAM PERFORMANCE ANALYSIS SUITE (KITTI 00)")
    print("==========================================================\n")

    # -------------------------------------------------------------------------
    # STAGE 1: INFRASTRUCTURE & DATA LOADING
    # -------------------------------------------------------------------------
    print("--- Stage 1: Loading Dataset Geometry & Feature Database ---")
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)
    K, M1, M2 = read_cameras(DATA_PATH)
    P1 = K @ M1
    P2 = K @ M2
    gtsam_calib = create_gtsam_calibration(K, M2)

    db = TrackingDB()
    if os.path.exists(DB_NAME + ".pkl"):
        db.load(DB_NAME)
    else:
        print(f"[ERROR] Database file {DB_NAME}.pkl not found. Please run ex4.py first.")
        return

    # -------------------------------------------------------------------------
    # STAGE 2: FRONT-END TRACKING & CONNECTIVITY ANALYTICS
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Front-End Tracking & Connectivity Statistics ---")
    analyze_tracking_statistics(db)
    analyze_reprojection_vs_distance(db, poses_4x4, P1, P2)

    # -------------------------------------------------------------------------
    # STAGE 3: PNP VISUAL ODOMETRY & LOCAL OPTIMIZATION CONVERGENCE
    # -------------------------------------------------------------------------
    print("\n--- Stage 3: PnP Trajectory & Local BA Convergence ---")
    # Generate pure PnP trajectory directly from DB matches
    pnp_poses_dict = generate_pnp_trajectory_from_db(db, K, P1, P2)

    # Analyze optimization convergence starting from noisy PnP initial guess
    analyze_optimization_errors(db, pnp_poses_dict, gtsam_calib)

    # -------------------------------------------------------------------------
    # STAGE 4: SEQUENTIAL POSE GRAPH (DEAD-RECKONING BACK-END)
    # -------------------------------------------------------------------------
    print("\n--- Stage 4: Sequential Pose Graph Construction ---")
    # Extract relative constraints via Information Matrix Conditioning
    rel_measurements, keyframes = generate_sequential_measurements(
        db, pnp_poses_dict, gtsam_calib, dist_threshold=5.0, cov_inflation=COV_SCALE
    )

    # Build and solve initial sequential pose graph
    global_graph, initial_guess = build_pose_graph(rel_measurements)
    optimizer = gtsam.LevenbergMarquardtOptimizer(global_graph, initial_guess)
    pg_base_opt_vals = optimizer.optimize()

    # Preserve base graph copy and compute marginal covariances BEFORE loop closures
    base_graph = gtsam.NonlinearFactorGraph(global_graph)
    print("Extracting Marginal Covariances for Base Pose Graph (Without Loops)...")
    base_marginals = gtsam.Marginals(base_graph, pg_base_opt_vals)

    # -------------------------------------------------------------------------
    # STAGE 5: ONLINE LOOP CLOSURE DETECTION & GLOBAL RE-OPTIMIZATION
    # -------------------------------------------------------------------------
    print("\n--- Stage 5: Online Loop Closure Detection & Factor Injection ---")
    dynamic_adj_list = build_initial_adjacency_list(global_graph)
    current_optimized_values = pg_base_opt_vals

    target_kf = min([kf for kf in keyframes if kf >= 1470])
    analyze_mahalanobis_distance_spectrum(target_kf, pg_base_opt_vals, dynamic_adj_list)

    successful_loops = []
    loop_cooldown = 0

    for n_idx, current_kf in enumerate(keyframes):
        if n_idx < 100:
            continue
        if loop_cooldown > 0:
            loop_cooldown -= 1
            continue

        current_key = gtsam.symbol('c', current_kf)
        accumulated_covs = get_shortest_paths_covariances(dynamic_adj_list, current_key)

        # Mahalanobis statistical distance check (Chi-square 99% confidence threshold = 16.81)
        candidates = filter_candidates_by_mahalanobis(
            current_optimized_values, current_key, accumulated_covs,
            mahalanobis_thresh=16.81, min_frame_gap=150
        )
        if not candidates:
            continue

        # 4-way RANSAC consensus validation
        valid_loop, candidate_kf, loop_data = q7_2(candidates, current_kf, K, P1, P2, M2)
        if not valid_loop:
            continue

        candidate_key = gtsam.symbol('c', candidate_kf)
        # Solve mini-bundle and extract conditional covariance via Information Matrix Conditioning
        rel_pose, rel_cov = q7_3(gtsam_calib, loop_data, candidate_kf, current_kf)

        # Inject BetweenFactorPose3 wrapped in Cauchy Robust Kernel into global graph
        global_graph, current_optimized_values = q7_4(
            global_graph, current_optimized_values, current_key, candidate_key, rel_pose, rel_cov
        )

        # Update dynamic graph adjacency list for Dijkstra covariance accumulation
        dynamic_adj_list[current_key].append((candidate_key, rel_cov))
        dynamic_adj_list[candidate_key].append((current_key, rel_cov))

        successful_loops.append({
            'current_kf': current_kf,
            'candidate_kf': candidate_kf,
            'match_data': loop_data
        })
        loop_cooldown = 5

    print(f"Loop Closure Pipeline Complete: {len(successful_loops)} valid loops detected and integrated.")

    # Extract final marginal covariances AFTER global loop closure optimization
    print("Extracting Marginal Covariances for Final Pose Graph (With Loops)...")
    final_marginals = gtsam.Marginals(global_graph, current_optimized_values)

    # -------------------------------------------------------------------------
    # STAGE 6: TRAJECTORY DICTIONARY CONVERSION & BENCHMARKING
    # -------------------------------------------------------------------------
    print("\n--- Stage 6: Extracting Global Trajectories & Benchmarking ---")
    ba_poses_dict = {}
    pg_base_dict = {}
    pg_final_poses_dict = {}

    # Invert GTSAM camera-to-world (T_c_w) poses into KITTI world-to-camera (T_w_c) format
    for kf in keyframes:
        key = gtsam.symbol('c', kf)
        if initial_guess.exists(key):
            ba_poses_dict[kf] = np.linalg.inv(initial_guess.atPose3(key).matrix())
        if pg_base_opt_vals.exists(key):
            pg_base_dict[kf] = np.linalg.inv(pg_base_opt_vals.atPose3(key).matrix())
        if current_optimized_values.exists(key):
            pg_final_poses_dict[kf] = np.linalg.inv(current_optimized_values.atPose3(key).matrix())

    # Master bird's-eye view trajectory plot
    plot_master_trajectory(poses_4x4, pnp_poses_dict, ba_poses_dict, pg_final_poses_dict, keyframes)

    # Absolute Error Graphs (X, Y, Z, Norm, Angle)
    analyze_absolute_errors(poses_4x4, pnp_poses_dict, "PnP Dead Reckoning")
    analyze_absolute_errors(poses_4x4, ba_poses_dict, "Local Bundle Adjustment")
    analyze_absolute_errors(poses_4x4, pg_base_dict, "Absolute Pose Graph (No Loop Closures)")
    analyze_absolute_errors(poses_4x4, pg_final_poses_dict, "Global Pose Graph (With Loop Closures)")

    # Consecutive Relative Keyframe Errors
    analyze_consecutive_relative_errors(poses_4x4, pnp_poses_dict, "PnP", keyframes)
    analyze_consecutive_relative_errors(poses_4x4, ba_poses_dict, "Bundle Adjustment", keyframes)
    analyze_bundle_reprojection_vs_distance(db, poses_4x4, gtsam_calib)

    # KITTI Sub-Sequence Relative Errors (100, 400, 800 frames)
    analyze_kitti_relative_errors(poses_4x4, pnp_poses_dict, "PnP Dead Reckoning")
    analyze_kitti_relative_errors(poses_4x4, ba_poses_dict, "Local Bundle Adjustment")

    # -------------------------------------------------------------------------
    # STAGE 7: UNCERTAINTY ISOLATION & LOOP CLOSURE METRICS
    # -------------------------------------------------------------------------
    print("\n--- Stage 7: Uncertainty Trace Isolation & Loop Match Analytics ---")
    plot_full_uncertainty_analysis(base_marginals, final_marginals, keyframes)
    analyze_loop_closure_stats(successful_loops)

    print("\n==========================================================")
    print("   PERFORMANCE ANALYSIS SUITE EXECUTED SUCCESSFULLY")
    print("==========================================================")


if __name__ == "__main__":
    main()
