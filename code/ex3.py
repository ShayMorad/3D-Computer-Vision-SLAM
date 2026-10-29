"""
ex3.py
Main execution file for VAN Exercise 3.
"""
import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from dataset_utils import read_ground_truth, read_stereo_pair, read_cameras
from geometry import compute_reprojection_error, run_custom_ransac
from feature_tracking import match_features, process_stereo_frame, prepare_4way_data
from visualization import plot_cameras_birds_eye, plot_supporters_on_images, format_plot

DATA_PATH = '../dataset/sequences/00/'
GT_PATH = '../dataset/poses/00.txt'
CURRENT_ALGO = 'AKAZE'


def q3_1(P1, P2):
    """Answers Q 3.1: Create point clouds for Pair 0 and Pair 1."""
    print("STARTING: --- Q 3.1 ---")

    # Process Frame 0
    (kp0_l, des0_l, kp0_r, des0_r, matches0, pts3D_0) = process_stereo_frame(DATA_PATH, 0, CURRENT_ALGO, P1, P2)
    print(f"Frame 0 -> Valid 3D Points: {len(pts3D_0)}")

    # Process Frame 1
    (kp1_l, des1_l, kp1_r, des1_r, matches1, pts3D_1) = process_stereo_frame(DATA_PATH, 1, CURRENT_ALGO, P1, P2)
    print(f"Frame 1 -> Valid 3D Points: {len(pts3D_1)}")

    print("FINISHED: --- Q 3.1 ---\n")

    return kp0_l, des0_l, kp0_r, kp1_l, des1_l, kp1_r, matches0, matches1, pts3D_0, pts3D_1


def q3_2(des0_l, des1_l):
    """Answers Q 3.2: Match features between the two left images."""
    print("STARTING: --- Q 3.2 ---")

    # We trust the descriptor match here without the vertical geometric filter
    matches_t = match_features(des0_l, des1_l, CURRENT_ALGO)
    print(f"Temporal Matches (Left 0 -> Left 1): {len(matches_t)}")

    print("FINISHED: --- Q 3.2 ---\n")

    return matches_t


def q3_3(matches0, matches1, matches_t, kp0_l, kp1_l, pts3D_0, K, M2):
    """Answers Q 3.3: Calculate [R|t] using 4 tracked points via PnP."""
    print("STARTING: --- Q 3.3 ---")

    # 1. Fetch the synced 4-way data
    obj_pts_3D, img_pts_2D, track_indices = prepare_4way_data(matches0, matches1, matches_t, kp0_l, kp1_l, pts3D_0)

    print(f"Total usable 4-way tracking points: {len(obj_pts_3D)}")

    # 2. Take exactly 4 points for the initial PnP estimation
    obj_points = obj_pts_3D[:4]
    img_points = img_pts_2D[:4]

    # 3. Solve Perspective-n-Point
    # We pass the 3D points, the 2D image points on left_1, and the Intrinsic Matrix K
    success, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, None, flags=cv2.SOLVEPNP_P3P)

    if success:
        # Convert the rotation vector into a 3x3 rotation matrix R
        R, _ = cv2.Rodrigues(rvec)
        print("\nExtrinsic Matrix of left_1 (relative to left_0):")
        print("Rotation Matrix (R):\n", R)
        print("Translation Vector (t):\n", tvec)

        # Calculate camera locations
        C_l0 = np.array([[0.0], [0.0], [0.0]])

        t_stereo = M2[:, 3].reshape(3, 1)
        C_r0 = -t_stereo

        C_l1 = -R.T @ tvec
        C_r1 = -R.T @ (tvec + t_stereo)

        # Build the lists for our generic plotter
        cameras = [C_l0, C_r0, C_l1, C_r1]
        labels = ['Left 0', 'Right 0', 'Left 1 (PnP est)', 'Right 1']
        colors = ['blue', 'cyan', 'red', 'orange']

        # Connect Left0 to Right0 (index 0 and 1), and Left1 to Right1 (index 2 and 3)
        connections = [(0, 1), (2, 3)]

        plot_cameras_birds_eye(cameras, labels, colors, "Part 3.3: Relative Position of Cameras", connections)
    else:
        print("WARNING: solvePnP failed.")
        exit()
    print("FINISHED: --- Q 3.3 ---\n")

    return R, tvec, obj_pts_3D, img_pts_2D, track_indices


def q3_4(R, tvec, obj_pts_3D, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1, P2, M2):
    """Answers Q 3.4: Find Supporters using Reprojection Error across all 4 cameras."""
    print("STARTING: --- Q 3.4 ---")

    # 1. Define the 4 Projection Matrices
    P_l0 = P1  # Frame 0 Left
    P_r0 = P2  # Frame 0 Right

    # Frame 1 Left: P_l1 = K * [R | t]
    P_l1 = K @ np.hstack((R, tvec))

    # Frame 1 Right: P_r1 = K * [R | t + t_stereo]
    t_stereo = M2[:, 3].reshape(3, 1)
    P_r1 = K @ np.hstack((R, tvec + t_stereo))

    supporters_mask = []

    # 2. Test every single 3D point
    for i, idx_tuple in enumerate(track_indices):
        idx_l0, idx_r0, idx_l1, idx_r1 = idx_tuple
        pt3D = obj_pts_3D[i]

        # Get the actual matched 2D pixels for this feature
        pt_l0 = kp0_l[idx_l0].pt
        pt_r0 = kp0_r[idx_r0].pt
        pt_l1 = kp1_l[idx_l1].pt
        pt_r1 = kp1_r[idx_r1].pt

        # Calculate the error for all 4 cameras
        err_l0 = compute_reprojection_error(pt3D, P_l0, pt_l0)
        err_r0 = compute_reprojection_error(pt3D, P_r0, pt_r0)
        err_l1 = compute_reprojection_error(pt3D, P_l1, pt_l1)
        err_r1 = compute_reprojection_error(pt3D, P_r1, pt_r1)

        # CONSENSUS: It is ONLY a supporter if ALL FOUR errors are <= 2.0 pixels
        if err_l0 <= 2.0 and err_r0 <= 2.0 and err_l1 <= 2.0 and err_r1 <= 2.0:
            supporters_mask.append(True)
        else:
            supporters_mask.append(False)

    num_supporters = sum(supporters_mask)
    print(f"Total points tested: {len(track_indices)}")
    print(f"Number of Supporters (Inliers) for this specific [R|t]: {num_supporters}")

    # 3. Plot the results
    img0_l, _ = read_stereo_pair(DATA_PATH, 0)
    img1_l, _ = read_stereo_pair(DATA_PATH, 1)

    # Extract just the left_0 and left_1 indices for plotting
    indices_l0 = [t[0] for t in track_indices]
    indices_l1 = [t[2] for t in track_indices]

    plot_supporters_on_images(img0_l, img1_l, kp0_l, kp1_l, indices_l0, indices_l1, supporters_mask)

    print("FINISHED: --- Q 3.4 ---\n")

    return supporters_mask


def q3_5(obj_pts_3D, pts3D_1, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1, P2, M2):
    """Answers Q 3.5: Execute Custom RANSAC and refine the transformation."""
    print("STARTING: --- Q 3.5: Custom RANSAC ---")

    # 1. Run our Custom RANSAC Engine
    refined_R, refined_t, best_mask = run_custom_ransac(obj_pts_3D, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1,
                                                        P2, M2, num_iterations=200)

    print("\nRefined Extrinsic Matrix [R|t]:")
    print("R:\n", refined_R)
    print("t:\n", refined_t)

    # 2. Transform Point Cloud 0 using T
    pts3D_0_hom = np.hstack((obj_pts_3D, np.ones((len(obj_pts_3D), 1))))

    T = np.vstack((np.hstack((refined_R, refined_t)), [0, 0, 0, 1]))
    print("T:\n", T)

    # Transform: X_new = T * X_old
    transformed_pts0 = (T @ pts3D_0_hom.T).T
    # Convert back to Cartesian by dropping the 4th coordinate
    transformed_pts0 = transformed_pts0[:, :3]

    # 3. Plot the Point Clouds (From Above: X and Z axes)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot transformed Cloud 0 (e.g., Blue)
    ax.scatter(transformed_pts0[:, 0], transformed_pts0[:, 2], s=2, c='blue', label='Transformed Cloud 0', alpha=0.5)

    # Plot Cloud 1 (e.g., Red)
    ax.scatter(pts3D_1[:, 0], pts3D_1[:, 2], s=2, c='red', label='Cloud 1', alpha=0.5)

    # Format the plot to crop "points at infinity" as requested by the assignment
    ax.set_xlim([-20, 20])
    ax.set_ylim([0, 60])  # Look up to 60 meters forward
    format_plot(ax, "Part 3.5: Superimposed 3D Point Clouds (Bird's Eye)", "X (Meters)", "Z (Meters)")
    ax.legend()
    plt.show()

    # 4. Plot Inliers and Outliers on the images
    img0_l, _ = read_stereo_pair(DATA_PATH, 0)
    img1_l, _ = read_stereo_pair(DATA_PATH, 1)

    indices_l0 = [t[0] for t in track_indices]
    indices_l1 = [t[2] for t in track_indices]

    plot_supporters_on_images(img0_l, img1_l, kp0_l, kp1_l, indices_l0, indices_l1, best_mask)
    print("FINISHED: --- Q 3.5: Custom RANSAC ---\n")

    return refined_R, refined_t


def q3_6(K, P1, P2, M2, num_frames=100):
    """Answers Q 3.6: Tracks the full sequence and plots against Ground Truth."""
    print(f"STARTING: --- Q 3.6: Full Sequence Trajectory Tracking ({num_frames} frames) ---")

    start_time = time.time()

    # 1. Initialize Global Pose (Start at Identity)
    current_pose = np.eye(4)

    # 2. Arrays to store trajectory
    traj_x = [0.0]
    traj_z = [0.0]

    # 3. Prime the loop by processing Frame 0
    (kp_prev_l, des_prev_l, kp_prev_r, des_prev_r, matches_prev, pts3D_prev) = \
        (process_stereo_frame(DATA_PATH, 0, CURRENT_ALGO, P1, P2))

    for i in range(1, num_frames):

        # A. Process new frame
        (kp_curr_l, des_curr_l, kp_curr_r, des_curr_r, matches_curr, pts3D_curr) = \
            (process_stereo_frame(DATA_PATH, i, CURRENT_ALGO, P1, P2))

        # B. Match Temporal
        matches_t = match_features(des_prev_l, des_curr_l, CURRENT_ALGO)

        # C. Prepare 4-way tracking
        obj_pts_3D, img_pts_2D, track_indices = \
            (prepare_4way_data(matches_prev, matches_curr, matches_t, kp_prev_l, kp_curr_l, pts3D_prev))

        # Failsafe for lost tracking
        if len(obj_pts_3D) < 10:
            print(f"WARNING: Tracking lost at frame {i}! Insufficient points.")
            break

        # D. Run Custom RANSAC
        R_rel, t_rel, best_mask = run_custom_ransac(obj_pts_3D, track_indices, kp_prev_l, kp_prev_r, kp_curr_l,
                                                    kp_curr_r,
                                                    K, P1, P2, M2, num_iterations=200, should_print=False)

        # E. Build 4x4 T matrix
        T_rel = np.eye(4)
        T_rel[:3, :3] = R_rel
        T_rel[:3, 3] = t_rel.flatten()

        # F. Update Global Pose: P_new = P_old * inv(T_rel)
        current_pose = current_pose @ np.linalg.inv(T_rel)

        # G. Save coordinates for plotting
        traj_x.append(current_pose[0, 3])
        traj_z.append(current_pose[2, 3])

        # H. --- LOOP OPTIMIZATION ---
        # Shift the current frame variables to become the prev variables
        kp_prev_l = kp_curr_l
        des_prev_l = des_curr_l
        kp_prev_r = kp_curr_r
        matches_prev = matches_curr
        pts3D_prev = pts3D_curr

    print("\nTracking Complete!")
    end_time = time.time()
    total_seconds = (end_time - start_time)
    print(f"Total Execution Time: {total_seconds:.2f} seconds")

    # 4. Load Ground Truth and Plot
    # Adjust this path if your poses directory is located elsewhere

    if os.path.exists(GT_PATH):
        gt_x, gt_z = read_ground_truth(GT_PATH, len(traj_x))

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(traj_x, traj_z, label=f'Calculated Trajectory ({CURRENT_ALGO})', color='blue', linewidth=2)
        ax.plot(gt_x, gt_z, label='Ground Truth', color='red', linestyle='--', linewidth=2)

        format_plot(ax, f"Part 3.6: Visual Odometry Trajectory vs Ground Truth", "X (Meters)", "Z (Meters)",
                    equal_axes=True)
        ax.legend()
        plt.show()
    else:
        print(f"Ground truth file not found at {GT_PATH}. Skipping comparison plot.")

    print(f"FINISHED: --- Q 3.6: Full Sequence Trajectory Tracking ({num_frames} frames) ---\n")

    return traj_x, traj_z


def main():
    print(f"--- Starting Exercise 3 using {CURRENT_ALGO} ---\n")

    K, M1, M2 = read_cameras(DATA_PATH)
    P1 = K @ M1
    P2 = K @ M2

    # Q 3.1
    (kp0_l, des0_l, kp0_r, kp1_l, des1_l, kp1_r, matches0, matches1, pts3D_0, pts3D_1) = q3_1(P1, P2)

    # Q 3.2
    matches_t = q3_2(des0_l, des1_l)

    # Q 3.3
    R, tvec, obj_pts_3D, img_pts_2D, track_indices = q3_3(matches0, matches1, matches_t, kp0_l, kp1_l, pts3D_0, K, M2)

    # Q 3.4
    supporters_mask = q3_4(R, tvec, obj_pts_3D, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1, P2, M2)

    # Q 3.5
    refined_R, refind_t = q3_5(obj_pts_3D, pts3D_1, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1, P2, M2)

    # Q 3.6
    traj_x, traj_z = q3_6(K, P1, P2, M2, num_frames=3300)


if __name__ == "__main__":
    main()
