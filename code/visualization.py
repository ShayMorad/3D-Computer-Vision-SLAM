import cv2
import matplotlib.pyplot as plt
import numpy as np
import gtsam
import gtsam.utils.plot as gtsam_plot
import matplotlib.lines as mlines
from matplotlib.patches import Ellipse


def format_plot(ax, title, xlabel, ylabel, equal_axes=False):
    """
    Standardizes plot formatting according to course guidelines.
    Ensures clear titles, labeled axes with units, and reduced whitespace.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if equal_axes:
        ax.axis('equal')
    plt.tight_layout()


def draw_inliers_outliers(img1, img2, kp1, kp2, inliers, outliers):
    """
    Creates a side-by-side visualization of inliers and outliers.
    Draws lines connecting the matched features across the two images.
    Outliers are drawn first in Cyan, Inliers second in Orange.
    """
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    canvas = np.zeros((max(h1, h2), w1 + w2), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # 1. Draw OUTLIERS (Cyan)
    # Drawn first so they fall to the background behind the valid inliers
    for m in outliers:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))

        # Draw the connecting line
        cv2.line(canvas, pt1, pt2, (255, 255, 0), 1)
        # Draw the keypoint circles
        cv2.circle(canvas, pt1, 3, (255, 255, 0), -1)
        cv2.circle(canvas, pt2, 3, (255, 255, 0), -1)

    # 2. Draw INLIERS (Orange)
    for m in inliers:
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
        pt2 = (int(kp2[m.trainIdx].pt[0]) + w1, int(kp2[m.trainIdx].pt[1]))

        # Draw the connecting line
        cv2.line(canvas, pt1, pt2, (0, 165, 255), 1)
        # Draw the keypoint circles
        cv2.circle(canvas, pt1, 3, (0, 165, 255), -1)
        cv2.circle(canvas, pt2, 3, (0, 165, 255), -1)

    return canvas


def plot_cameras_birds_eye(cameras, labels, colors, title, connections=None):
    """
    Plots a bird's-eye view (X-Z plane) of 3D camera positions.

    :param cameras: List of 3x1 numpy arrays representing [X, Y, Z]^T coordinates.
    :param labels: List of strings for the legend/text.
    :param colors: List of colors for each camera.
    :param connections: List of tuples (index_A, index_B) to draw dashed lines between cameras.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot each camera
    for i, cam in enumerate(cameras):
        x, z = cam[0, 0], cam[2, 0]
        ax.scatter(x, z, color=colors[i], s=100, label=labels[i])
        ax.text(x + 0.1, z + 0.1, labels[i], fontsize=10)

    # Draw optional connecting lines (like stereo baselines)
    if connections:
        for idx1, idx2 in connections:
            x_vals = [cameras[idx1][0, 0], cameras[idx2][0, 0]]
            z_vals = [cameras[idx1][2, 0], cameras[idx2][2, 0]]
            ax.plot(x_vals, z_vals, 'k--', alpha=0.5)

    format_plot(ax, title, "X (Meters) [Right/Left]", "Z (Meters) [Forward/Backward]", equal_axes=True)

    ax.legend()
    ax.grid(True)
    plt.show()


def plot_supporters_on_images(img0, img1, kp0, kp1, indices_l0, indices_l1, is_supporter_mask):
    """
    Plots the tracked features on Left 0 and Left 1 side-by-side.
    Supporters are colored green, Outliers are colored red.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Convert images to RGB for Matplotlib
    img0_rgb = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    axes[0].imshow(img0_rgb)
    axes[1].imshow(img1_rgb)

    # Plot the points
    for i in range(len(indices_l0)):
        pt0 = kp0[indices_l0[i]].pt
        pt1 = kp1[indices_l1[i]].pt

        # Green for supporters, Red for outliers
        color = 'lime' if is_supporter_mask[i] else 'red'
        marker_size = 12 if is_supporter_mask[i] else 4
        alpha = 1.0 if is_supporter_mask[i] else 0.3

        axes[0].scatter(pt0[0], pt0[1], c=color, s=marker_size, alpha=alpha)
        axes[1].scatter(pt1[0], pt1[1], c=color, s=marker_size, alpha=alpha)

    format_plot(axes[0], "Left 0: Supporters (Green) vs Outliers (Red)", "X [pixels]", "Y [pixels]")
    format_plot(axes[1], "Left 1: Supporters (Green) vs Outliers (Red)", "X [pixels]", "Y [pixels]")

    plt.tight_layout()
    plt.show()


def plot_track_feature_crops(frames, track_id, db, data_path, read_image_func, crop_size=20):
    """
    Plots the full image and a zoomed-in crop around a specific tracked feature across multiple frames.
    """
    num_frames = len(frames)
    fig, axes = plt.subplots(num_frames, 2, figsize=(10, 2.5 * num_frames))
    fig.suptitle(f'Track #{track_id} across {num_frames} frames', fontsize=14)

    for i, fId in enumerate(frames):
        # Read the left image using the injected reading function
        img_l, _ = read_image_func(data_path, fId)
        link = db.link(fId, track_id)
        cx, cy = int(link.x_left), int(link.y)

        # Define bounding box with image boundary protection
        h, w = img_l.shape
        half_crop = crop_size // 2
        x1, x2 = max(0, cx - half_crop), min(w, cx + half_crop)
        y1, y2 = max(0, cy - half_crop), min(h, cy + half_crop)
        crop = img_l[y1:y2, x1:x2]

        # Left Column: Full image with marker
        img_rgb = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB)
        cv2.circle(img_rgb, (cx, cy), 5, (255, 0, 0), -1)
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f'Frame {fId}')
        axes[i, 0].axis('off')

        # Right Column: Crop with center cross
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop_cx, crop_cy = cx - x1, cy - y1
        cv2.drawMarker(crop_rgb, (crop_cx, crop_cy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=5,
                       thickness=1)
        axes[i, 1].imshow(crop_rgb)
        axes[i, 1].set_title(f'{crop_size}x{crop_size} Crop')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def plot_sequence_stat(x_data, y_data, title, xlabel, ylabel, mean_line=False):
    """Generic function to plot sequential statistics over frames."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_data, y_data)

    if mean_line:
        ax.axhline(y=np.mean(y_data), color='g', linestyle='-', label="Mean")
        ax.legend()

    ax.set_xlim(0, len(x_data))
    if max(y_data) <= 100 and "%" in ylabel:
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, max(y_data) * 1.1)

    format_plot(ax, title, xlabel, ylabel)
    plt.show()


def plot_histogram(data, bins, title, xlabel, ylabel, log_scale=False):
    """Generic histogram plotter utilizing format_plot."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(data, bins=bins, log=log_scale)
    format_plot(ax, title, xlabel, ylabel)
    plt.show()


def plot_reprojection_errors(distances, errors_left, errors_right):
    """Plots the L2 reprojection error of a triangulated point across frames."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances, errors_left, label='Left', color='slategray')
    ax.plot(distances, errors_right, label='Right', color='goldenrod', alpha=0.8)

    format_plot(ax, 'PnP - projection error vs track length', 'distance from reference', 'projection error (pixels)')
    ax.legend()
    plt.show()


def plot_gtsam_factor_vs_reprojection(distances, l2_errors, factor_errors, track_id):
    """
    Plots the standard L2 reprojection error alongside the GTSAM factor error
    for a single tracked 3D point over time. Answers Q 5.1 and 5.2.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(distances, factor_errors, label="GTSAM Factor Error (Mahalanobis)", marker='o', color='darkred')
    ax.plot(distances, l2_errors, label="L2 Reprojection Error (Pixels)", marker='x', linestyle='--', color='darkcyan')

    format_plot(ax, f"Track #{track_id} Backwards Reprojection Verification",
                "Frames Backward from Reference", "Error Magnitude")

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_stereo_projection(img_l, img_r, meas_uL, meas_uR, meas_v, proj_uL, proj_uR, proj_v, title):
    """Plots the actual measurement vs the GTSAM projection on the stereo images."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(title, fontsize=14)

    # Convert grayscale to RGB for colored markers
    img_l_rgb = cv2.cvtColor(img_l, cv2.COLOR_GRAY2RGB)
    img_r_rgb = cv2.cvtColor(img_r, cv2.COLOR_GRAY2RGB)

    axes[0].imshow(img_l_rgb)
    axes[0].scatter(meas_uL, meas_v, c='lime', marker='+', s=100, label='Measurement (DB)')
    axes[0].scatter(proj_uL, proj_v, c='red', marker='x', s=100, label='Projection (GTSAM)')
    format_plot(axes[0], "Left Camera", "u (pixels)", "v (pixels)")
    axes[0].legend()

    axes[1].imshow(img_r_rgb)
    axes[1].scatter(meas_uR, meas_v, c='lime', marker='+', s=100, label='Measurement (DB)')
    axes[1].scatter(proj_uR, proj_v, c='red', marker='x', s=100, label='Projection (GTSAM)')
    format_plot(axes[1], "Right Camera", "u (pixels)", "v (pixels)")
    axes[1].legend()

    # Zoom in around the points for clarity
    margin = 50
    for ax in axes:
        ax.set_xlim(min(meas_uL, proj_uL, meas_uR, proj_uR) - margin,
                    max(meas_uL, proj_uL, meas_uR, proj_uR) + margin)
        ax.set_ylim(max(meas_v, proj_v) + margin, min(meas_v, proj_v) - margin)  # Y is inverted in images

    plt.show()


def custom_3d_trajectory_plot(ax, values, frame_keys, scale=0.5):
    """Plots a tight 3D camera trajectory mapped to a ground-plane perspective."""
    xs_plt, ys_plt, zs_plt = [], [], []

    for key in frame_keys:
        pose = values.atPose3(key)
        t = pose.translation()
        R = pose.rotation().matrix()

        # Map KITTI coordinates to standard plotting heights
        x_plt, y_plt, z_plt = t[0], t[2], -t[1]
        xs_plt.append(x_plt)
        ys_plt.append(y_plt)
        zs_plt.append(z_plt)

        # Draw Camera Axes
        ax.plot([x_plt, x_plt + R[0, 0] * scale], [y_plt, y_plt + R[2, 0] * scale], [z_plt, z_plt - R[1, 0] * scale],
                color='red', linewidth=1.2)
        ax.plot([x_plt, x_plt + R[0, 1] * scale], [y_plt, y_plt + R[2, 1] * scale], [z_plt, z_plt - R[1, 1] * scale],
                color='green', linewidth=1.2)
        ax.plot([x_plt, x_plt + R[0, 2] * scale], [y_plt, y_plt + R[2, 2] * scale], [z_plt, z_plt - R[1, 2] * scale],
                color='dodgerblue', linewidth=1.2)
        ax.scatter(x_plt, y_plt, z_plt, color='black', s=15, depthshade=True)

    ax.plot(xs_plt, ys_plt, zs_plt, color='black', linestyle='--', linewidth=1.5)

    # Force a tight, equal bounding box around the data points
    max_range = np.array([max(xs_plt) - min(xs_plt), max(ys_plt) - min(ys_plt), max(zs_plt) - min(zs_plt)]).max() / 2.0
    mid_x, mid_y, mid_z = (max(xs_plt) + min(xs_plt)) * 0.5, (max(ys_plt) + min(ys_plt)) * 0.5, (
            max(zs_plt) + min(zs_plt)) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def plot_bundle_results(initial_values, optimized_values, frame_keys, marginals=None):
    """Plots tight, professionally titled bundle results."""

    # Figure 1: GTSAM Native Plot
    fig1 = plt.figure(1, figsize=(7, 5))
    ax1 = fig1.add_subplot(111, projection='3d')
    gtsam_plot.plot_trajectory(1, optimized_values, scale=0.3, marginals=marginals)
    ax1.set_title("GTSAM Factor Graph State with Marginal Covariances", fontsize=11, fontweight='semibold')
    ax1.view_init(elev=15, azim=-60)
    fig1.tight_layout()

    # Figure 2: Custom Ground Plane 3D Plot
    fig2 = plt.figure(2, figsize=(7, 5))
    ax2 = fig2.add_subplot(111, projection='3d')
    custom_3d_trajectory_plot(ax2, optimized_values, frame_keys, scale=0.4)
    ax2.set_title("Local Window Bundle Adjustment: 3D Optimized Trajectory", fontsize=11, fontweight='semibold')
    ax2.set_xlabel("X (Right) [m]")
    ax2.set_ylabel("Z (Forward) [m]")
    ax2.set_zlabel("-Y (Up) [m]")
    ax2.view_init(elev=8, azim=12)
    fig2.tight_layout()

    # Figure 3: 2D Bird's-Eye View (FIXED: Un-cropped)
    fig3 = plt.figure(3, figsize=(8, 8))
    ax3 = fig3.add_subplot(111)

    x_init, z_init = [], []
    x_opt, z_opt = [], []

    for key in frame_keys:
        p_init = initial_values.atPose3(key)
        p_opt = optimized_values.atPose3(key)
        x_init.append(p_init.translation()[0])
        z_init.append(p_init.translation()[2])
        x_opt.append(p_opt.translation()[0])
        z_opt.append(p_opt.translation()[2])

    ax3.plot(x_init, z_init, 'bo-', label='Initial (PnP)', alpha=0.4, linewidth=2, markersize=4)
    ax3.plot(x_opt, z_opt, 'ro-', label='Optimized (BA)', linewidth=1.5, markersize=4)

    # Plot Landmarks
    lx, lz = [], []
    for key in optimized_values.keys():
        if gtsam.symbolChr(key) == ord('q'):
            pt = optimized_values.atPoint3(key)
            lx.append(pt[0])
            lz.append(pt[2])
    ax3.scatter(lx, lz, c='gray', s=1.5, alpha=0.3, label='Landmarks')

    # Apply formatting and let Matplotlib naturally bounds the landmarks
    ax3.set_title("Local Window Bundle Adjustment: 2D Bird's-Eye View Trajectory", fontsize=11, fontweight='semibold')
    ax3.set_xlabel("X Coordinate (East) [m]")
    ax3.set_ylabel("Z Coordinate (North) [m]")
    ax3.grid(True, linestyle=':', alpha=0.6)

    # This ensures 1m X = 1m Z without aggressively cropping the data
    ax3.axis('equal')

    ax3.legend(loc='best', frameon=True, facecolor='white', framealpha=0.9)
    fig3.tight_layout()
    plt.show()


def plot_global_trajectory(gt_poses, est_poses, est_landmarks):
    """Plots the full optimized trajectory against the Ground Truth."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # 1. Extract true Global Ground Truth position using C = -R^T * t
    gt_x, gt_z = [], []
    for pose in gt_poses:
        R = pose[:3, :3]
        t = pose[:3, 3]
        C = -R.T @ t
        gt_x.append(C[0])
        gt_z.append(C[2])

    # 2. Estimated poses are GTSAM Pose3 (T_camera -> world),
    # so .translation() natively returns the correct global coordinates!
    est_x = [pose.translation()[0] for pose in est_poses.values()]
    est_z = [pose.translation()[2] for pose in est_poses.values()]

    lm_x = [pt[0] for pt in est_landmarks]
    lm_z = [pt[2] for pt in est_landmarks]

    ax.plot(gt_x, gt_z, 'r--o', markersize=2, linewidth=1.5, alpha=0.8)
    ax.plot(est_x, est_z, 'b-o', markersize=2, linewidth=1.5)
    ax.scatter(lm_x, lm_z, c='gray', s=0.5, alpha=0.1)
    gt_line = mlines.Line2D([], [], color='red', marker='o', markersize=4, linestyle='--', linewidth=1.5, alpha=0.8,
                            label='Ground Truth (Keyframes)')
    est_line = mlines.Line2D([], [], color='blue', marker='o', markersize=4, linestyle='-', linewidth=1.5,
                             label='Estimated (BA Keyframes)')
    lm_dot = mlines.Line2D([], [], color='white', marker='o', markerfacecolor='gray', markersize=8,
                           label='3D Landmarks')
    format_plot(ax, "Global Trajectory: Ground Truth vs Optimized Visual Odometry", "X (East) [m]", "Z (North) [m]",
                equal_axes=True)
    ax.legend(handles=[gt_line, est_line, lm_dot], loc='best', frameon=True, facecolor='white', framealpha=0.9)
    plt.show()


def plot_localization_error(keyframes, errors):
    """Plots Euclidean localization error over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(keyframes, errors, color='firebrick', marker='o', markersize=3, linewidth=1.5)

    format_plot(ax, "Keyframe Localization Error (Euclidean Distance)", "Frame Index", "Error (Meters)")
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_first_bundle_with_axes(values, marginals, title):
    """
    Standard zoomed-out 3D plot utilizing GTSAM's native RGB camera axes.
    """
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Scale adjusts the size of the RGB axes
    gtsam_plot.plot_trajectory(1, values, marginals=marginals, scale=0.5)

    ax.set_title(title, fontsize=12, fontweight='semibold')
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    ax.view_init(elev=20, azim=-45)
    plt.tight_layout()
    plt.show()


def plot_first_bundle_3d(values, marginals, keys, min_radius=0.15):
    """
    'Curb-Side View' 3D plot with an adjusted azimuth to prevent label overlap.
    """
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    xs_plt = [values.atPose3(k).translation()[2] for k in keys]
    ys_plt = [values.atPose3(k).translation()[0] for k in keys]
    zs_plt = [-values.atPose3(k).translation()[1] for k in keys]

    ax.plot(xs_plt, ys_plt, zs_plt, color='#1f77b4', linewidth=2.5, label='Optimized camera trajectory')
    ax.scatter(xs_plt, ys_plt, zs_plt, color='#1f77b4', s=80, depthshade=False)

    for i, (x, y, z) in enumerate(zip(xs_plt, ys_plt, zs_plt)):
        ax.text(x, y, z, f'{i}', size=10, zorder=10, color='k', ha='center', va='bottom')

    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    for k, x, y, z in zip(keys, xs_plt, ys_plt, zs_plt):
        cov6x6 = marginals.marginalCovariance(k)
        cov3x3 = np.array([
            [cov6x6[5, 5], cov6x6[5, 3], -cov6x6[5, 4]],
            [cov6x6[3, 5], cov6x6[3, 3], -cov6x6[3, 4]],
            [-cov6x6[4, 5], -cov6x6[4, 3], cov6x6[4, 4]]
        ])

        evals, evecs = np.linalg.eigh(cov3x3)
        radii = np.sqrt(np.maximum(evals, 0))
        radii_visual = np.maximum(radii, min_radius)
        transform = evecs @ np.diag(radii_visual)

        points = np.stack((sphere_x.flatten(), sphere_y.flatten(), sphere_z.flatten()), axis=0)
        transformed = transform @ points

        ell_x = transformed[0, :].reshape(sphere_x.shape) + x
        ell_y = transformed[1, :].reshape(sphere_y.shape) + y
        ell_z = transformed[2, :].reshape(sphere_z.shape) + z

        ax.plot_wireframe(ell_x, ell_y, ell_z, color='#cc8952', alpha=0.4, linewidth=0.5)

    ax.set_title("Initial BA Bundle: Camera Positions and 1σ Covariances from a Roadside View", fontsize=14)
    ax.set_xlabel("\nForward (Z) [m]")
    ax.set_ylabel("\n\n\nSide (X) [m]")
    ax.set_zlabel("\n\nUp (-Y) [m]")

    # Adjusted angle: elev=15, azim=-70 rotates it counter-clockwise slightly
    ax.view_init(elev=15, azim=-70)
    ax.legend(loc='upper left', fontsize=12)
    fig.text(0.5, 0.02, "Visual scaling only: The minimum ellipsoid radius does not alter the underlying covariance values.", ha='center',
             fontsize=9, color='gray')

    plt.tight_layout()
    plt.show()


def plot_pose_graph_2d_initial(initial_values, keys):
    fig, ax = plt.subplots(figsize=(8, 8))
    xs = [initial_values.atPose3(k).translation()[0] for k in keys]
    zs = [initial_values.atPose3(k).translation()[2] for k in keys]

    ax.plot(xs, zs, 'o-', color='#1f77b4', markersize=3, linewidth=1.5)
    ax.set_title("Exercise 6.2: Initial Pose Graph Trajectory", fontsize=14)
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Z (North) [m]")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pose_graph_2d_optimized(optimized_values, keys):
    fig, ax = plt.subplots(figsize=(8, 8))
    o_xs = [optimized_values.atPose3(k).translation()[0] for k in keys]
    o_zs = [optimized_values.atPose3(k).translation()[2] for k in keys]

    ax.plot(o_xs, o_zs, 'o-', color='#ff7f0e', markersize=3, linewidth=1.5)
    ax.set_title("Exercise 6.2: Optimized Pose Graph Trajectory", fontsize=14)
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Z (North) [m]")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pose_graph_2d_covariances(values, marginals, keys, scale=30.0):
    """
    Plots the 2D Global Trajectory with hollow orange covariance ellipses.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = [values.atPose3(k).translation()[0] for k in keys]
    zs = [values.atPose3(k).translation()[2] for k in keys]

    ax.plot(xs, zs, 'o-', color='#1f77b4', markersize=1.5, linewidth=1.0, label="Optimized trajectory")

    for k, x, z in zip(keys, xs, zs):
        cov6x6 = marginals.marginalCovariance(k)
        cov2d = np.array([[cov6x6[3, 3], cov6x6[3, 5]],
                          [cov6x6[5, 3], cov6x6[5, 5]]])

        evals, evecs = np.linalg.eigh(cov2d)
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        width, height = 2 * scale * np.sqrt(evals[0]), 2 * scale * np.sqrt(evals[1])

        # Switched to facecolor='none' to prevent alpha stacking
        ell = Ellipse(xy=(x, z), width=width, height=height, angle=angle,
                      edgecolor='#d95f02', facecolor='none', linewidth=1.2, alpha=0.5)
        ax.add_patch(ell)

    cov_patch = mlines.Line2D([], [], color='#d95f02', alpha=0.5, linewidth=2,
                              label=f'1σ covariance (uncertainty)')
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(0, cov_patch)

    ax.set_title("Exercise 6.2: Optimized Pose Graph Trajectory with Marginal Covariances", fontsize=14)
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Z (North) [m]")
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(handles=handles, loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


def plot_pose_graph_3d_global_covariances(values, marginals, keys, step=50, cov_scale=10.0):
    """
    Plots the full global trajectory in 3D with subsampled wireframe ellipsoids.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    xs = [values.atPose3(k).translation()[0] for k in keys]
    ys = [values.atPose3(k).translation()[1] for k in keys]
    zs = [values.atPose3(k).translation()[2] for k in keys]

    ax.plot(xs, ys, zs, color='#1f77b4', linewidth=1.5, label='Camera Trajectory')

    # Start and End points for context
    ax.scatter(xs[0], ys[0], zs[0], color='green', s=50, label='Start', zorder=5)
    ax.scatter(xs[-1], ys[-1], zs[-1], color='red', s=50, label='End', zorder=5)

    u = np.linspace(0, 2 * np.pi, 15)
    v = np.linspace(0, np.pi, 15)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    # Loop with step to distance the covariances
    for i in range(0, len(keys), step):
        k = keys[i]
        x, y, z = xs[i], ys[i], zs[i]

        cov6x6 = marginals.marginalCovariance(k)
        cov3x3 = cov6x6[3:6, 3:6]

        evals, evecs = np.linalg.eigh(cov3x3)
        radii = cov_scale * np.sqrt(np.maximum(evals, 0))
        transform = evecs @ np.diag(radii)

        points = np.stack((sphere_x.flatten(), sphere_y.flatten(), sphere_z.flatten()), axis=0)
        transformed = transform @ points

        ell_x = transformed[0, :].reshape(sphere_x.shape) + x
        ell_y = transformed[1, :].reshape(sphere_y.shape) + y
        ell_z = transformed[2, :].reshape(sphere_z.shape) + z

        ax.plot_wireframe(ell_x, ell_y, ell_z, color='#cc8952', alpha=0.3, linewidth=0.6)

    ax.set_title(f"Exercise 6.2: Global 3D Trajectory with Covariances (Step={step}, Scale={cov_scale}x)", fontsize=14)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    max_range = np.array([np.ptp(xs), np.ptp(ys), np.ptp(zs)]).max()
    mid_x = np.mean([np.max(xs), np.min(xs)])
    mid_y = np.mean([np.max(ys), np.min(ys)])
    mid_z = np.mean([np.max(zs), np.min(zs)])

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    cov_line = mlines.Line2D([], [], color='#cc8952', label=f'3D Covariance')
    handles, labels = ax.get_legend_handles_labels()
    handles.append(cov_line)

    ax.legend(handles=handles, loc='upper left')
    ax.view_init(elev=35, azim=-55)
    plt.tight_layout()
    plt.show()


def plot_loop_closure_hunter(poses_dict, start_idx=0, end_idx=500, label_step=5):
    """
    Plots a 2D bird's-eye view of the trajectory and annotates the points
    with their keyframe IDs to visually hunt for physical loop closures.
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    xs, zs = [], []

    # Filter and sort keys within the desired range
    valid_keys = [k for k in poses_dict.keys() if start_idx <= k <= end_idx]
    valid_keys.sort()

    for k in valid_keys:
        pose = poses_dict[k]
        t = pose.translation()
        xs.append(t[0])
        zs.append(t[2])

    # Plot the continuous trajectory line
    ax.plot(xs, zs, 'b-', alpha=0.5, linewidth=2, label='Dead-Reckoning Trajectory')

    # Scatter the points and annotate the IDs
    for i, k in enumerate(valid_keys):
        x, z = xs[i], zs[i]

        # Highlight the very first frame to make finding the "start" easy
        if k == 0:
            ax.plot(x, z, 'go', markersize=8, label='Start (Frame 0)')
            ax.text(x + 1.0, z + 1.0, "Start (0)", fontsize=12, fontweight='bold', color='green')
        else:
            ax.plot(x, z, 'ro', markersize=3)

            # Print the ID based on the step size to prevent text overlap
            if i % label_step == 0 or k == valid_keys[-1]:
                ax.text(x + 1.0, z + 1.0, str(k), fontsize=9, color='darkred')

    ax.set_title(f"Loop Closure Hunter (Keyframes {start_idx} to {end_idx})", fontsize=14, fontweight='semibold')
    ax.set_xlabel("X (East) [m]")
    ax.set_ylabel("Z (North) [m]")
    ax.axis('equal')  # CRITICAL: Ensures the loop geometry isn't distorted
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.show()


def plot_consensus_match(img_i, img_n, kp_i, kp_n, matches, mask, f_i, f_n):
    """Section 7.5: Plots the inliers and outliers of the chosen consensus match."""
    inliers = [m for idx, m in enumerate(matches) if mask[idx]]
    outliers = [m for idx, m in enumerate(matches) if not mask[idx]]

    canvas = draw_inliers_outliers(img_i, img_n, kp_i, kp_n, inliers, outliers)

    plt.figure(figsize=(15, 6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title(f"Consensus Match: Frame {f_i} -> {f_n} | Inliers (Orange), Outliers (Cyan)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_pose_graph_snapshots(snapshots_data, keyframes, visual_scale=3.0, de_inflation=150.0):
    """
    Section 7.5: Plots 4 versions of the pose graph along the optimization process.
    De-inflates the covariance by 100.0 to show the 'true' mathematical uncertainty,
    then applies a standard 3-sigma visual scale.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.flatten()

    for i, snap in enumerate(snapshots_data):
        ax = axes[i]
        values = snap['values']
        graph = snap['graph']
        title = snap['title']

        print(f"Calculating marginals for {title}...")
        try:
            marginals = gtsam.Marginals(graph, values)
        except Exception:
            marginals = None

        xs, zs, valid_keys = [], [], []

        # 1. Plot the Trajectory
        for kf in keyframes:
            key = gtsam.symbol('c', kf)
            if values.exists(key):
                pose = values.atPose3(key)
                xs.append(pose.x())
                zs.append(pose.z())
                valid_keys.append(key)

        ax.plot(xs, zs, 'b-', linewidth=2, label='Trajectory')

        # 2. Plot the Covariance Ellipses
        if marginals is not None:
            # Subsample the ellipses so the plot is readable
            step = max(1, len(valid_keys) // 40)
            for j in range(0, len(valid_keys), step):
                key = valid_keys[j]
                x, z = values.atPose3(key).x(), values.atPose3(key).z()

                # A. Get inflated covariance from the graph
                cov6x6 = marginals.marginalCovariance(key)

                # B. De-inflate it back to its true physical scale
                cov6x6_true = cov6x6 / de_inflation

                # Extract X and Z positional covariance
                cov2d = np.array([[cov6x6_true[3, 3], cov6x6_true[3, 5]],
                                  [cov6x6_true[5, 3], cov6x6_true[5, 5]]])

                evals, evecs = np.linalg.eigh(cov2d)
                angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))

                # C. Apply a standard visual scale (3.0 represents a 3-sigma confidence interval)
                width = 2 * visual_scale * np.sqrt(max(evals[0], 0))
                height = 2 * visual_scale * np.sqrt(max(evals[1], 0))

                ell = Ellipse(xy=(x, z), width=width, height=height, angle=angle,
                              edgecolor='#d95f02', facecolor='none', linewidth=1.2, alpha=0.5)
                ax.add_patch(ell)

        # (Assuming format_plot is defined elsewhere in your visualization.py)
        format_plot(ax, title, "X [m]", "Z [m]", equal_axes=True)
        ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

def get_true_gt_trajectory(poses_4x4, keyframes):
    """Internal helper to extract the true global center C = -R^T * t."""
    gt_x, gt_z = [], []
    for kf in keyframes:
        R = poses_4x4[kf][:3, :3]
        t = poses_4x4[kf][:3, 3]
        C = -R.T @ t
        gt_x.append(C[0])
        gt_z.append(C[2])
    return gt_x, gt_z


def plot_trajectory_comparison(poses_4x4, base_values, final_values, keyframes):
    """Section 7.5: Compares Ground Truth vs Dead Reckoning vs Loop Closures."""
    gt_x, gt_z = get_true_gt_trajectory(poses_4x4, keyframes)

    base_x = [base_values.atPose3(gtsam.symbol('c', kf)).x() for kf in keyframes]
    base_z = [base_values.atPose3(gtsam.symbol('c', kf)).z() for kf in keyframes]

    final_x = [final_values.atPose3(gtsam.symbol('c', kf)).x() for kf in keyframes]
    final_z = [final_values.atPose3(gtsam.symbol('c', kf)).z() for kf in keyframes]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(gt_x, gt_z, 'g-', linewidth=2, label='Ground Truth')
    ax.plot(base_x, base_z, 'r--', linewidth=2, label='Dead Reckoning (No Loops)')
    ax.plot(final_x, final_z, 'b-', linewidth=2, label='Optimized (With Loops)')

    format_plot(ax, "Global Trajectory Comparison", "X [m]", "Z [m]", equal_axes=True)
    ax.legend()
    ax.grid(True, linestyle=':')
    plt.show()


def plot_absolute_location_error_comparison(poses_4x4, base_values, final_values, keyframes):
    """Section 7.5: Plots the absolute error magnitude over time."""
    gt_x, gt_z = get_true_gt_trajectory(poses_4x4, keyframes)
    base_err, final_err = [], []

    for i, kf in enumerate(keyframes):
        key = gtsam.symbol('c', kf)
        p_base = base_values.atPose3(key)
        p_final = final_values.atPose3(key)

        base_err.append(np.sqrt((gt_x[i] - p_base.x()) ** 2 + (gt_z[i] - p_base.z()) ** 2))
        final_err.append(np.sqrt((gt_x[i] - p_final.x()) ** 2 + (gt_z[i] - p_final.z()) ** 2))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(keyframes, base_err, 'r--', label='Without Loop Closures')
    ax.plot(keyframes, final_err, 'b-', label='With Loop Closures')

    format_plot(ax, "Absolute Euclidean Location Error", "Keyframe Index", "Error [m]")
    ax.legend()
    ax.grid(True, linestyle=':')
    plt.show()
    return base_err, final_err


def plot_uncertainty_size_comparison(base_marginals, final_marginals, keyframes):
    """Section 7.5: Plots the trace of the positional covariance (uncertainty size)."""
    base_unc, final_unc = [], []
    for kf in keyframes:
        key = gtsam.symbol('c', kf)

        # Extract 3x3 positional block and compute Trace
        cov_base = base_marginals.marginalCovariance(key)[3:6, 3:6]
        base_unc.append(np.trace(cov_base))

        cov_final = final_marginals.marginalCovariance(key)[3:6, 3:6]
        final_unc.append(np.trace(cov_final))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(keyframes, base_unc, 'r--', label='Without Loop Closures')
    ax.plot(keyframes, final_unc, 'b-', label='With Loop Closures')

    format_plot(ax, "Location Uncertainty Size (Trace of Positional Covariance)", "Keyframe Index", "Trace [m^2]")
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, linestyle=':', which='both')
    plt.show()