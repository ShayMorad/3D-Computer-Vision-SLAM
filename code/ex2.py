"""
ex2.py
Main execution file for VAN Exercise 2.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from visualization import format_plot, draw_inliers_outliers
from dataset_utils import read_stereo_pair, read_cameras
from geometry import triangulate_points_dlt
from feature_tracking import extract_features, match_features

DATA_PATH = '../dataset/sequences/00/'
CURRENT_ALGO = "ORB"


def q2_1(img1, img2):
    """Answers Q 2.1: Geometric Outlier Rejection Pattern."""
    print("--- Q 2.1 ---")
    kp1, des1 = extract_features(img1, algorithm=CURRENT_ALGO)
    kp2, des2 = extract_features(img2, algorithm=CURRENT_ALGO)
    matches = match_features(des1, des2, algorithm=CURRENT_ALGO)

    # Calculate deviations
    deviations = np.array([abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in matches])

    # Analytics
    outliers_mask = deviations > 2
    print(f"Total Matches: {len(matches)}")
    print(f"Matches discarded (> 2px): {np.sum(outliers_mask)}")
    print(f"Outlier Percentage: {(np.sum(outliers_mask) / len(matches)) * 100:.2f}%\n")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(deviations, bins=50, edgecolor='black', alpha=0.7)
    format_plot(ax, 'Figure 1: Deviation from Rectified Stereo Pattern', 'Deviation [pixels]', 'Number of matches',
                equal_axes=False)
    plt.show()

    return kp1, kp2, matches, deviations


def q2_2(img1, img2, kp1, kp2, matches, deviations):
    """Answers Q 2.2: Visualizing Inliers and Outliers."""
    print("--- Q 2.2 ---")

    # Split into lists
    inliers = [matches[i] for i, dev in enumerate(deviations) if dev <= 2]
    outliers = [matches[i] for i, dev in enumerate(deviations) if dev > 2]

    # Use library function for visualization
    canvas = draw_inliers_outliers(img1, img2, kp1, kp2, inliers, outliers)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    format_plot(ax, 'Figure 2: Geometric Filter (Orange=Inliers, Cyan=Outliers)',
                'X [pixels]', 'Y [pixels]', equal_axes=True)
    plt.show()

    return inliers


def q2_3(kp1, kp2, inliers):
    """Answers Q 2.3: Manual vs OpenCV Triangulation."""
    print("--- Q 2.3 ---")

    K, M1, M2 = read_cameras(DATA_PATH)

    # Reconstruct the raw projection matrices (P = K * M)
    # Because our keypoints are in raw pixels, our matrices must be in raw pixels
    P1 = K @ M1
    P2 = K @ M2

    # 2. Extract Nx2 arrays of (x, y) coordinates for our inliers
    pts1 = np.array([kp1[m.queryIdx].pt for m in inliers])
    pts2 = np.array([kp2[m.trainIdx].pt for m in inliers])

    # 3. Manual Triangulation using the reconstructed P matrices
    pts3D_manual = triangulate_points_dlt(pts1, pts2, P1, P2)

    # 4. OpenCV Triangulation
    pts3D_cv_homog = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3D_cv_homog /= pts3D_cv_homog[3, :]
    pts3D_cv = pts3D_cv_homog[:3, :].T

    # 5. Compare the results
    distances = np.linalg.norm(pts3D_manual - pts3D_cv, axis=1)
    median_distance = np.median(distances)
    print(f"Median distance between Manual and OpenCV 3D points: {median_distance:.4e} meters")

    # 6. Plot the 3D Point Cloud
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    X = pts3D_cv[:, 0]
    Y = -pts3D_cv[:, 1]  # Invert Y for visualization
    Z = pts3D_cv[:, 2]

    ax.scatter(X, Z, Y, c=Z, cmap='viridis', s=10, alpha=0.7)

    # Labeling
    ax.set_title('Figure 3: Triangulated 3D Point Cloud')
    ax.set_xlabel('X [meters] (Left/Right)')
    ax.set_ylabel('Z [meters] (Depth)')
    ax.set_zlabel('Y [meters] (Up/Down)')

    max_range = np.array([X.max() - X.min(), Z.max() - Z.min(), Y.max() - Y.min()]).max() / 2.0
    mid_x = (X.max() + X.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_z - max_range, mid_z + max_range)
    ax.set_zlim(mid_y - max_range, mid_y + max_range)

    plt.tight_layout()
    plt.show()


def q2_4():
    """Answers Q 2.4: Sequential Processing (Matches and 3D Points)."""
    print("--- Q 2.4 ---")

    K, M1, M2 = read_cameras(DATA_PATH)
    P1 = K @ M1
    P2 = K @ M2

    # Run this process (matching and triangulation) over a few pairs of images
    for idx in range(100, 103):
        print(f"\nProcessing Frame {idx}...")
        img1, img2 = read_stereo_pair(DATA_PATH, idx)

        # 1. Extract and Match
        kp1, des1 = extract_features(img1, algorithm=CURRENT_ALGO)
        kp2, des2 = extract_features(img2, algorithm=CURRENT_ALGO)
        matches = match_features(des1, des2, algorithm=CURRENT_ALGO)

        # 2. Apply the Vertical Geometric Filter from 2.2
        deviations = np.array([abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) for m in matches])
        inliers = [matches[i] for i, dev in enumerate(deviations) if dev <= 2]

        # 3. Triangulate
        pts1 = np.array([kp1[m.queryIdx].pt for m in inliers])
        pts2 = np.array([kp2[m.trainIdx].pt for m in inliers])
        pts3D = triangulate_points_dlt(pts1, pts2, P1, P2)

        infinities = np.isinf(pts3D[:, 2]) | (np.abs(pts3D[:, 2]) > 1e6)
        num_infinities = np.sum(infinities)

        if num_infinities > 0:
            print(f"   [WARNING] Mathematical Singularity Detected!")
            print(f"   -> {num_infinities} points projected to infinity (disparity ~ 0).")
            print(f"   -> Safely discarding these points to prevent plotting crashes.")

        valid_mask = (
                ~infinities &
                (pts3D[:, 2] > 0) &
                (pts3D[:, 2] < 80) &
                (pts3D[:, 1] > -5) &
                (pts3D[:, 1] < 5)
        )

        pts3D = pts3D[valid_mask]

        # --- VISUALIZATION 1: The 2D Matches ---
        img_matches = cv2.drawMatches(
            img1, kp1, img2, kp2, inliers, None,
            matchColor=(0, 255, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        fig_2d, ax_2d = plt.subplots(figsize=(15, 5))
        ax_2d.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        ax_2d.set_title(f'Frame {idx}: 2D Inlier Matches')
        plt.tight_layout()
        plt.show()

        # --- VISUALIZATION 2: The 3D Point Cloud ---
        fig_3d = plt.figure(figsize=(10, 6))
        ax_3d = fig_3d.add_subplot(111, projection='3d')

        X = pts3D[:, 0]
        Y = -pts3D[:, 1]  # Invert Y for visualization
        Z = pts3D[:, 2]

        ax_3d.scatter(X, Z, Y, c=Z, cmap='viridis', s=10, alpha=0.7)
        ax_3d.set_title(f'Raw 3D Point Cloud - Frame {idx}')
        ax_3d.set_xlabel('X [m]')
        ax_3d.set_ylabel('Z [m] (Depth)')
        ax_3d.set_zlabel('Y [m] (Up/Down)')

        # Equalize axes roughly
        max_r = np.array([X.max() - X.min(), Z.max() - Z.min(), Y.max() - Y.min()]).max() / 2.0
        mx, mz, my = (X.max() + X.min()) / 2, (Z.max() + Z.min()) / 2, (Y.max() + Y.min()) / 2
        ax_3d.set_xlim(mx - max_r, mx + max_r)
        ax_3d.set_ylim(mz - max_r, mz + max_r)
        ax_3d.set_zlim(my - max_r, my + max_r)

        plt.tight_layout()
        plt.show()


def main():
    # Q2.1
    img1, img2 = read_stereo_pair(DATA_PATH, 0)
    kp1, kp2, matches, deviations = q2_1(img1, img2)

    # Q2.2
    inliers = q2_2(img1, img2, kp1, kp2, matches, deviations)

    # Q2.3
    q2_3(kp1, kp2, inliers)

    # Q2.4
    q2_4()


if __name__ == "__main__":
    main()
