import gtsam
import numpy as np
import matplotlib.pyplot as plt

from tracking_database import TrackingDB
from dataset_utils import read_ground_truth, read_cameras
from bundle_adjustment import create_gtsam_calibration
from pose_graph import build_pose_graph

from ex6 import q6_1
import loop_closure

DATA_PATH = '../dataset/sequences/00/'
POSES_PATH = '../dataset/poses/00.txt'
DB_NAME = "kitti_00_database"


def plot_mahalanobis_for_target(target_key, optimized_values, accumulated_covs):
    """Plots the computed distances. We want to see a dip near frame 0!"""
    frame_ids = []
    distances = []

    pose_n = optimized_values.atPose3(target_key)

    for candidate_key, cov_matrix in accumulated_covs.items():
        if not optimized_values.exists(candidate_key):
            continue

        pose_i = optimized_values.atPose3(candidate_key)

        # Mahalanobis Math
        relative_pose = pose_i.between(pose_n)
        error_vector = gtsam.Pose3.Logmap(relative_pose)
        info_matrix = np.linalg.pinv(cov_matrix)
        md_squared = error_vector.T @ info_matrix @ error_vector

        frame_ids.append(gtsam.symbolIndex(candidate_key))
        distances.append(md_squared)

    plt.figure(figsize=(10, 5))
    plt.plot(frame_ids, distances, 'b.', alpha=0.6, label="Mahalanobis $D^2$")
    plt.axhline(y=16.81, color='r', linestyle='--', label="Chi-Square Threshold (16.81)")

    plt.yscale('log')
    plt.title(f"Mahalanobis Distances from Loop Frame (Keyframe {gtsam.symbolIndex(target_key)})")
    plt.xlabel("Candidate Historical Frame ID")
    plt.ylabel("Squared Mahalanobis Distance (Log Scale)")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    print("Loading database and building base Pose Graph (Ex 6)...")
    db = TrackingDB()
    db.load(DB_NAME)
    K, _, M2 = read_cameras(DATA_PATH)
    gtsam_calib = create_gtsam_calibration(K, M2)
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)

    # We only need to run Ex 6 up to frame ~1500 to save computation time for this test
    # (Since we know the loop is at 1452)
    truncated_poses = poses_4x4[:1550]

    # Build Graph with INFLATED covariances (Make sure ex6.py multiplies rel_cov by 100!)
    relative_measurements, keyframes = q6_1(db, truncated_poses, gtsam_calib)
    graph, initial_guess = build_pose_graph(relative_measurements)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_guess)
    optimized_values = optimizer.optimize()

    # Find the keyframe index that represents the loop at frame 1452
    target_kf = min(keyframes, key=lambda x: abs(x - 1479))
    target_key = gtsam.symbol('c', target_kf)

    print(f"\n--- Testing Loop Candidate Detection for Frame {target_kf} ---")

    # 1. Accumulate Covariances
    accumulated_covs = loop_closure.get_shortest_paths_covariances(graph, target_key)

    # 2. Filter Candidates
    candidates = loop_closure.filter_candidates_by_mahalanobis(
        optimized_values, target_key, accumulated_covs,
        mahalanobis_thresh=16.81, min_frame_gap=100
    )

    if candidates:
        print(f"SUCCESS! Found {len(candidates)} viable candidates below 16.81 threshold.")
        for key, md in candidates[:5]:
            print(f"  -> Frame {gtsam.symbolIndex(key)} | Mahalanobis D^2: {md:.2f}")
    else:
        print("FAILURE: No candidates found. Covariance might need more inflation.")

    # 3. Plot to verify visually
    plot_mahalanobis_for_target(target_key, optimized_values, accumulated_covs)


if __name__ == "__main__":
    main()