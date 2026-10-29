"""
ex6.py
Main execution file for VAN Exercise 6: Pose Graph
"""
import numpy as np
import gtsam
from tracking_database import TrackingDB
from dataset_utils import read_ground_truth, read_cameras
from bundle_adjustment import create_gtsam_calibration, select_keyframes, run_bundle_adjustment
from pose_graph import extract_relative_constraint, build_pose_graph
from visualization import plot_first_bundle_3d, plot_pose_graph_2d_initial, plot_pose_graph_2d_covariances, \
    plot_first_bundle_with_axes, plot_pose_graph_2d_optimized, plot_pose_graph_3d_global_covariances

DATA_PATH = '../dataset/sequences/00/'
POSES_PATH = '../dataset/poses/00.txt'
DB_NAME = "kitti_00_database"


def q6_1(db, poses_4x4, gtsam_calib):
    """Answers Section 6.1: Extracts relative poses and marginal covariances from local BA."""
    print("--- Q 6.1: Extracting Relative Constraints ---")
    keyframes = select_keyframes(poses_4x4, dist_threshold=5.0, max_frames=15)
    relative_measurements = []

    for i in range(len(keyframes) - 1):
        kf_0, kf_k = keyframes[i], keyframes[i + 1]
        bundle_frames = list(range(kf_0, kf_k + 1))

        # Local optimization anchored to the origin
        _, opt_vals, frame_keys, graph, _, _ = run_bundle_adjustment(
            db, bundle_frames, poses_4x4, gtsam_calib, pixel_sigma=2, make_relative=True
        )
        marginals = gtsam.Marginals(graph, opt_vals)

        # Extract constraints
        key_0 = gtsam.symbol('c', kf_0)
        key_k = gtsam.symbol('c', kf_k)
        rel_pose, rel_cov = extract_relative_constraint(opt_vals, marginals, key_0, key_k)

        relative_measurements.append({
            'kf_0': kf_0, 'kf_k': kf_k,
            'relative_pose': rel_pose, 'covariance': rel_cov
        })

        # Explicit requirements for the FIRST bundle
        if i == 0:
            print(f"\n[First Bundle: Frames {kf_0} to {kf_k}]")
            print("Relative Pose (T_0->k):\n", rel_pose)
            print("\nMarginal Covariance Matrix (6x6):\n", np.array_str(rel_cov, precision=4, suppress_small=True))

            # Plot 1: Standard 3D with RGB camera axes
            plot_first_bundle_with_axes(opt_vals, marginals, "First BA Bundle: Local Locations with Covariances")

            # Plot 2: Clean horizontal Curb-Side view
            plot_first_bundle_3d(opt_vals, marginals, frame_keys)

    return relative_measurements, keyframes


def q6_2(relative_measurements, keyframes):
    """Answers Section 6.2: Builds and optimizes the global Pose Graph."""
    print("\n--- Q 6.2: Global Pose Graph Optimization ---")
    graph, initial_guess = build_pose_graph(relative_measurements)

    # Optimize Global Graph
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_guess)
    optimized_values = optimizer.optimize()

    # Error Printing
    print(f"Graph Error BEFORE Optimization: {graph.error(initial_guess):.6e}")
    print(f"Graph Error AFTER Optimization:  {graph.error(optimized_values):.6e}")

    # Global Plotting
    graph_keys = [gtsam.symbol('c', kf) for kf in keyframes]

    plot_pose_graph_2d_initial(initial_guess, graph_keys)
    plot_pose_graph_2d_optimized(optimized_values, graph_keys)

    final_marginals = gtsam.Marginals(graph, optimized_values)
    plot_pose_graph_2d_covariances(optimized_values, final_marginals, graph_keys, scale=10.0)
    plot_pose_graph_3d_global_covariances(optimized_values, final_marginals, graph_keys, step=50, cov_scale=10.0)

def main():
    db = TrackingDB()
    db.load(DB_NAME)
    K, _, M2 = read_cameras(DATA_PATH)
    gtsam_calib = create_gtsam_calibration(K, M2)
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)

    # Execute 6.1 and pass the extracted constraints to 6.2
    relative_measurements, keyframes = q6_1(db, poses_4x4, gtsam_calib)
    q6_2(relative_measurements, keyframes)

if __name__ == "__main__":
    main()