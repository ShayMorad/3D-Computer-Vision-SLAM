"""
ex7.py
Main execution file for VAN Exercise 6: Pose Graph
"""
from tracking_database import TrackingDB
from dataset_utils import read_ground_truth, read_cameras
from bundle_adjustment import create_gtsam_calibration
from pose_graph import build_pose_graph, extract_relative_constraint, generate_sequential_measurements
from loop_closure import *
from visualization import plot_consensus_match, plot_pose_graph_snapshots, plot_trajectory_comparison, \
    plot_absolute_location_error_comparison, plot_uncertainty_size_comparison

MAHALANOBIS_THRESH = 16.81
COV_SCALE = 1500.0
REL_COV_SCALE = 100
LOOP_COOLDOWN = 5

DATA_PATH = '../dataset/sequences/00/'
POSES_PATH = '../dataset/poses/00.txt'
DB_NAME = "kitti_00_database"


def q7_1(global_graph, optimized_values, current_key):
    """
    Section 7.1: Detect Loop Closure Candidates
    Accumulates covariance via Dijkstra and filters via Mahalanobis Distance.
    """
    accumulated_covs = get_shortest_paths_covariances(global_graph, current_key)

    # Using 16.81 (99% Chi-Square threshold) since covariance was inflated in Ex 6
    candidates = filter_candidates_by_mahalanobis(
        optimized_values, current_key, accumulated_covs,
        mahalanobis_thresh=MAHALANOBIS_THRESH, min_frame_gap=150
    )
    return candidates


def q7_2(candidates, current_kf, K, P1, P2, M2):
    """
    Section 7.2: Visual Validation (Consensus)
    Iterates through candidates and runs the 4-way stereo RANSAC pipeline.
    Returns the first mathematically and visually verified loop.
    """
    for candidate_key, md in candidates:
        candidate_kf = gtsam.symbolIndex(candidate_key)

        is_valid, loop_data = validate_and_estimate_loop(
            DATA_PATH, K, P1, P2, M2, candidate_kf, current_kf, min_inliers=30
        )

        if is_valid:
            print(f"LOOP FOUND: Frame {current_kf} -> Frame {candidate_kf} | Mahalanobis D^2: {md:.2f}")
            return True, candidate_kf, loop_data

    return False, None, None


def q7_3(gtsam_calib, loop_data, candidate_kf, current_kf):
    # 1. Run the mini-BA
    opt_vals, marginals, key_i, key_n = calculate_loop_closure_bundle(
        gtsam_calib, loop_data, candidate_kf, current_kf
    )

    # 2. Extract strictly via Information Matrix Conditioning (Course-Compliant)
    rel_pose, conditional_cov = extract_relative_constraint(opt_vals, marginals, key_i, key_n)

    # 3. Inflate by same magnitude as sequential odometry (e.g., Ex 6 scaling)
    rel_cov_inflated = conditional_cov * REL_COV_SCALE

    return rel_pose, rel_cov_inflated


def q7_4(global_graph, current_optimized_values, current_key, candidate_key, rel_pose, rel_cov):
    """
    Section 7.4: Update Pose Graph
    Injects the new constraint and re-optimizes the global trajectory.
    """
    noise_model = gtsam.noiseModel.Gaussian.Covariance(rel_cov)

    # Wrap in Cauchy robust kernel to protect against catastrophic false positives
    robust_noise_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(2.0), noise_model)

    # Inject Edge
    global_graph.add(gtsam.BetweenFactorPose3(candidate_key, current_key, rel_pose, robust_noise_model))

    # Re-optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(global_graph, current_optimized_values)
    new_optimized_values = optimizer.optimize()

    return global_graph, new_optimized_values


def q7_5(successful_loops, graph_snapshots, base_graph, base_values,
         final_graph, final_values, keyframes, poses_4x4, data_path):
    """
    Section 7.5: Final Analysis and Plotting
    """
    print("\n==========================================================")
    print("--- Q7.5: Generating Final Assignment Plots ---")

    print(f"Total successful loop closures detected: {len(successful_loops)}")
    if not successful_loops:
        print("No loops found to plot.")
        return

    # 1. Consensus Match
    loop = successful_loops[1]
    f_n, f_i = loop['current_kf'], loop['candidate_kf']

    img_n_l = cv2.imread(f"{data_path}image_0/{f_n:06d}.png", cv2.IMREAD_GRAYSCALE)
    img_i_l = cv2.imread(f"{data_path}image_0/{f_i:06d}.png", cv2.IMREAD_GRAYSCALE)

    kp_i = loop['match_data']['kp_i_l']
    kp_n = loop['match_data']['kp_n_l']
    matches = loop['match_data']['matches_4way']
    mask = loop['match_data']['mask']

    plot_consensus_match(img_i_l, img_n_l, kp_i, kp_n, matches, mask, f_i, f_n)

    # 2. Plot 4 Versions of the Pose Graph
    if len(graph_snapshots) < 4:
        while len(graph_snapshots) < 4:
            graph_snapshots.append(graph_snapshots[-1])

    selected_snapshots = [
        graph_snapshots[0],  # Snapshot 1: Baseline
        graph_snapshots[1],  # Snapshot 2: First Loop Correction
        graph_snapshots[len(graph_snapshots) // 2],  # Snapshot 3: Midway Progress
        graph_snapshots[-1]  # Snapshot 4: Final State
    ]
    selected_snapshots[-1]['title'] = f"Final Graph (All Loops)"

    # 2. Plot 4 Versions of the Pose Graph
    plot_pose_graph_snapshots(selected_snapshots, keyframes)

    # 3. Trajectory Comparison
    plot_trajectory_comparison(poses_4x4, base_values, final_values, keyframes)

    # 4. Absolute Location Error
    base_errors, final_errors = plot_absolute_location_error_comparison(poses_4x4, base_values, final_values, keyframes)
    print("\n--- Quantitative Results ---")
    print(f"Mean Error (No Loops): {np.mean(base_errors):.2f}m -> (With Loops): {np.mean(final_errors):.2f}m")
    print(f"Final Error (No Loops): {base_errors[-1]:.2f}m -> (With Loops): {final_errors[-1]:.2f}m")

    # 5. Location Uncertainty Size
    base_marginals = gtsam.Marginals(base_graph, base_values)
    final_marginals = gtsam.Marginals(final_graph, final_values)

    plot_uncertainty_size_comparison(base_marginals, final_marginals, keyframes)

    print("\nExercise 7 Pipeline Complete!")


def main():
    print("Initializing VAN Exercise 7: Loop Closures Pipeline...\n")

    # 1. Load Core Data
    db = TrackingDB()
    db.load(DB_NAME)
    K, M1, M2 = read_cameras(DATA_PATH)
    P1 = K @ M1
    P2 = K @ M2
    gtsam_calib = create_gtsam_calibration(K, M2)
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)

    # 2. Bootstrap the Baseline Pose Graph (Ex 6 Logic)
    print("--- Bootstrapping Global Pose Graph ---")
    relative_measurements, keyframes = generate_sequential_measurements(db, poses_4x4, gtsam_calib, dist_threshold=5.0,
                                                                        cov_inflation=COV_SCALE)

    global_graph, initial_guess = build_pose_graph(relative_measurements)

    optimizer = gtsam.LevenbergMarquardtOptimizer(global_graph, initial_guess)
    current_optimized_values = optimizer.optimize()
    dynamic_adj_list = build_initial_adjacency_list(global_graph)

    base_graph = gtsam.NonlinearFactorGraph(global_graph)
    base_values = gtsam.Values(current_optimized_values)

    # NEW: Initialize snapshots list with the Base Graph (Snapshot 1)
    graph_snapshots = [{
        'title': "Snapshot 1: Dead Reckoning (0 Loops)",
        'values': gtsam.Values(base_values),
        'graph': gtsam.NonlinearFactorGraph(base_graph)
    }]

    # 3. Online Loop Closure Execution
    print("\n--- Initiating Online Loop Closure Execution ---")
    successful_loops = []
    loop_cooldown = 0

    for n_idx, current_kf in enumerate(keyframes):
        if n_idx < 100: continue
        if loop_cooldown > 0:
            loop_cooldown -= 1
            continue

        current_key = gtsam.symbol('c', current_kf)
        accumulated_covs = get_shortest_paths_covariances(dynamic_adj_list, current_key)

        candidates = filter_candidates_by_mahalanobis(
            current_optimized_values, current_key, accumulated_covs,
            mahalanobis_thresh=MAHALANOBIS_THRESH, min_frame_gap=150
        )
        if not candidates: continue

        valid_loop, candidate_kf, loop_data = q7_2(candidates, current_kf, K, P1, P2, M2)
        if not valid_loop: continue

        candidate_key = gtsam.symbol('c', candidate_kf)
        rel_pose, rel_cov = q7_3(gtsam_calib, loop_data, candidate_kf, current_kf)

        global_graph, current_optimized_values = q7_4(
            global_graph, current_optimized_values, current_key, candidate_key, rel_pose, rel_cov
        )

        dynamic_adj_list[current_key].append((candidate_key, rel_cov))
        dynamic_adj_list[candidate_key].append((current_key, rel_cov))

        successful_loops.append({
            'current_kf': current_kf,
            'candidate_kf': candidate_kf,
            'match_data': loop_data
        })
        loop_cooldown = LOOP_COOLDOWN

        num_loops = len(successful_loops)
        graph_snapshots.append({
            'title': f"Snapshot: After {num_loops} Loop(s)",
            'values': gtsam.Values(current_optimized_values),
            'graph': gtsam.NonlinearFactorGraph(global_graph)
        })

    # 4. Final Analysis and Plotting
    q7_5(successful_loops, graph_snapshots, base_graph, base_values,
         global_graph, current_optimized_values, keyframes, poses_4x4, DATA_PATH)


if __name__ == '__main__':
    main()
