"""
Global Trajectory Optimization Back-End (Pose Graph)
====================================================

Implements the macro-scale global mapping and optimization engine.
By abstracting massive local point clouds into concise relative pose-pose constraints,
this module chains sequential keyframes into a highly efficient, unified global trajectory.
It leverages marginal covariances extracted from the local back-end to form mathematically
rigorous constraint weights.

Key Responsibilities:
- Extracting relative 3D transformations between locally optimized keyframes.
- Extracting 6x6 marginal covariance matrices to quantify relative spatial uncertainty.
- Constructing the global sequential Pose Graph using BetweenFactorPose3.
- Executing final global trajectory optimization via Levenberg-Marquardt to minimize
  long-term dead-reckoning drift.
"""

import gtsam
import numpy as np

from bundle_adjustment import select_keyframes, run_bundle_adjustment


def extract_relative_constraint(optimized_values, marginals, key_0, key_k):
    """
    Extracts the relative transformation and conditional uncertainty between two keyframes.
    Calculates P(c_k | c_0) using the Information Matrix pipeline.
    """
    pose_0 = optimized_values.atPose3(key_0)
    pose_k = optimized_values.atPose3(key_k)

    # Calculate relative pose Delta c
    relative_pose = pose_0.between(pose_k)

    # 1. Extract the 12x12 Joint Marginal Covariance matrix (Sigma_1N)
    keys = gtsam.KeyVector()
    keys.append(key_0)
    keys.append(key_k)
    joint_cov = marginals.jointMarginalCovariance(keys).fullMatrix()

    # 2. Convert to Information Matrix (inv. -> Omega_1,N)
    # The Information matrix is the inverse of the Covariance matrix.
    joint_info = np.linalg.inv(joint_cov)

    # 3. Conditioning in Information Form (cond. -> Omega_N|1)
    # To find P(c_k | c_0), we just extract the information block for c_k.
    # c_0 occupies indices 0:6, and c_k occupies indices 6:12.
    conditional_info = joint_info[6:12, 6:12]

    # 4. Convert back to Covariance Matrix (inv. -> Sigma_N|1)
    # The Conditional Covariance is the inverse of the Conditional Information.
    conditional_cov = np.linalg.inv(conditional_info)

    return relative_pose, conditional_cov


def build_pose_graph(relative_measurements):
    """
    Constructs the global Pose Graph using chained relative constraints.
    """
    graph = gtsam.NonlinearFactorGraph()
    initial_guess = gtsam.Values()

    # Anchor the first node to the origin to eliminate gauge freedom
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
    key_0 = gtsam.symbol('c', relative_measurements[0]['kf_0'])
    graph.add(gtsam.PriorFactorPose3(key_0, gtsam.Pose3(), prior_noise))

    current_pose = gtsam.Pose3()
    initial_guess.insert(key_0, current_pose)

    # Chain the measurements
    for meas in relative_measurements:
        k1 = gtsam.symbol('c', meas['kf_0'])
        k2 = gtsam.symbol('c', meas['kf_k'])

        noise_model = gtsam.noiseModel.Gaussian.Covariance(meas['covariance'])

        # Add relative constraint
        graph.add(gtsam.BetweenFactorPose3(k1, k2, meas['relative_pose'], noise_model))

        # Kinematic chaining for the initial guess
        current_pose = current_pose.compose(meas['relative_pose'])
        initial_guess.insert(k2, current_pose)

    return graph, initial_guess


def generate_sequential_measurements(db, poses_4x4, gtsam_calib, dist_threshold=5.0, cov_inflation=1.0):
    """
    Pure logic function to generate relative constraints across the entire dataset.
    Abstracts the core loop of Ex 6 so it can be reused silently in Ex 7.
    """
    keyframes = select_keyframes(poses_4x4, dist_threshold=dist_threshold, max_frames=15)
    relative_measurements = []

    for i in range(len(keyframes) - 1):
        kf_0, kf_k = keyframes[i], keyframes[i + 1]
        bundle_frames = list(range(kf_0, kf_k + 1))

        # Run local bundle adjustment
        _, opt_vals, _, graph, _, _ = run_bundle_adjustment(
            db, bundle_frames, poses_4x4, gtsam_calib, pixel_sigma=2.0, make_relative=True
        )
        marginals = gtsam.Marginals(graph, opt_vals)

        # Extract constraint
        key_0 = gtsam.symbol('c', kf_0)
        key_k = gtsam.symbol('c', kf_k)
        rel_pose, rel_cov = extract_relative_constraint(opt_vals, marginals, key_0, key_k)

        # Explicitly apply the covariance inflation to handle dead-reckoning overconfidence
        inflated_cov = rel_cov * cov_inflation

        relative_measurements.append({
            'kf_0': kf_0, 'kf_k': kf_k,
            'relative_pose': rel_pose, 'covariance': inflated_cov
        })

    return relative_measurements, keyframes
