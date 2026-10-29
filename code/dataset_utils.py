"""
Dataset Utilities Module
========================

Provides specialized I/O operations and parsers for the KITTI Vision Benchmark Suite.
This module abstracts the file system interactions required to load stereo image pairs,
camera calibration matrices, and ground truth poses. It ensures raw physical data is
properly formatted and synchronized before entering the visual odometry pipeline.

Key Responsibilities:
- Parsing 4x4 homogeneous transformation matrices from ground truth text files.
- Iterative loading of synchronized grayscale stereo image pairs.
- Extracting and normalizing intrinsic and extrinsic camera matrices from calibration files.
"""

import cv2
import numpy as np


def read_ground_truth(filepath, num_frames=None, return_transforms=False):
    """
    Parses the KITTI 00.txt pose file.
    Can return either 2D trajectory coordinates (for plotting)
    or full 4x4 homogeneous transformation matrices (for 3D projection).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    if num_frames is None:
        num_frames = len(lines)

    gt_x, gt_z = [], []
    transforms_4x4 = []

    limit = min(num_frames, len(lines))
    for i in range(limit):
        pose = np.array([float(x) for x in lines[i].split()]).reshape(3, 4)

        # 1. Build the 4x4 Transformation Matrix (For Ex 4 Projections)
        T_w_c = np.eye(4)
        T_w_c[:3, :] = pose
        transforms_4x4.append(T_w_c)

        # 2. Extract true global camera position (For Ex 3 Plotting)
        R = pose[:, :3]
        t = pose[:, 3]
        cam_pos = -R.T @ t

        gt_x.append(cam_pos[0])
        gt_z.append(cam_pos[2])

    if return_transforms:
        return transforms_4x4

    return gt_x, gt_z


def read_stereo_pair(data_path, idx):
    """Reads a stereo pair of images from the KITTI dataset."""
    img_name = f"{idx:06d}.png"
    img1 = cv2.imread(data_path + 'image_0/' + img_name, 0)  # 0 = grayscale
    img2 = cv2.imread(data_path + 'image_1/' + img_name, 0)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"Images not found at {data_path}")
    return img1, img2


def read_cameras(data_path):
    """
    Reads camera calibration according to the provided reference style.
    Returns:
        k: 3x3 Intrinsic matrix
        m1: 3x4 Normalized projection matrix for Left camera
        m2: 3x4 Normalized projection matrix for Right camera
    """
    with open(data_path + 'calib.txt') as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]

    m1_raw = np.array([float(i) for i in l1]).reshape(3, 4)
    m2_raw = np.array([float(i) for i in l2]).reshape(3, 4)

    # Extract the Intrinsic Matrix K
    k = m1_raw[:, :3]

    # Normalize the matrices by removing K
    m1 = np.linalg.inv(k) @ m1_raw
    m2 = np.linalg.inv(k) @ m2_raw

    return k, m1, m2
