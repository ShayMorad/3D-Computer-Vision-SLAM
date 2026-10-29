"""
hunt_loops.py
A lightweight script to visually find the first physical loop closure in the KITTI 00 sequence.
"""
from dataset_utils import read_ground_truth
from bundle_adjustment import select_keyframes, create_gtsam_pose3
from visualization import plot_loop_closure_hunter

POSES_PATH = '../dataset/poses/00.txt'


def main():
    print("Loading trajectory...")
    # 1. Read the ground truth poses (Simulating the Ex 6 Pose Graph)
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)

    # 2. Extract keyframes using our exact 5-meter logic
    print("Extracting keyframes...")
    keyframes = select_keyframes(poses_4x4, dist_threshold=5.0, max_frames=15)

    # 3. Build a dictionary mapping Keyframe ID -> gtsam.Pose3
    global_poses_dict = {}
    for kf in keyframes:
        T_w_c = poses_4x4[kf]
        # Use your established conversion function
        global_poses_dict[kf] = create_gtsam_pose3(T_w_c)

    print(f"Total Keyframes Extracted: {len(keyframes)}")

    # 4. Plot the first 600 frames to hunt for the intersection
    print("Plotting map to hunt for loop closures...")
    plot_loop_closure_hunter(global_poses_dict, start_idx=0, end_idx=4000, label_step=1)


if __name__ == "__main__":
    main()