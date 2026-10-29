"""
ex4.py
Main execution file for VAN Exercise 4: Long-Term Feature Tracking
"""

import time
import random

from bundle_adjustment import calculate_backward_reprojection
from tracking_database import TrackingDB
from feature_tracking import match_features, process_stereo_frame, prepare_4way_data, generate_database_data, \
    extract_safe_database_links
from geometry import run_custom_ransac
from dataset_utils import read_ground_truth, read_stereo_pair, read_cameras
from visualization import *

DATA_PATH = '../dataset/sequences/00/'
POSES_PATH = '../dataset/poses/00.txt'
CURRENT_ALGO = "AKAZE"
DB_NAME = "kitti_00_database"
NUM_FRAMES = 3300
DB_EXIST = True


def q4_1():
    """Answers Q 4.1: Database Engineering and Sequence Serialization."""
    print("--- Q 4.1: Building Tracking Database ---")
    start_time = time.time()

    # Initialize Cameras and Database
    K, M1, M2 = read_cameras(DATA_PATH)
    P1 = K @ M1
    P2 = K @ M2
    db = TrackingDB()

    # Pre-process Frame 0
    kp0_l, des0_l, kp0_r, des0_r, inliers_stereo0, pts3D_0 = process_stereo_frame(DATA_PATH, 0, CURRENT_ALGO, P1, P2)

    # Initialize DB with Frame 0
    features0, links0 = extract_safe_database_links(des0_l, kp0_l, kp0_r, inliers_stereo0)
    db.add_frame(links0, features0)

    print(f"Initialized Database with Frame 0: {len(links0)} features.")

    # Sequential Tracking Loop
    for idx in range(1, NUM_FRAMES):
        if idx % 100 == 0:
            print(f"Processing Frame {idx}/{NUM_FRAMES}...")

        # 1. Process Current Frame (Stereo)
        kp1_l, des1_l, kp1_r, des1_r, inliers_stereo1, pts3D_1 = process_stereo_frame(DATA_PATH, idx, CURRENT_ALGO, P1,
                                                                                      P2)

        # 2. Temporal Matching (Left 0 -> Left 1)
        matches_t = match_features(des0_l, des1_l, algorithm=CURRENT_ALGO)

        # 3. 4-Way Sync & RANSAC Consensus
        synced_3D_0, synced_2D_1, track_indices = prepare_4way_data(inliers_stereo0, inliers_stereo1, matches_t, kp0_l,
                                                                    kp1_l, pts3D_0)

        if len(synced_3D_0) >= 4:
            R, tvec, best_mask = run_custom_ransac(synced_3D_0, track_indices, kp0_l, kp0_r, kp1_l, kp1_r, K, P1, P2,
                                                   M2, should_print=False)

            # 4. Generate the DB-Compatible Matches and Mask using our new Adapter
            db_matches, db_inliers = generate_database_data(
                track_indices, best_mask,
                inliers_stereo0, kp0_l,
                inliers_stereo1, kp1_l
            )
        else:
            # Fallback if the sequence completely drops
            num_filtered_0 = len(extract_safe_database_links(des0_l, kp0_l, kp0_r, inliers_stereo0)[0])
            db_matches = [cv2.DMatch()] * num_filtered_0
            db_inliers = [False] * num_filtered_0

        # 5. Push to Database
        features1, links1 = extract_safe_database_links(des1_l, kp1_l, kp1_r, inliers_stereo1)

        # We now pass db_matches instead of the raw matches_t
        db.add_frame(links1, features1, db_matches, db_inliers)

        # Update references for next iteration
        kp0_l, des0_l, kp0_r = kp1_l, des1_l, kp1_r
        inliers_stereo0 = inliers_stereo1
        pts3D_0 = pts3D_1

    print(f"Tracking Complete. Total Execution Time: {(time.time() - start_time):.2f} seconds.")

    # Serialize the long-term memory
    db.serialize(DB_NAME)
    return db


def q4_2(db):
    """Answers Q 4.2: Present tracking statistics."""
    print("--- Q 4.2: Tracking Statistics ---")

    # 1. Isolate valid tracks (length > 1) as explicitly requested
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) > 1]

    # Convert to set for O(1) lookups during the frame links calculation
    valid_tracks_set = set(valid_tracks)

    # 2. Extract track lengths
    track_lengths = [len(db.frames(tId)) for tId in valid_tracks]

    # 3. Calculate frame links (valid tracks per frame)
    tracks_per_frame = []
    for fId in db.all_frames():
        frame_tracks = db.tracks(fId)
        # Count only tracks on this frame that have a total length > 1
        valid_count = sum(1 for tId in frame_tracks if tId in valid_tracks_set)
        tracks_per_frame.append(valid_count)

    # 4. Analytics Output
    print(f"Total number of tracks (length > 1): {len(valid_tracks)}")
    print(f"Number of frames: {db.frame_num()}")
    print(f"Mean track length: {np.mean(track_lengths):.2f} frames")
    print(f"Maximum track length: {np.max(track_lengths)} frames")
    print(f"Minimum track length: {np.min(track_lengths)} frames")
    print(f"Mean number of frame links (tracks per image): {np.mean(tracks_per_frame):.2f}\n")


def q4_3(db):
    """Answers Q 4.3: 20x20 feature crop visualization."""
    print("--- Q 4.3: Plotting 20x20 Feature Track ---")
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) >= 6]
    target_track = random.choice(valid_tracks)

    if target_track is not None:
        plot_track_feature_crops(db.frames(target_track), target_track, db, DATA_PATH, read_stereo_pair)
    else:
        print("No tracks of length >= 6 found.")


def q4_4(db):
    """Answers Q 4.4: Connectivity graph."""
    print("--- Q 4.4: Plotting Connectivity Graph ---")
    connectivity = []

    for fId in range(db.frame_num() - 1):
        tracks_current = set(db.tracks(fId))
        tracks_next = set(db.tracks(fId + 1))
        connectivity.append(len(tracks_current.intersection(tracks_next)))

    frames_x = list(range(len(connectivity)))
    plot_sequence_stat(frames_x, connectivity, 'Connectivity', 'frame', 'outgoing tracks', mean_line=True)


def q4_5(db):
    """Answers Q 4.5: Percentage of inliers per frame."""
    print("--- Q 4.5: Plotting Inlier Percentage ---")
    percentages = []

    for fId in db.all_frames():
        total = db.features(fId).shape[0]
        if total == 0:
            percentages.append(0)
            continue

        inliers = sum(1 for tId in db.frameId_to_trackIds_list[fId] if tId != -1)
        percentages.append((inliers / total) * 100)

    frames_x = list(range(len(percentages)))
    plot_sequence_stat(frames_x, percentages, 'Inlier Percentage per Frame', 'frame', 'Inliers (%)')


def q4_6(db):
    """Answers Q 4.6: Track length histogram."""
    print("--- Q 4.6: Plotting Track Length Histogram ---")
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) > 1]
    track_lengths = [len(db.frames(tId)) for tId in valid_tracks]

    plot_histogram(track_lengths, 50, 'Track length histogram', 'Track length', 'Track #', log_scale=True)


def q4_7(db):
    """Answers Q 4.7: Ground Truth Reprojection Analysis."""
    print("--- Q 4.7: Ground Truth Reprojection Analysis ---")

    # Setup Camera Geometry
    K, M1, M2 = read_cameras(DATA_PATH)
    P1_local = K @ M1
    P2_local = K @ M2

    # Load all 4x4 transformation matrices once
    poses_4x4 = read_ground_truth(POSES_PATH, return_transforms=True)

    # Find a random track of length >= 10
    valid_tracks = [tId for tId in db.all_tracks() if len(db.frames(tId)) >= 10]
    if valid_tracks:
        target_track = random.choice(valid_tracks)
        distances, err_l, err_r = calculate_backward_reprojection(db, target_track, db.frames(target_track), poses_4x4,
                                                                  P1_local, P2_local)
        plot_reprojection_errors(distances, err_l, err_r)
    else:
        print("No tracks of length >= 10 found.")


def main():
    # Q 4.1
    db = TrackingDB()
    if not DB_EXIST:
        db = q4_1()
    else:
        db.load(DB_NAME)

    # Q 4.2
    q4_2(db)

    # Q 4.3
    q4_3(db)

    # Q 4.4
    q4_4(db)

    # Q 4.5
    q4_5(db)

    # Q 4.6
    q4_6(db)

    # Q 4.7
    q4_7(db)


if __name__ == "__main__":
    main()
