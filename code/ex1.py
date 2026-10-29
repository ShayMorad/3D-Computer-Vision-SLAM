"""
ex1.py
Main execution file for VAN Exercise 1.
"""
import cv2
import matplotlib.pyplot as plt
import random
from dataset_utils import read_stereo_pair
from feature_tracking import extract_features, match_features, match_features_ratio_test
from visualization import format_plot

DATA_PATH = '../dataset/sequences/00/'
CURRENT_ALGO = "SIFT"

def q1_1(img1, img2):
    """Answers Q 1.1 and 1.2: Extraction and Descriptors."""
    print("--- Q 1.1 & 1.2 ---")
    kp1, des1 = extract_features(img1, algorithm=CURRENT_ALGO)
    kp2, des2 = extract_features(img2, algorithm=CURRENT_ALGO)

    print(f"Key-points Left: {len(kp1)}, Right: {len(kp2)}")
    print(f"\nDescriptor of the first feature (Left Image):")
    print(des1[0])
    print(f"\nDescriptor of the first feature (Right Image):")
    print(des2[0])

    img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
    img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(img1_kp)
    format_plot(ax1, 'Left Image Key-points', 'X [pixels]', 'Y [pixels]', equal_axes=True)
    ax2.imshow(img2_kp)
    format_plot(ax2, 'Right Image Key-points', 'X [pixels]', 'Y [pixels]', equal_axes=True)
    plt.show()

    return kp1, des1, kp2, des2


def q1_3(img1, img2, kp1, des1, kp2, des2):
    """Answers Q 1.3: Descriptor Matching."""
    matches = match_features(des1, des2, algorithm=CURRENT_ALGO)
    random_matches = random.sample(matches, 20)

    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, random_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(img_matches)
    format_plot(ax, 'Figure 2: 20 Random Matches (Pre-Significance Test)', 'X [pixels]', 'Y [pixels]', equal_axes=True)
    plt.show()


def q1_4(img1, img2, kp1, des1, kp2, des2):
    """Answers Q 1.4: Significance Test."""
    print("--- Q 1.4 ---")
    ratio = 0.75
    good, rejected = match_features_ratio_test(des1, des2, algorithm=CURRENT_ALGO, ratio=ratio)

    print(f"Ratio used: {ratio}")
    print(f"Matches discarded: {len(rejected)}\n")

    random_good = random.sample(good, 20)
    img_good = cv2.drawMatches(img1, kp1, img2, kp2, random_good, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.imshow(img_good)
    format_plot(ax, f'Figure 3: 20 Good Matches (Ratio={ratio})', 'X [pixels]', 'Y [pixels]', equal_axes=True)
    plt.show()

    # Find a failed match by strengthening the test
    _, strict_rejected = match_features_ratio_test(des1, des2, algorithm=CURRENT_ALGO, ratio=0.4)
    random_strict = random.sample(strict_rejected, 5)
    img_rejected = cv2.drawMatches(img1, kp1, img2, kp2, random_strict, None, matchColor=(255, 0, 0),
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.imshow(img_rejected)
    format_plot(ax2, 'Figure 4: Rejected Matches (Strict Ratio=0.4)', 'X [pixels]', 'Y [pixels]', equal_axes=True)
    plt.show()


def main():
    img1, img2 = read_stereo_pair(DATA_PATH, 0)
    kp1, des1, kp2, des2 = q1_1(img1, img2) # 1.1 and 1.2
    q1_3(img1, img2, kp1, des1, kp2, des2) # 1.3
    q1_4(img1, img2, kp1, des1, kp2, des2) # 1.4


if __name__ == "__main__":
    main()