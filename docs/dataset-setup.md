# KITTI Odometry setup

The implementation expects **KITTI Odometry grayscale stereo sequence 00** and corresponding calibration and ground-truth poses. Download the necessary archives from the [official KITTI odometry page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

## Directory layout

```text
3D-Computer-Vision-SLAM/
├── code/
├── dataset/                       # not tracked by Git
│   ├── sequences/
│   │   └── 00/
│   │       ├── image_0/          # left grayscale camera
│   │       │   ├── 000000.png
│   │       │   └── ...
│   │       ├── image_1/          # right grayscale camera
│   │       │   ├── 000000.png
│   │       │   └── ...
│   │       └── calib.txt
│   └── poses/
│       └── 00.txt
└── requirements.txt
```

This layout matches the `../dataset/sequences/00/` and `../dataset/poses/00.txt` paths referenced in the existing Python scripts **when run from `code/`**.

## Intermediate artifacts

Later experiments use a serialized feature-tracking database, commonly named `kitti_00_database.pkl`. The repository omits this generated binary. Build it using the tracking/database stage (`code/ex4.py`) before running downstream optimization and analysis.

KITTI imagery, calibration files, and pose data are obtained separately under the dataset's [published terms](https://www.cvlibs.net/datasets/kitti/). They are not redistributed here.
