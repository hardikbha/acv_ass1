# Standard Operating Procedure (SOP) for ACV Assignment 1

## Overview
This document outlines the steps to execute the solution scripts for Assignment 1.

## Prerequisites
- Python 3.x
- Libraries: `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `scikit-image`
- Input Data:
    - `imagenet_images/` directory containing sample images.
    - `v_BrushingTeeth_g01_c01.avi` video file.
    - `mnist_png/` directory containing MNIST dataset (training/testing folders).

## Execution Steps

### Question 1: Image Processing
1.  Run the script:
    ```bash
    python q1_image_processing.py
    ```
2.  **Expected Output**:
    - Displays: Side-by-side RGB vs Gray, Original vs Flipped, Original with crop rectangle vs Cropped.
    - File Created: A grayscale JPG image in the current directory (e.g., `processed_gray.jpg`).

### Question 2: Video Processing
1.  Run the script (replace `k` with desired frame interval, default is 10):
    ```bash
    python q2_video_processing.py --k 10
    ```
2.  **Expected Output**:
    - Directory Created: `frames_BrushingTeeth/` (or similar).
    - Files Created: Resized JPG frames extracted from the video.

### Question 3: MNIST Analysis
1.  Run the script:
    ```bash
    python q3_mnist_analysis.py
    ```
2.  **Expected Output**:
    - Files Created: `train.csv` and `test.csv` containing histogram features.
    - Displays: t-SNE plot (saved as `tsne_plot.png` or displayed).
    - Console Output: Classification accuracy of the Least Square Method.

### Question 4: Feature Clustering
1.  Run the script:
    ```bash
    python q4_feature_clustering.py
    ```
2.  **Expected Output**:
    - Displays: Scatter plots of K-Means clusters (k=3 and k=6) for extracted features.
    - Warning: Feature extraction (e.g., SIFT) might take a moment.

## Trouble Shooting
- **Missing Libraries**: Install via `pip install numpy opencv-python matplotlib scikit-learn scikit-image`.
- **SIFT Error**: If SIFT is unavailable (due to patent), the script will try to use ORB or warn the user. Ensure `opencv-contrib-python` is installed if needed.
- **Data Not Found**: Ensure the script is run from the directory containing `imagenet_images`, `mnist_png`, etc.
