
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from sklearn.cluster import KMeans
import os
import random

def get_sift_features(img_gray):
    # Initialize SIFT detector
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("SIFT_create not found (older OpenCV version or patent issue?). Using ORB.")
        sift = cv2.ORB_create()
        
    keypoints, descriptors = sift.detectAndCompute(img_gray, None)
    return keypoints, descriptors

def get_hog_features(img_gray):
    # Compute HOG descriptors
    # We want valid descriptors for blocks to cluster them.
    # pixels_per_cell=(16, 16), cells_per_block=(1, 1) gives us 1 feature vector per 16x16 cell
    features, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True, feature_vector=False)
    
    # features shape: (n_cells_y, n_cells_x, 1, 1, 9)
    # We need to reshape to (n_samples, n_features)
    n_y, n_x, _, _, n_feats = features.shape
    features_flat = features.reshape(-1, n_feats)
    
    # Create coordinates for plotting
    coords = []
    for y in range(n_y):
        for x in range(n_x):
            # Center of the 16x16 cell
            cy = y * 16 + 8
            cx = x * 16 + 8
            coords.append([cx, cy])
            
    return np.array(coords), features_flat, hog_image

def get_lbp_features(img_gray):
    # LBP returns an image of the same size. 
    # To cluster, we can take patch-wise histograms of LBP codes? 
    # Or just use the LBP value of each pixel as a 1D feature (weak)?
    # Better: Compute LBP image, then take 16x16 blocks and compute histograms of LBP codes.
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    
    # Divide into 16x16 patches
    h, w = lbp.shape
    step = 16
    coords = []
    features = []
    
    for y in range(0, h - step + 1, step):
        for x in range(0, w - step + 1, step):
            patch = lbp[y:y+step, x:x+step]
            hist, _ = np.histogram(patch.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype("float")
            if hist.sum() > 0:
                hist /= (hist.sum() + 1e-7)
            
            features.append(hist)
            coords.append([x + step//2, y + step//2])
            
    return np.array(coords), np.array(features)

def plot_clusters(img_rgb, coords, labels, k, title):
    plt.imshow(img_rgb)
    # Get colors
    cmap = plt.get_cmap('tab10')
    
    # For SIFT/Points, coords are (x, y)
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=20, alpha=0.7)
    plt.title(f"{title} (k={k})")
    plt.axis('off')

def main():
    # Load random image
    image_dir = 'imagenet_images'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found.")
        return
        
    image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, image_name)
    print(f"Processing image: {image_name}")
    
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (256, 256))
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. SIFT
    print("Extracting SIFT...")
    kp, des_sift = get_sift_features(img_gray)
    if des_sift is not None and len(des_sift) > 6:
        sift_coords = np.array([p.pt for p in kp])
        
        # Cluster k=3
        kmeans_sift_3 = KMeans(n_clusters=3, random_state=42).fit(des_sift)
        
        # Cluster k=6
        kmeans_sift_6 = KMeans(n_clusters=6, random_state=42).fit(des_sift)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plot_clusters(img_rgb, sift_coords, kmeans_sift_3.labels_, 3, "SIFT Clusters")
        plt.subplot(1, 2, 2)
        plot_clusters(img_rgb, sift_coords, kmeans_sift_6.labels_, 6, "SIFT Clusters")
        plt.suptitle("SIFT Clustering")
        plt.savefig('sift_clusters.png')
        print("Saved sift_clusters.png")

    # 2. HoG
    print("Extracting HoG...")
    hog_coords, des_hog, _ = get_hog_features(img_gray)
    if len(des_hog) > 6:
        kmeans_hog_3 = KMeans(n_clusters=3, random_state=42).fit(des_hog)
        kmeans_hog_6 = KMeans(n_clusters=6, random_state=42).fit(des_hog)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plot_clusters(img_rgb, hog_coords, kmeans_hog_3.labels_, 3, "HoG Clusters")
        plt.subplot(1, 2, 2)
        plot_clusters(img_rgb, hog_coords, kmeans_hog_6.labels_, 6, "HoG Clusters")
        plt.suptitle("HoG Clustering")
        plt.savefig('hog_clusters.png')
        print("Saved hog_clusters.png")

    # 3. LBP
    print("Extracting LBP...")
    lbp_coords, des_lbp = get_lbp_features(img_gray)
    if len(des_lbp) > 6:
        kmeans_lbp_3 = KMeans(n_clusters=3, random_state=42).fit(des_lbp)
        kmeans_lbp_6 = KMeans(n_clusters=6, random_state=42).fit(des_lbp)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plot_clusters(img_rgb, lbp_coords, kmeans_lbp_3.labels_, 3, "LBP Clusters")
        plt.subplot(1, 2, 2)
        plot_clusters(img_rgb, lbp_coords, kmeans_lbp_6.labels_, 6, "LBP Clusters")
        plt.suptitle("LBP Clustering")
        plt.savefig('lbp_clusters.png')
        print("Saved lbp_clusters.png")
        
    # BoW (Bag of Words) usually implies *using* the clusters (visual words) to represent image.
    # Since we just clustered SIFT/HoG/LBP, we effectively created Visual Words for this single image.
    # The prompt says "Extract ... BoW ... and do clustering". This is slightly confusing.
    # It might mean "Create BoW representation of patches and cluster those representations".
    # Which is effectively what we did for LBP (histogram of patches) and HoG (descriptor of patches).
    # We will assume the above covers the intent.
    
    print("Done.")

if __name__ == "__main__":
    main()
