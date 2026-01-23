
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def manual_resize(img, new_width, new_height):
    """
    Manual bilinear interpolation resize using NumPy.
    """
    old_height, old_width = img.shape[:2]
    
    if len(img.shape) == 3:
        channels = img.shape[2]
        resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    else:
        resized = np.zeros((new_height, new_width), dtype=np.uint8)
    
    x_ratio = old_width / new_width
    y_ratio = old_height / new_height
    
    # Create meshgrid for vectorized coordinate calculation
    # (Much faster than loops, but loops are explicitly 'manual' and safer for strict assignments.
    # We will stick to the loop implementation from Q2 for consistency and simplicity in logic demonstration)
    
    for i in range(new_height):
        # Optimization: Calculate y logic once per row
        y = i * y_ratio
        y0 = int(y)
        y1 = min(y0 + 1, old_height - 1)
        y_diff = y - y0
        
        for j in range(new_width):
            x = j * x_ratio
            x0 = int(x)
            x1 = min(x0 + 1, old_width - 1)
            x_diff = x - x0
            
            if len(img.shape) == 3:
                for c in range(channels):
                    val = (img[y0, x0, c] * (1 - x_diff) * (1 - y_diff) +
                           img[y0, x1, c] * x_diff * (1 - y_diff) +
                           img[y1, x0, c] * (1 - x_diff) * y_diff +
                           img[y1, x1, c] * x_diff * y_diff)
                    resized[i, j, c] = int(val)
            else:
                val = (img[y0, x0] * (1 - x_diff) * (1 - y_diff) +
                       img[y0, x1] * x_diff * (1 - y_diff) +
                       img[y1, x0] * (1 - x_diff) * y_diff +
                       img[y1, x1] * x_diff * y_diff)
                resized[i, j] = int(val)
    
    return resized

def manual_lbp(img_gray, radius=1, n_points=8):
    height, width = img_gray.shape
    lbp_img = np.zeros((height, width), dtype=np.uint8)
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = img_gray[y, x]
            binary_code = 0
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                px = x + int(round(radius * np.cos(angle)))
                py = y - int(round(radius * np.sin(angle)))
                if img_gray[py, px] >= center:
                    binary_code |= (1 << i)
            lbp_img[y, x] = binary_code
    return lbp_img

def manual_hog(img_gray, cell_size=16, n_bins=9):
    height, width = img_gray.shape
    img = img_gray.astype(np.float32)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            gx[y, x] = img[y, x + 1] - img[y, x - 1]
            gy[y, x] = img[y + 1, x] - img[y - 1, x]
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    orientation = np.arctan2(gy, gx) * 180 / np.pi
    orientation[orientation < 0] += 180
    n_cells_y = height // cell_size
    n_cells_x = width // cell_size
    features = []
    coords = []
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            y_start = cy * cell_size
            x_start = cx * cell_size
            cell_mag = magnitude[y_start:y_start + cell_size, x_start:x_start + cell_size]
            cell_ori = orientation[y_start:y_start + cell_size, x_start:x_start + cell_size]
            hist = np.zeros(n_bins)
            bin_width = 180 / n_bins
            for i in range(cell_size):
                for j in range(cell_size):
                    bin_idx = int(cell_ori[i, j] / bin_width) % n_bins
                    hist[bin_idx] += cell_mag[i, j]
            norm = np.sqrt(np.sum(hist ** 2) + 1e-6)
            hist = hist / norm
            features.append(hist)
            coords.append([x_start + cell_size // 2, y_start + cell_size // 2])
    return np.array(coords), np.array(features)

def manual_sift_keypoints(img_gray, threshold=0.03):
    height, width = img_gray.shape
    img = img_gray.astype(np.float32) / 255.0
    sigma = 1.6
    k = np.sqrt(2)
    def gaussian_blur(img, sigma):
        size = int(6 * sigma) | 1
        kernel = np.ones((size, size)) / (size * size)
        result = np.zeros_like(img)
        pad = size // 2
        padded = np.pad(img, pad, mode='reflect')
        for y in range(height):
            for x in range(width):
                result[y, x] = np.sum(padded[y:y + size, x:x + size] * kernel)
        return result
    g1 = gaussian_blur(img, sigma)
    g2 = gaussian_blur(img, sigma * k)
    dog = g2 - g1
    keypoints = []
    descriptors = []
    for y in range(16, height - 16, 8):
        for x in range(16, width - 16, 8):
            patch = dog[y - 1:y + 2, x - 1:x + 2]
            center = dog[y, x]
            if abs(center) > threshold:
                if center == patch.max() or center == patch.min():
                    keypoints.append([x, y])
                    desc_patch = img[y - 8:y + 8, x - 8:x + 8]
                    if desc_patch.shape == (16, 16):
                        gx = np.diff(desc_patch, axis=1); gy = np.diff(desc_patch, axis=0)
                        gx = gx[:15, :]; gy = gy[:, :15]
                        mag = np.sqrt(gx ** 2 + gy ** 2)
                        desc = mag.flatten()[:128]
                        if len(desc) == 128:
                            desc = desc / (np.linalg.norm(desc) + 1e-6)
                            descriptors.append(desc)
                        else: keypoints.pop()
    return np.array(keypoints), np.array(descriptors) if descriptors else None

def manual_kmeans(X, k, max_iters=100):
    n_samples = X.shape[0]
    np.random.seed(42)
    indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[indices].copy()
    labels = np.zeros(n_samples, dtype=int)
    for iteration in range(max_iters):
        for i in range(n_samples):
            dist = np.sum((X[i] - centroids) ** 2, axis=1)
            labels[i] = np.argmin(dist)
        new_centroids = np.zeros_like(centroids)
        counts = np.zeros(k)
        for i in range(n_samples):
            new_centroids[labels[i]] += X[i]
            counts[labels[i]] += 1
        for j in range(k):
            if counts[j] > 0: new_centroids[j] /= counts[j]
        if np.allclose(centroids, new_centroids): break
        centroids = new_centroids
    return labels, centroids

def plot_clusters(img_rgb, coords, labels, k, title):
    plt.imshow(img_rgb)
    cmap = plt.get_cmap('tab10')
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=cmap, s=20, alpha=0.7)
    plt.title(f"{title} (k={k})")
    plt.axis('off')

def main():
    image_dir = 'imagenet_images'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files: return
    image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, image_name)
    print(f"Processing image: {image_name}")
    
    img_bgr = cv2.imread(image_path)
    img_rgb = img_bgr[:, :, ::-1].copy()
    
    # Manual Resize in Q4 (Since Q4 doesn't explicitly permit OpenCV resize like Q1 does)
    img_rgb = manual_resize(img_rgb, 256, 256)
    
    img_gray = (0.299 * img_rgb[:, :, 0] + 0.587 * img_rgb[:, :, 1] + 0.114 * img_rgb[:, :, 2]).astype(np.uint8)
    
    print("Extracting SIFT-like features (manual)...")
    kp_coords, des_sift = manual_sift_keypoints(img_gray)
    if des_sift is not None and len(des_sift) > 6:
        labels_3, _ = manual_kmeans(des_sift, 3)
        labels_6, _ = manual_kmeans(des_sift, 6)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plot_clusters(img_rgb, kp_coords, labels_3, 3, "SIFT")
        plt.subplot(1, 2, 2); plot_clusters(img_rgb, kp_coords, labels_6, 6, "SIFT")
        plt.suptitle("SIFT Clustering (Manual)")
        plt.savefig('sift_clusters.png')
        print("Saved sift_clusters.png")

    print("Extracting HoG features (manual)...")
    hog_coords, des_hog = manual_hog(img_gray)
    if len(des_hog) > 6:
        labels_3, _ = manual_kmeans(des_hog, 3); labels_6, _ = manual_kmeans(des_hog, 6)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plot_clusters(img_rgb, hog_coords, labels_3, 3, "HoG")
        plt.subplot(1, 2, 2); plot_clusters(img_rgb, hog_coords, labels_6, 6, "HoG")
        plt.suptitle("HoG Clustering (Manual)")
        plt.savefig('hog_clusters.png')
        print("Saved hog_clusters.png")

    print("Extracting LBP features (manual)...")
    lbp_img = manual_lbp(img_gray)
    step = 16
    lbp_coords = []; lbp_features = []
    for y in range(0, 256 - step, step):
        for x in range(0, 256 - step, step):
            patch = lbp_img[y:y + step, x:x + step]
            hist = np.zeros(256)
            for val in patch.flatten(): hist[val] += 1
            hist = hist / (np.sum(hist) + 1e-6)
            lbp_features.append(hist)
            lbp_coords.append([x + step // 2, y + step // 2])
    lbp_coords = np.array(lbp_coords)
    lbp_features = np.array(lbp_features)
    if len(lbp_features) > 6:
        labels_3, _ = manual_kmeans(lbp_features, 3); labels_6, _ = manual_kmeans(lbp_features, 6)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1); plot_clusters(img_rgb, lbp_coords, labels_3, 3, "LBP")
        plt.subplot(1, 2, 2); plot_clusters(img_rgb, lbp_coords, labels_6, 6, "LBP")
        plt.suptitle("LBP Clustering (Manual)")
        plt.savefig('lbp_clusters.png')
        print("Saved lbp_clusters.png")
    print("Done.")

if __name__ == "__main__":
    main()
