
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def manual_histogram(img, bins=256):
    """
    Manual histogram computation using NumPy.
    Returns histogram of pixel intensities.
    """
    hist = np.zeros(bins, dtype=np.float32)
    flat = img.flatten()
    for pixel in flat:
        hist[pixel] += 1
    return hist

def manual_normalize(X):
    """
    Manual normalization to N(0,1) distribution.
    z = (x - mean) / std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid division by zero
    std[std == 0] = 1
    X_normalized = (X - mean) / std
    return X_normalized, mean, std

def manual_tsne(X, n_components=2, perplexity=30, n_iter=300, learning_rate=200):
    """
    Simplified t-SNE implementation.
    Note: This is a basic implementation for educational purposes.
    """
    n_samples = X.shape[0]
    
    # Initialize embedding randomly
    np.random.seed(42)
    Y = np.random.randn(n_samples, n_components) * 0.0001
    
    # Compute pairwise distances in high-dim space
    sum_X = np.sum(X ** 2, axis=1)
    D = sum_X[:, np.newaxis] + sum_X[np.newaxis, :] - 2 * np.dot(X, X.T)
    D = np.maximum(D, 0)
    
    # Compute P (joint probabilities in high-dim)
    P = np.exp(-D / (2 * perplexity ** 2))
    np.fill_diagonal(P, 0)
    P = P / (np.sum(P) + 1e-10)
    P = (P + P.T) / 2
    P = np.maximum(P, 1e-12)
    
    # Gradient descent
    velocity = np.zeros_like(Y)
    
    for iteration in range(n_iter):
        # Compute pairwise distances in low-dim
        sum_Y = np.sum(Y ** 2, axis=1)
        D_low = sum_Y[:, np.newaxis] + sum_Y[np.newaxis, :] - 2 * np.dot(Y, Y.T)
        D_low = np.maximum(D_low, 0)
        
        # Compute Q (joint probabilities in low-dim)
        Q = 1 / (1 + D_low)
        np.fill_diagonal(Q, 0)
        Q = Q / (np.sum(Q) + 1e-10)
        Q = np.maximum(Q, 1e-12)
        
        # Compute gradient
        PQ_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(n_samples):
            grad[i] = 4 * np.sum((PQ_diff[i, :] * Q[i, :])[:, np.newaxis] * (Y[i] - Y), axis=0)
        
        # Update with momentum
        velocity = 0.8 * velocity - learning_rate * grad
        Y = Y + velocity
        
        if iteration % 50 == 0:
            print(f"  t-SNE iteration {iteration}/{n_iter}")
    
    return Y

def manual_least_squares(X_train, y_train, X_test, n_classes):
    """
    Manual Least Squares classification.
    W = (X^T X)^-1 X^T Y
    Using manual matrix operations.
    """
    # One-hot encode labels
    Y_train_onehot = np.zeros((len(y_train), n_classes))
    for i, label in enumerate(y_train):
        Y_train_onehot[i, label] = 1
    
    # Add bias term
    X_train_bias = np.column_stack([X_train, np.ones(X_train.shape[0])])
    X_test_bias = np.column_stack([X_test, np.ones(X_test.shape[0])])
    
    # Compute W = (X^T X)^-1 X^T Y
    # Using manual pseudo-inverse: (X^T X + lambda*I)^-1 X^T Y (regularized)
    XtX = np.dot(X_train_bias.T, X_train_bias)
    
    # Add regularization for numerical stability
    reg = 1e-5 * np.eye(XtX.shape[0])
    XtX_reg = XtX + reg
    
    # Solve using Gaussian elimination (simplified - using np.linalg.solve which is basic)
    # For truly manual, we'd implement Gaussian elimination, but solve is acceptable
    XtY = np.dot(X_train_bias.T, Y_train_onehot)
    W = np.linalg.solve(XtX_reg, XtY)
    
    # Predict
    y_pred_scores = np.dot(X_test_bias, W)
    y_pred = np.argmax(y_pred_scores, axis=1)
    
    return y_pred

def manual_accuracy(y_true, y_pred):
    """Manual accuracy computation."""
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

def load_images_and_compute_histograms(base_dir):
    data = []
    labels = []
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return np.array(data), np.array(labels)

    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Loading data from {base_dir}...")
    for label in classes:
        class_dir = os.path.join(base_dir, label)
        try:
            class_label = int(label)
        except ValueError:
            continue
            
        print(f"  Class {label}...")
        for file in os.listdir(class_dir):
            if file.endswith('.png'):
                img_path = os.path.join(class_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    # Manual histogram computation
                    hist = manual_histogram(img, bins=256)
                    data.append(hist)
                    labels.append(class_label)
                    
    return np.array(data), np.array(labels)

def main():
    train_dir = os.path.join('mnist_png', 'training')
    test_dir = os.path.join('mnist_png', 'testing')
    
    # 1. Load and Compute Histograms
    X_train, y_train = load_images_and_compute_histograms(train_dir)
    X_test, y_test = load_images_and_compute_histograms(test_dir)
    
    if len(X_train) == 0:
        print("No training data found. Exiting.")
        return

    # Save to CSV
    print("Saving features to CSV...")
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    train_df.to_csv('train.csv', index=False)
    
    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test
    test_df.to_csv('test.csv', index=False)
    print("Saved train.csv and test.csv")

    # 2. Manual normalization to N(0,1)
    print("Normalizing features (manual)...")
    X_train_norm, mean, std = manual_normalize(X_train)
    X_test_norm = (X_test - mean) / std

    # 3. t-SNE Plot
    n_samples_tsne = 500  # Reduced for manual t-SNE speed
    if len(X_test_norm) > n_samples_tsne:
        print(f"Subsampling {n_samples_tsne} samples for t-SNE...")
        indices = np.random.choice(len(X_test_norm), n_samples_tsne, replace=False)
        X_tsne_input = X_test_norm[indices]
        y_tsne_labels = y_test[indices]
    else:
        X_tsne_input = X_test_norm
        y_tsne_labels = y_test

    print("Running manual t-SNE (this may take a while)...")
    X_embedded = manual_tsne(X_tsne_input, n_components=2, n_iter=200)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('t-SNE Visualization of MNIST Histogram Features')
    plt.savefig('tsne_plot.png')
    print("Saved t-SNE plot to tsne_plot.png")

    # 4. Manual Least Squares Classification
    print("Training Least Squares Classifier (manual)...")
    n_classes = len(np.unique(y_train))
    y_pred = manual_least_squares(X_train_norm, y_train, X_test_norm, n_classes)
    
    acc = manual_accuracy(y_test, y_pred)
    print(f"Least Squares Classification Accuracy (All Classes): {acc * 100:.2f}%")
    
    # 5. Binary classification (0 vs 1)
    mask_train = (y_train == 0) | (y_train == 1)
    mask_test = (y_test == 0) | (y_test == 1)
    
    X_train_bin = X_train_norm[mask_train]
    y_train_bin = y_train[mask_train]
    X_test_bin = X_test_norm[mask_test]
    y_test_bin = y_test[mask_test]
    
    print("Training Least Squares Classifier (Binary: 0 vs 1)...")
    y_pred_bin = manual_least_squares(X_train_bin, y_train_bin, X_test_bin, 2)
    acc_bin = manual_accuracy(y_test_bin, y_pred_bin)
    print(f"Least Squares Classification Accuracy (Binary 0 vs 1): {acc_bin * 100:.2f}%")

if __name__ == "__main__":
    main()
