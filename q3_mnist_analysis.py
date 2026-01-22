
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def load_images_and_compute_histograms(base_dir):
    data = []
    labels = []
    
    # Check if directory exists
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return np.array(data), np.array(labels)

    classes = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    print(f"Loading data from {base_dir}...")
    for label in classes:
        # Assuming folder names are class labels (0-9)
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
                    # Compute histogram
                    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
                    hist = hist.flatten() # 256 features
                    
                    data.append(hist)
                    labels.append(class_label)
                    
    return np.array(data), np.array(labels)

def least_squares_classifier(X_train, y_train, X_test):
    # One-hot encode labels
    n_classes = len(np.unique(y_train))
    Y_train_onehot = np.eye(n_classes)[y_train]
    
    # Add bias term to X
    X_train_bias = np.c_[X_train, np.ones(X_train.shape[0])]
    X_test_bias = np.c_[X_test, np.ones(X_test.shape[0])]
    
    # Weights W = (X^T X)^-1 X^T Y
    # Using pseudo-inverse for stability: W = pinv(X) * Y
    print("Computing Least Squares weights...")
    W = np.linalg.pinv(X_train_bias) @ Y_train_onehot
    
    # Predict
    y_pred_onehot = X_test_bias @ W
    y_pred = np.argmax(y_pred_onehot, axis=1)
    
    return y_pred

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
    print("Saving headers to CSV...")
    # Create DataFrame for easier saving
    train_df = pd.DataFrame(X_train)
    train_df['label'] = y_train
    train_df.to_csv('train.csv', index=False)
    
    test_df = pd.DataFrame(X_test)
    test_df['label'] = y_test
    test_df.to_csv('test.csv', index=False)
    print("Saved train.csv and test.csv")

    # 2. Normalize features to N(0,1)
    print("Normalizing features...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # 3. t-SNE Plot
    # Determine how many points to use for t-SNE to avoid slowness
    n_samples_tsne = 1000
    if len(X_test_norm) > n_samples_tsne:
        print(f"Subsampling {n_samples_tsne} samples for t-SNE...")
        indices = np.random.choice(len(X_test_norm), n_samples_tsne, replace=False)
        X_tsne_input = X_test_norm[indices]
        y_tsne_labels = y_test[indices]
    else:
        X_tsne_input = X_test_norm
        y_tsne_labels = y_test

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_embedded = tsne.fit_transform(X_tsne_input)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne_labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter, label='Digit Class')
    plt.title('t-SNE Visualization of MNIST Histogram Features')
    plt.savefig('tsne_plot.png')
    print("Saved t-SNE plot to tsne_plot.png")
    # plt.show() # Commented out to run non-interactively if needed, but SOP says display. 
    # Since I'm running via command, I might as well save it.

    # 4. Least Square Method
    print("Training Least Squares Classifier...")
    y_pred = least_squares_classifier(X_train_norm, y_train, X_test_norm)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"Least Squares Classification Accuracy (All Classes): {acc * 100:.2f}%")
    
    # 5. Evaluate on two classes (e.g., 0 and 1)
    mask_train = (y_train == 0) | (y_train == 1)
    mask_test = (y_test == 0) | (y_test == 1)
    
    X_train_bin = X_train_norm[mask_train]
    y_train_bin = y_train[mask_train]
    X_test_bin = X_test_norm[mask_test]
    y_test_bin = y_test[mask_test]
    
    print("Training Least Squares Classifier (Binary: 0 vs 1)...")
    y_pred_bin = least_squares_classifier(X_train_bin, y_train_bin, X_test_bin)
    acc_bin = accuracy_score(y_test_bin, y_pred_bin)
    print(f"Least Squares Classification Accuracy (Binary 0 vs 1): {acc_bin * 100:.2f}%")

if __name__ == "__main__":
    main()
