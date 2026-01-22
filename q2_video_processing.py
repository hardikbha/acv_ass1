
import cv2
import argparse
import os
import numpy as np

def manual_resize(img, new_width, new_height):
    """
    Manual bilinear interpolation resize using NumPy.
    """
    old_height, old_width = img.shape[:2]
    
    # Handle both grayscale and color images
    if len(img.shape) == 3:
        channels = img.shape[2]
        resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)
    else:
        resized = np.zeros((new_height, new_width), dtype=np.uint8)
    
    # Scale factors
    x_ratio = old_width / new_width
    y_ratio = old_height / new_height
    
    for i in range(new_height):
        for j in range(new_width):
            # Find corresponding position in original image
            x = j * x_ratio
            y = i * y_ratio
            
            # Get the four nearest pixels
            x0 = int(x)
            y0 = int(y)
            x1 = min(x0 + 1, old_width - 1)
            y1 = min(y0 + 1, old_height - 1)
            
            # Bilinear interpolation weights
            x_diff = x - x0
            y_diff = y - y0
            
            if len(img.shape) == 3:
                for c in range(channels):
                    # Bilinear interpolation formula
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

def main():
    parser = argparse.ArgumentParser(description='Video Processing Task')
    parser.add_argument('--k', type=int, default=10, help='Extract every kth frame')
    parser.add_argument('--video_path', type=str, default='v_BrushingTeeth_g01_c01.avi', help='Path to video file')
    args = parser.parse_args()
    
    video_path = args.video_path
    k = args.k
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"frames_{video_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    saved_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"Extracting every {k}th frame...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame should be processed
        if frame_count % k == 0:
            # Manual resize to 256x256
            resized_frame = manual_resize(frame, 256, 256)
            
            # Save frame
            output_filename = os.path.join(output_dir, f"{frame_count}.JPG")
            cv2.imwrite(output_filename, resized_frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Finished processing. Extracted {saved_count} frames to {output_dir}/")

if __name__ == "__main__":
    main()
