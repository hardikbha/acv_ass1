
import cv2
import argparse
import os

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
        
        # Check if this frame should be processed (1-based index or 0-based index? "every kth frame" usually means k, 2k, 3k...)
        # Let's say frame 0 is the 1st frame. So if k=10, we want 9, 19, 29... or 0, 10, 20...
        # Let's use 0-indexed count divisible by k.
        if frame_count % k == 0:
            # Rescale to 256x256
            resized_frame = cv2.resize(frame, (256, 256))
            
            # Save frame
            output_filename = os.path.join(output_dir, f"{frame_count}.JPG")
            cv2.imwrite(output_filename, resized_frame)
            saved_count += 1
            
        frame_count += 1

    cap.release()
    print(f"Finished processing. Extracted {saved_count} frames to {output_dir}/")

if __name__ == "__main__":
    main()
