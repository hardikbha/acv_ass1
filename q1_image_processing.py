
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def main():
    # 1. Load an RGB image of any image format
    image_dir = 'imagenet_images'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in imagenet_images directory.")
        return

    # Select a random image
    image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, image_name)
    print(f"Processing image: {image_name}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to load image: {image_path}")
        return

    # Manual BGR to RGB conversion (swap channels using NumPy)
    img_rgb = img_bgr[:, :, ::-1].copy()  # Reverse channel order: BGR -> RGB

    # Resize using OpenCV and convert to gray image with a resolution of 256x256
    img_resized_rgb = cv2.resize(img_rgb, (256, 256))
    
    # Manual grayscale conversion using weighted formula: Y = 0.299*R + 0.587*G + 0.114*B
    R = img_resized_rgb[:, :, 0].astype(np.float32)
    G = img_resized_rgb[:, :, 1].astype(np.float32)
    B = img_resized_rgb[:, :, 2].astype(np.float32)
    img_resized_gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    # 2. Display both RGB and Gray image side-by-side using matplotlib
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_resized_rgb)
    plt.title('Resized RGB (256x256)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_resized_gray, cmap='gray')
    plt.title('Resized Gray (256x256)')
    plt.axis('off')
    plt.suptitle("Task 1 & 2: RGB and Gray Side-by-Side")
    plt.show() # In a script, this might block, but required by assignment.

    # 3. Save the Gray image as <same-name>.JPG
    output_name = os.path.splitext(image_name)[0] + '.JPG'
    # cv2.imwrite expects BGR or Gray.
    cv2.imwrite(output_name, img_resized_gray)
    print(f"Saved gray image to {output_name}")

    # 4. Flip the RGB Image horizontally and vertically and display
    # Manual flip using NumPy slicing
    img_flip_h = img_resized_rgb[:, ::-1, :]  # Horizontal flip: reverse columns
    img_flip_v = img_resized_rgb[::-1, :, :]  # Vertical flip: reverse rows
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_resized_rgb)
    plt.title('Original RGB')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img_flip_h)
    plt.title('Flipped Horizontally')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img_flip_v)
    plt.title('Flipped Vertically')
    plt.axis('off')
    plt.suptitle("Task 4: Horizontal and Vertical Flip")
    plt.show()

    # 5. Perform random crops of 128x128 and rescale it to 256x256.
    h, w, _ = img_resized_rgb.shape
    crop_h, crop_w = 128, 128
    
    # Random top-left corner
    max_x = w - crop_w
    max_y = h - crop_h
    
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)
    
    # Crop
    crop_img = img_resized_rgb[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Rescale to 256x256
    rescaled_crop = cv2.resize(crop_img, (256, 256))

    # Display center point and rectangle on RGB image
    # We need to draw on a copy to avoid modifying the original used elsewhere
    img_with_rect = img_resized_rgb.copy()
    
    # Center of the crop
    center_x = start_x + crop_w // 2
    center_y = start_y + crop_h // 2
    
    # Draw Rectangle (use Red color)
    cv2.rectangle(img_with_rect, (start_x, start_y), (start_x+crop_w, start_y+crop_h), (255, 0, 0), 2)
    
    # Draw Center point (use Green)
    cv2.circle(img_with_rect, (center_x, center_y), 3, (0, 255, 0), -1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_rect)
    plt.title('RGB with Crop Rect (128x128)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rescaled_crop)
    plt.title('Cropped & Rescaled (256x256)')
    plt.axis('off')
    plt.suptitle("Task 5: Random Crop & Rescale")
    plt.show()

if __name__ == "__main__":
    main()
