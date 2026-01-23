
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def draw_rectangle_manual(img, top_left, bottom_right, color, thickness=2):
    """
    Draw a rectangle manually using NumPy slicing.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w, c = img.shape
    
    # Clip coordinates
    x1 = max(0, min(x1, w-1))
    y1 = max(0, min(y1, h-1))
    x2 = max(0, min(x2, w-1))
    y2 = max(0, min(y2, h-1))
    
    # Top line
    img[y1:y1+thickness, x1:x2] = color
    # Bottom line
    img[y2-thickness:y2, x1:x2] = color
    # Left line
    img[y1:y2, x1:x1+thickness] = color
    # Right line
    img[y1:y2, x2-thickness:x2] = color
    
    return img

def draw_circle_manual(img, center, radius, color):
    """
    Draw a filled circle manually.
    """
    cx, cy = center
    h, w, c = img.shape
    
    # Bounding box of circle
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius + 1)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius + 1)
    
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            if (x - cx)**2 + (y - cy)**2 <= radius**2:
                img[y, x] = color
    return img

def main():
    # 1. Load an RGB image of any image format
    image_dir = 'imagenet_images'
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not image_files:
        print("No images found in imagenet_images directory.")
        return

    image_name = random.choice(image_files)
    image_path = os.path.join(image_dir, image_name)
    print(f"Processing image: {image_name}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Failed to load image: {image_path}")
        return

    # Manual BGR to RGB conversion
    img_rgb = img_bgr[:, :, ::-1].copy()

    # Resize using OpenCV (Explicitly instructed by Assignment Q1)
    img_resized_rgb = cv2.resize(img_rgb, (256, 256))
    
    # Manual grayscale conversion
    R = img_resized_rgb[:, :, 0].astype(np.float32)
    G = img_resized_rgb[:, :, 1].astype(np.float32)
    B = img_resized_rgb[:, :, 2].astype(np.float32)
    img_resized_gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    # 2. Display
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
    plt.show()

    # 3. Save Gray image
    output_name = os.path.splitext(image_name)[0] + '.JPG'
    cv2.imwrite(output_name, img_resized_gray)
    print(f"Saved gray image to {output_name}")

    # 4. Manual Flip
    img_flip_h = img_resized_rgb[:, ::-1, :]
    img_flip_v = img_resized_rgb[::-1, :, :]
    
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

    # 5. Random Crops
    h, w, _ = img_resized_rgb.shape
    crop_h, crop_w = 128, 128
    
    max_x = w - crop_w
    max_y = h - crop_h
    
    start_x = random.randint(0, max_x)
    start_y = random.randint(0, max_y)
    
    crop_img = img_resized_rgb[start_y:start_y+crop_h, start_x:start_x+crop_w]
    
    # Rescale to 256x256 (Using cv2.resize as per Q1 instruction context or just consistent resize)
    # The prompt says "Perform random crops ... and rescale it". Doesn't specify method.
    # To be safe and consistent with Q1 "using OpenCV" resize, we can use cv2.resize.
    rescaled_crop = cv2.resize(crop_img, (256, 256))

    # Display center point and rectangle manually
    img_with_rect = img_resized_rgb.copy()
    
    center_x = start_x + crop_w // 2
    center_y = start_y + crop_h // 2
    
    # Manual Rectangle (Red)
    img_with_rect = draw_rectangle_manual(img_with_rect, (start_x, start_y), 
                                        (start_x+crop_w, start_y+crop_h), (255, 0, 0), 2)
    
    # Manual Center point (Green) - radius 3
    img_with_rect = draw_circle_manual(img_with_rect, (center_x, center_y), 3, (0, 255, 0))

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_with_rect)
    plt.title('RGB with Crop Rect')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rescaled_crop)
    plt.title('Cropped & Rescaled')
    plt.axis('off')
    plt.suptitle("Task 5: Random Crop & Rescale")
    plt.show()

if __name__ == "__main__":
    main()
