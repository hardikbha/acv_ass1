# ACV Assignment 1 - Warm-up

**Roll Number:** D25092  
**Email:** d25092@students.iitmandi.ac.in

Python scripts for CS673 Advanced Computer Vision Assignment 1.


## Files

| File | Description |
|------|-------------|
| `q1_image_processing.py` | Image loading, grayscale conversion, flipping, cropping |
| `q2_video_processing.py` | Video frame extraction and resizing |
| `q3_mnist_analysis.py` | MNIST histogram features, t-SNE, Least Squares classifier |
| `q4_feature_clustering.py` | LBP, HoG, SIFT feature extraction and K-Means clustering |
| `SOP.md` | Standard Operating Procedure for running scripts |

## Requirements

```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image pandas
```

## Usage

See `SOP.md` for detailed instructions.

```bash
python3 q1_image_processing.py
python3 q2_video_processing.py --k 10
python3 q3_mnist_analysis.py
python3 q4_feature_clustering.py
```

## Data (not included in repo)
- `imagenet_images/` - Sample ImageNet images
- `mnist_png/` - MNIST dataset in PNG format
- `v_BrushingTeeth_g01_c01.avi` - UCF101 video sample
