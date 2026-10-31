"""
ECE 491 - Python Lab 2
Basic Image Processing Using Python

Required libraries:
    numpy, opencv-python, matplotlib, scipy

Place boat.png and Barbara.jpeg in the same directory as this script.
The script follows the professor's handout, including convolve2d's default
'full' convolution mode. It saves all report figures in ./outputs.
"""

from pathlib import Path
import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)


def save_gray_display(path, arr, title, shift=0.0):
    """Display a grayscale result and save the exact displayed view."""
    display = np.clip(arr + shift, 0.0, 1.0)
    plt.figure(figsize=(6, 6))
    plt.imshow(display, cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    return display


def save_rgb_display(path, arr, title, shift=0.0):
    """Display an RGB result and save the exact displayed view."""
    display = np.clip(arr + shift, 0.0, 1.0)
    plt.figure(figsize=(6, 6))
    plt.imshow(display)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close()
    return display


def float_to_uint8(arr):
    """Convert a normalized floating image in [0,1] to uint8 safely."""
    return np.round(np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)


def filter_rgb_full(img_rgb, kernel):
    """Apply 2-D full convolution independently to R, G, and B planes."""
    planes = [convolve2d(img_rgb[:, :, c], kernel) for c in range(3)]
    return np.stack(planes, axis=2)


# ============================================================================
# Read and normalize Boat image
# ============================================================================
boat_path = BASE / "boat.png"
img_u8 = cv.imread(str(boat_path), cv.IMREAD_GRAYSCALE)
if img_u8 is None:
    raise FileNotFoundError("boat.png was not found in the script directory.")

print("Boat uint8 shape:", img_u8.shape)

plt.figure(figsize=(6, 6))
plt.imshow(img_u8, cmap="gray")
plt.title("Original Boat image")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT / "01_boat_original.png", dpi=170, bbox_inches="tight", pad_inches=0.08)
plt.close()

img = img_u8.astype(np.float32) / 255.0
print("Boat normalized dtype:", img.dtype)

# 1) Pixel at location (132,112) -- NumPy convention: row 132, column 112.
pixel_value = img[132, 112]
print("Q1 pixel img[132,112] =", float(pixel_value))

# 2) Top-left quarter. Boat is 512x512, so a=0 and b=256.
rows, cols = img.shape
quarter = img[0:rows//2, 0:cols//2]
print("Q2 top-left quarter shape:", quarter.shape)
save_gray_display(OUT / "02_boat_top_left_quarter.png", quarter,
                  "Boat - top-left quarter")

# 3) 3x3 moving-average box filter.
ma = (1.0/9.0) * np.array([
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
])
img_filtered = convolve2d(img, ma)  # default mode='full'
print("Q3 moving-average output shape:", img_filtered.shape)
save_gray_display(OUT / "03_boat_moving_average.png", img_filtered,
                  "Boat - 3x3 moving-average filter")

# 4) Save the moving-average image to disk.
cv.imwrite(str(OUT / "04_boat_moving_average_saved.png"),
           float_to_uint8(img_filtered))
print("Q4 saved:", OUT / "04_boat_moving_average_saved.png")

# 5) 3x3 difference filter exactly as stated in the handout.
difference_filter = np.array([
    [1.0, 0.0, -1.0],
    [2.0, 0.0, -2.0],
    [1.0, 0.0, -1.0]
])
print("Q5 difference filter:\n", difference_filter)
img_difference = convolve2d(img, difference_filter)
print("Q5 difference output shape:", img_difference.shape)
save_gray_display(OUT / "05_boat_difference_filter.png", img_difference,
                  "Boat - 3x3 difference filter (+0.5 for display)", shift=0.5)

# 6) Horizontal-edge task, following the professor's stated row filter.
horizontal_filter = np.array([[-1.0, 2.0, -1.0]])
boat_horizontal = convolve2d(img, horizontal_filter)
print("Q6 horizontal-edge output shape:", boat_horizontal.shape)
save_gray_display(OUT / "06_boat_horizontal_edges.png", boat_horizontal,
                  "Boat - horizontal-edge experiment [-1 2 -1] (+0.5)", shift=0.5)

# 7) Vertical-edge task, using the transpose exactly as requested.
vertical_filter = horizontal_filter.T
boat_vertical = convolve2d(img, vertical_filter)
print("Q7 vertical-edge output shape:", boat_vertical.shape)
save_gray_display(OUT / "07_boat_vertical_edges.png", boat_vertical,
                  "Boat - vertical-edge experiment [-1 2 -1]^T (+0.5)", shift=0.5)


# ============================================================================
# 8) Repeat the filtering experiments with color Barbara image
# ============================================================================
barbara_path = BASE / "Barbara.jpeg"
barbara_bgr_u8 = cv.imread(str(barbara_path), cv.IMREAD_COLOR)
if barbara_bgr_u8 is None:
    raise FileNotFoundError("Barbara.jpeg was not found in the script directory.")

# OpenCV is BGR; convert to RGB before displaying/processing.
barbara_rgb_u8 = cv.cvtColor(barbara_bgr_u8, cv.COLOR_BGR2RGB)
barbara = barbara_rgb_u8.astype(np.float32) / 255.0
print("Barbara RGB shape:", barbara.shape)

plt.figure(figsize=(6, 6))
plt.imshow(barbara)
plt.title("Original Barbara image - RGB")
plt.axis("off")
plt.tight_layout()
plt.savefig(OUT / "08_barbara_original_rgb.png", dpi=170,
            bbox_inches="tight", pad_inches=0.08)
plt.close()

# Optional top-left quarter for completeness.
br, bc, _ = barbara.shape
barbara_quarter = barbara[0:br//2, 0:bc//2, :]
save_rgb_display(OUT / "09_barbara_top_left_quarter.png", barbara_quarter,
                 "Barbara - top-left quarter")

# Moving average: filter each R/G/B plane independently.
barbara_ma = filter_rgb_full(barbara, ma)
print("Barbara moving-average shape:", barbara_ma.shape)
save_rgb_display(OUT / "10_barbara_moving_average.png", barbara_ma,
                 "Barbara - 3x3 moving-average filter")

# Save RGB result correctly: convert RGB -> BGR before cv.imwrite.
barbara_ma_u8_rgb = float_to_uint8(barbara_ma)
barbara_ma_u8_bgr = cv.cvtColor(barbara_ma_u8_rgb, cv.COLOR_RGB2BGR)
cv.imwrite(str(OUT / "11_barbara_moving_average_saved.png"), barbara_ma_u8_bgr)

# Difference filter.
barbara_difference = filter_rgb_full(barbara, difference_filter)
print("Barbara difference shape:", barbara_difference.shape)
save_rgb_display(OUT / "12_barbara_difference_filter.png", barbara_difference,
                 "Barbara - 3x3 difference filter (+0.5)", shift=0.5)

# Horizontal and vertical edge experiments, following handout naming/order.
barbara_horizontal = filter_rgb_full(barbara, horizontal_filter)
print("Barbara horizontal-edge shape:", barbara_horizontal.shape)
save_rgb_display(OUT / "13_barbara_horizontal_edges.png", barbara_horizontal,
                 "Barbara - horizontal-edge experiment (+0.5)", shift=0.5)

barbara_vertical = filter_rgb_full(barbara, vertical_filter)
print("Barbara vertical-edge shape:", barbara_vertical.shape)
save_rgb_display(OUT / "14_barbara_vertical_edges.png", barbara_vertical,
                 "Barbara - vertical-edge experiment (+0.5)", shift=0.5)

print("\nLab completed successfully. All output images are in:", OUT)
