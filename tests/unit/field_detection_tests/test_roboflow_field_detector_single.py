import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import numpy as np
from glob import glob
from src.vision.detector import RoboflowFieldDetector

# Settings
IMAGE_DIR = "data/soccer_ball/images/train"
OUTPUT_DIR = "tests/unit/outputs/test_roboflow_field_detector_single"
MODEL_ID = "football-field-detection-f07vi/15"
NUM_IMAGES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample a few images
image_paths = sorted(glob(os.path.join(IMAGE_DIR, "*.jpg")))[:NUM_IMAGES]
if not image_paths:
    print(f"No images found in {IMAGE_DIR}")
    exit(1)

# Initialize detector
field_detector = RoboflowFieldDetector(model_id=MODEL_ID)

for idx, img_path in enumerate(image_paths):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Failed to load {img_path}")
        continue
    mask, contours = field_detector.detect_field(frame)
    mask_pixels = int(np.sum(mask > 0))
    print(f"Image {os.path.basename(img_path)}: mask pixels={mask_pixels}, num contours={len(contours)}")
    # Draw contours in red
    vis = frame.copy()
    if contours:
        cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)
    # Optionally, overlay mask in blue
    if np.any(mask):
        vis[mask > 0] = cv2.addWeighted(vis[mask > 0], 0.7, np.full_like(vis[mask > 0], [255, 0, 0]), 0.3, 0)
    out_path = os.path.join(OUTPUT_DIR, f"roboflow_field_{idx:02d}.jpg")
    cv2.imwrite(out_path, vis)
    print(f"  Saved: {out_path}") 