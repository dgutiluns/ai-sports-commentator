import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import numpy as np
from src.vision.field_segmentation import segment_field

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'field_segmentation_log.txt')

def test_field_segmentation():
    log_lines = []
    image_path = "data/detector_test_images/ball_detection1.jpg"
    frame = cv2.imread(image_path)
    mask = segment_field(frame)
    field_pixels = np.count_nonzero(mask)
    log_lines.append(f"Field segmentation: {field_pixels} field pixels detected.")
    # Overlay mask on image
    overlay = frame.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0,255,0])).astype(np.uint8)
    out_path = os.path.join(OUTPUT_DIR, "field_segmentation_result.jpg")
    cv2.imwrite(out_path, overlay)
    log_lines.append(f"Field segmentation result saved as {out_path}")
    assert field_pixels > 0, "Field segmentation failed (no field detected)"
    log_lines.append("Field segmentation passed.")
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            print(line)
            f.write(line + '\n')
    print(f"\nLog saved to {LOG_PATH}")

if __name__ == "__main__":
    test_field_segmentation() 