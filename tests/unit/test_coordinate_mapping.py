import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
from src.vision.homography_auto import HomographyEstimator

def test_coordinate_mapping():
    FIELD_COORDS = np.array([
        [0, 0],
        [0, 68],
        [105, 68],
        [105, 0]
    ], dtype=np.float32)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[100:600, 200:1000] = 255
    estimator = HomographyEstimator(FIELD_COORDS, smoothing=0)
    H = estimator.estimate(mask)
    x, y = estimator.map_point(600, 350, H)
    print(f"Mapped point (600,350) -> ({x:.2f}, {y:.2f})")
    assert isinstance(x, float) and isinstance(y, float), "Coordinate mapping failed"
    print("Coordinate mapping passed.")

if __name__ == "__main__":
    test_coordinate_mapping() 