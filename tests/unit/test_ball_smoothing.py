import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.vision.ball_smoothing import filter_and_smooth_ball_positions

def test_ball_smoothing():
    # Simulate ball positions with missing frames
    positions = [
        {'frame': 0, 'x': 100, 'y': 200},
        None,
        {'frame': 2, 'x': 120, 'y': 220},
        None,
        {'frame': 4, 'x': 140, 'y': 240}
    ]
    smoothed = filter_and_smooth_ball_positions(positions)
    print("Original positions:")
    for i, p in enumerate(positions):
        print(f"  Frame {i}: {p}")
    print("Smoothed positions:")
    for i, p in enumerate(smoothed):
        print(f"  Frame {i}: {p}")
    assert smoothed[1] is not None and smoothed[3] is not None, "Ball smoothing failed to fill missing frames"
    print("Ball smoothing test passed.")

if __name__ == "__main__":
    test_ball_smoothing() 