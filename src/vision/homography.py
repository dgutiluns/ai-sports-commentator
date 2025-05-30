import cv2
import numpy as np

# Calibration data for each segment
image_points_start = np.array([
    [725, 312],
    [53, 142],
    [42, 670],
    [1251, 145],
    [1272, 677]
], dtype=np.float32)
field_points_start = np.array([
    [52.5, 34],
    [21.25, 68],
    [40.5, 0],
    [69.5, 68],
    [62.5, 0]
], dtype=np.float32)

image_points_one_third = np.array([
    [404, 82],
    [30, 194],
    [806, 170],
    [997, 109],
    [139, 671]
], dtype=np.float32)
field_points_one_third = np.array([
    [0, 68],
    [0, 43],
    [16.5, 54],
    [21.25, 68],
    [23, 0]
], dtype=np.float32)

image_points_two_third = np.array([
    [919, 105],
    [532, 221],
    [563, 716],
    [135, 341],
    [1276, 122],
    [668, 509]
], dtype=np.float32)
field_points_two_third = np.array([
    [0, 68],
    [0, 43],
    [22, 0],
    [0, 25],
    [11, 68],
    [16.5, 14]
], dtype=np.float32)

# Compute homographies
H_start, _ = cv2.findHomography(image_points_start, field_points_start)
H_one_third, _ = cv2.findHomography(image_points_one_third, field_points_one_third)
H_two_third, _ = cv2.findHomography(image_points_two_third, field_points_two_third)

# Frame ranges (update these based on your video length)
def get_homography_for_frame(frame_idx, total_frames):
    if frame_idx < total_frames // 3:
        return H_start
    elif frame_idx < (2 * total_frames) // 3:
        return H_one_third
    else:
        return H_two_third

def map_to_field(x, y, H):
    pt = np.array([[x, y]], dtype=np.float32)
    pt = np.array([pt])
    mapped = cv2.perspectiveTransform(pt, H)
    return mapped[0][0][0], mapped[0][0][1] 