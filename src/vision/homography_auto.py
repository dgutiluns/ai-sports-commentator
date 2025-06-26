import numpy as np
import cv2
from src.vision.field_segmentation import get_field_corners_from_mask

class HomographyEstimator:
    def __init__(self, field_coords, smoothing=0):
        """
        field_coords: np.ndarray of shape (4,2), real-world field corners (order: TL, BL, BR, TR)
        smoothing: int, number of frames for running average smoothing (0 = no smoothing)
        """
        self.field_coords = field_coords.astype(np.float32)
        self.last_valid_H = None
        self.smoothing = smoothing
        self.H_history = []

    def estimate(self, field_mask):
        corners = get_field_corners_from_mask(field_mask)
        H = None
        if corners is not None and len(corners) == 4:
            corners = corners.astype(np.float32)
            try:
                H, status = cv2.findHomography(corners, self.field_coords)
                if H is not None and self._is_valid_homography(H):
                    self.last_valid_H = H
                    if self.smoothing > 0:
                        self.H_history.append(H)
                        if len(self.H_history) > self.smoothing:
                            self.H_history.pop(0)
                        H = np.mean(self.H_history, axis=0)
            except Exception as e:
                print(f"Homography estimation failed: {e}")
        if H is None:
            H = self.last_valid_H
        return H

    def _is_valid_homography(self, H):
        # Simple check: determinant should not be too close to zero
        return np.abs(np.linalg.det(H)) > 1e-8

    def map_point(self, x, y, H=None):
        if H is None:
            H = self.last_valid_H
        if H is None:
            return None, None
        pt = np.array([[x, y]], dtype=np.float32)
        pt = np.array([pt])
        mapped = cv2.perspectiveTransform(pt, H)
        return mapped[0][0][0], mapped[0][0][1] 