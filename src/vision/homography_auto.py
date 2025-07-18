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
        corners_result = get_field_corners_from_mask(field_mask)
        if corners_result is None:
            return None
        
        corners, fallback_used = corners_result
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
        pt = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pt, H)
        return mapped[0][0][0], mapped[0][0][1] 

    def get_field_corners(self, field_mask):
        """Extract field corners from the field mask for visualization."""
        try:
            result = get_field_corners_from_mask(field_mask)
            if result is None:
                return None
            corners, fallback_used = result
            return corners
        except Exception as e:
            print(f"Error getting field corners: {e}")
            return None

    def create_warped_grid(self, H, image_shape):
        """Create a warped grid visualization for homography validation."""
        if H is None:
            return None
        
        try:
            height, width = image_shape
            # Create a grid pattern
            grid_size = 20
            grid_points = []
            
            for i in range(0, width, grid_size):
                for j in range(0, height, grid_size):
                    grid_points.append([i, j])
            
            if not grid_points:
                return None
            
            grid_points = np.array(grid_points, dtype=np.float32).reshape(-1, 1, 2)
            
            # Apply homography transformation
            warped_points = cv2.perspectiveTransform(grid_points, H)
            
            # Create output image
            output = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw warped grid points
            for point in warped_points:
                x, y = int(point[0][0]), int(point[0][1])
                if 0 <= x < width and 0 <= y < height:
                    cv2.circle(output, (x, y), 2, (255, 255, 255), -1)
            
            return output
            
        except Exception as e:
            print(f"Error creating warped grid: {e}")
            return None 