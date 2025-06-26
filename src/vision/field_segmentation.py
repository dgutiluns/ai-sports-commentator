import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

def morphological_opening(img, diameter_ratio=0.01):
    """
    Apply morphological opening to each channel to blend white lines into the grass.
    """
    h, w = img.shape[:2]
    diameter = int(diameter_ratio * np.sqrt(h**2 + w**2))
    if diameter < 1:
        diameter = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    opened = np.zeros_like(img)
    for c in range(3):
        opened[..., c] = cv2.morphologyEx(img[..., c], cv2.MORPH_OPEN, kernel)
    return opened

def green_chromaticity_mask(img):
    """
    Segment the field using green chromaticity and GMM thresholding.
    """
    img = img.astype(np.float32) + 1e-6
    R, G, B = img[..., 2], img[..., 1], img[..., 0]
    g = G / (R + G + B)
    g_flat = g.flatten().reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(g_flat)
    means = gmm.means_.flatten()
    green_idx = np.argmax(means)
    green_mask = (gmm.predict(g_flat) == green_idx).reshape(g.shape)
    return green_mask.astype(np.uint8)

def chromatic_distortion_filter(img, mask, angle_thresh_deg=15):
    """
    Remove non-grass regions (ads, uniforms) using chromatic distortion filtering.
    """
    # Only consider masked pixels
    pixels = img[mask > 0].reshape(-1, 3).astype(np.float32)
    if len(pixels) == 0:
        return mask
    mean_vec = np.mean(pixels, axis=0)
    mean_vec /= np.linalg.norm(mean_vec) + 1e-6
    # Compute angle between each pixel and mean
    norm_pixels = pixels / (np.linalg.norm(pixels, axis=1, keepdims=True) + 1e-6)
    cos_angles = np.dot(norm_pixels, mean_vec)
    angles = np.arccos(np.clip(cos_angles, -1, 1)) * 180 / np.pi
    keep = angles < angle_thresh_deg
    # Create new mask
    new_mask = np.zeros_like(mask, dtype=np.uint8)
    idx = np.argwhere(mask > 0)
    for i, k in enumerate(keep):
        if k:
            y, x = idx[i]
            new_mask[y, x] = 1
    return new_mask

def largest_region_mask(mask):
    """
    Keep only the largest connected region in the mask.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

def segment_field(img):
    """
    Full pipeline: preprocess, segment, filter, and return field mask.
    """
    opened = morphological_opening(img)
    green_mask = green_chromaticity_mask(opened)
    filtered_mask = chromatic_distortion_filter(opened, green_mask)
    closed = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    field_mask = largest_region_mask(closed)
    return field_mask

def get_field_corners_from_mask(field_mask):
    """
    Extract field corners from a segmentation mask.
    Returns 4 points (corners) or None if not found.
    """
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) == 4:
        corners = approx.reshape(4, 2)
        return corners
    else:
        # Fallback: use minimum area rectangle
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        return box.astype(int) 