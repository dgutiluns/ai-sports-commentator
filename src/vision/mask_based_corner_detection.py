#!/usr/bin/env python3
"""
Mask-based corner detection for soccer field boundaries.
This module implements a method to detect field corners by:
1. Detecting line transitions (white-to-black edges) in field masks
2. Fitting average lines for each side of the field
3. Computing intersections to find corner points
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LineSegment:
    """Represents a fitted line segment with start and end points."""
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    angle: float  # in degrees
    confidence: float
    side: str  # 'top', 'bottom', 'left', 'right'

@dataclass
class CornerDetectionResult:
    """Result of corner detection with corners and metadata."""
    corners: Optional[np.ndarray]  # 4x2 array of corner points
    lines: List[LineSegment]
    confidence: float
    debug_info: Dict[str, Any]

def sample_field_boundary(field_mask: np.ndarray, num_samples: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points along the field boundary with outlier filtering.
    
    Args:
        field_mask: Binary field mask (field=255, background=0)
        num_samples: Number of points to sample along boundary
    
    Returns:
        Tuple of (boundary_points, boundary_angles)
    """
    # Find the largest contour (field boundary)
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([]), np.array([])
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour to reduce noise
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)  # More aggressive simplification
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Sample points at regular intervals along the contour
    boundary_points = []
    boundary_angles = []
    
    contour_len = len(simplified_contour)
    if contour_len == 0:
        return np.array([]), np.array([])
    
    # Sample points at regular intervals
    step = max(1, contour_len // num_samples)
    
    for i in range(0, contour_len, step):
        point = simplified_contour[i][0]
        
        # Calculate angle (tangent) at this point
        if i > 0 and i < contour_len - 1:
            prev_point = simplified_contour[i-1][0]
            next_point = simplified_contour[i+1][0]
            dx = next_point[0] - prev_point[0]
            dy = next_point[1] - prev_point[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
        else:
            angle = 0
        
        # Filter out points with extreme angles (likely noise)
        if abs(angle) < 80:  # Filter out near-vertical lines which are usually noise
            boundary_points.append(point)
            boundary_angles.append(angle)
    
    return np.array(boundary_points), np.array(boundary_angles)

def group_boundary_points_by_side(boundary_points: np.ndarray, boundary_angles: np.ndarray, 
                                 field_center: Tuple[float, float]) -> Dict[str, np.ndarray]:
    """
    Group boundary points by which side of the field they belong to using angle-based clustering.
    
    Args:
        boundary_points: Array of boundary points
        boundary_angles: Array of angles at boundary points
        field_center: Center point of the field (x, y)
    
    Returns:
        Dictionary mapping side names to arrays of points
    """
    if len(boundary_points) == 0:
        return {'top': np.array([]), 'bottom': np.array([]), 'left': np.array([]), 'right': np.array([])}
    
    cx, cy = field_center
    sides = {'top': [], 'bottom': [], 'left': [], 'right': []}
    
    # First pass: group by angle clusters
    angle_clusters = {'horizontal': [], 'vertical': []}
    
    for i, (point, angle) in enumerate(zip(boundary_points, boundary_angles)):
        # Normalize angle to 0-180 range
        norm_angle = abs(angle) % 180
        
        # Classify as horizontal or vertical based on angle
        if norm_angle < 45 or norm_angle > 135:  # Horizontal-ish
            angle_clusters['horizontal'].append((point, angle))
        else:  # Vertical-ish
            angle_clusters['vertical'].append((point, angle))
    
    # Second pass: subdivide horizontal and vertical clusters by position
    for point, angle in angle_clusters['horizontal']:
        x, y = point
        if y < cy:  # Above center
            sides['top'].append(point)
        else:  # Below center
            sides['bottom'].append(point)
    
    for point, angle in angle_clusters['vertical']:
        x, y = point
        if x < cx:  # Left of center
            sides['left'].append(point)
        else:  # Right of center
            sides['right'].append(point)
    
    # Convert to numpy arrays
    return {side: np.array(points) if points else np.array([]) for side, points in sides.items()}

def fit_line_ransac(points: np.ndarray, min_inliers: float = 0.6, 
                   max_iterations: int = 100, threshold: float = 5.0) -> Optional[LineSegment]:
    """
    Fit a line to points using RANSAC for robustness.
    
    Args:
        points: Array of points to fit line to
        min_inliers: Minimum fraction of points that must be inliers
        max_iterations: Maximum RANSAC iterations
        threshold: Distance threshold for inlier classification
    
    Returns:
        LineSegment if successful, None otherwise
    """
    if len(points) < 2:
        return None
    
    best_line = None
    best_inliers = 0
    
    for _ in range(max_iterations):
        # Randomly sample 2 points
        if len(points) == 2:
            idx1, idx2 = 0, 1
        else:
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
        
        p1, p2 = points[idx1], points[idx2]
        
        # Fit line through these points
        if p2[0] - p1[0] == 0:  # Vertical line
            a, b, c = 1, 0, -p1[0]
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            a, b, c = -slope, 1, slope * p1[0] - p1[1]
        
        # Count inliers
        inliers = 0
        for point in points:
            distance = abs(a * point[0] + b * point[1] + c) / np.sqrt(a**2 + b**2)
            if distance < threshold:
                inliers += 1
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_line = (a, b, c, p1, p2)
    
    if best_line is None or best_inliers < min_inliers * len(points):
        return None
    
    a, b, c, p1, p2 = best_line
    
    # Calculate line properties
    angle = np.arctan2(-a, b) * 180 / np.pi
    confidence = best_inliers / len(points)
    
    # Determine side based on angle and position
    if abs(angle) < 45:  # Horizontal line
        side = 'top' if p1[1] < np.mean(points[:, 1]) else 'bottom'
    else:  # Vertical line
        side = 'left' if p1[0] < np.mean(points[:, 0]) else 'right'
    
    return LineSegment(
        start_point=tuple(p1),
        end_point=tuple(p2),
        angle=angle,
        confidence=confidence,
        side=side
    )

def extend_line_to_frame_boundaries(line: LineSegment, frame_shape: Tuple[int, int]) -> LineSegment:
    """
    Extend a line segment to the frame boundaries.
    
    Args:
        line: LineSegment to extend
        frame_shape: (height, width) of the frame
    
    Returns:
        Extended LineSegment
    """
    height, width = frame_shape
    
    # Get line parameters
    x1, y1 = line.start_point
    x2, y2 = line.end_point
    
    # Calculate slope and intercept
    if x2 - x1 == 0:  # Vertical line
        slope = float('inf')
        intercept = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    
    # Find intersections with frame boundaries
    intersections = []
    
    # Top boundary (y = 0)
    if slope != float('inf'):
        x_top = -intercept / slope if slope != 0 else x1
        if 0 <= x_top <= width:
            intersections.append((x_top, 0))
    
    # Bottom boundary (y = height)
    if slope != float('inf'):
        x_bottom = (height - intercept) / slope if slope != 0 else x1
        if 0 <= x_bottom <= width:
            intersections.append((x_bottom, height))
    
    # Left boundary (x = 0)
    if slope != float('inf'):
        y_left = intercept
        if 0 <= y_left <= height:
            intersections.append((0, y_left))
    else:
        intersections.append((intercept, 0))
        intersections.append((intercept, height))
    
    # Right boundary (x = width)
    if slope != float('inf'):
        y_right = slope * width + intercept
        if 0 <= y_right <= height:
            intersections.append((width, y_right))
    
    # Find the two intersections that are furthest apart
    if len(intersections) >= 2:
        max_distance = 0
        best_pair = None
        
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                p1, p2 = intersections[i], intersections[j]
                distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                if distance > max_distance:
                    max_distance = distance
                    best_pair = (p1, p2)
        
        if best_pair:
            return LineSegment(
                start_point=best_pair[0],
                end_point=best_pair[1],
                angle=line.angle,
                confidence=line.confidence,
                side=line.side
            )
    
    # Fallback: return original line
    return line

def compute_line_intersections(lines: List[LineSegment]) -> Optional[np.ndarray]:
    """
    Compute intersections of fitted lines to find corners.
    
    Args:
        lines: List of fitted LineSegment objects
    
    Returns:
        4x2 array of corner points if successful, None otherwise
    """
    if len(lines) < 2:
        logger.debug(f"Not enough lines for intersection: {len(lines)}")
        return None
    
    corners = []
    
    # Find intersections between horizontal and vertical lines
    horizontal_lines = [line for line in lines if line.side in ['top', 'bottom']]
    vertical_lines = [line for line in lines if line.side in ['left', 'right']]
    
    logger.debug(f"Horizontal lines: {len(horizontal_lines)}, Vertical lines: {len(vertical_lines)}")
    
    # Compute intersections between horizontal and vertical lines
    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            # Calculate intersection
            intersection = compute_line_intersection(h_line, v_line)
            if intersection is not None:
                corners.append(intersection)
                logger.debug(f"Found intersection: {intersection} between {h_line.side} and {v_line.side}")
    
    # If we don't have enough corners, try intersecting lines with similar orientations
    if len(corners) < 4:
        logger.debug(f"Only {len(corners)} corners found, trying additional intersections")
        
        # Try intersecting lines with different sides but similar orientations
        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i+1:], i+1):
                if line1.side != line2.side:  # Different sides
                    intersection = compute_line_intersection(line1, line2)
                    if intersection is not None:
                        corners.append(intersection)
                        logger.debug(f"Found additional intersection: {intersection} between {line1.side} and {line2.side}")
    
    logger.debug(f"Total corners found: {len(corners)}")
    
    # If we have enough corners, return them
    if len(corners) >= 4:
        corners = np.array(corners)
        # Sort corners in standard order: TL, TR, BR, BL
        corners = sort_corners_standard_order(corners)
        logger.debug(f"Returning {len(corners[:4])} sorted corners")
        return corners[:4]  # Return exactly 4 corners
    elif len(corners) > 0:
        logger.debug(f"Returning {len(corners)} corners (less than 4)")
        return np.array(corners)
    
    logger.debug("No corners found")
    return None

def compute_line_intersection(line1: LineSegment, line2: LineSegment) -> Optional[Tuple[float, float]]:
    """
    Compute intersection point of two line segments.
    
    Args:
        line1, line2: LineSegment objects
    
    Returns:
        Intersection point (x, y) if lines intersect, None otherwise
    """
    # Convert line segments to general form ax + by + c = 0
    def line_to_general_form(line):
        x1, y1 = line.start_point
        x2, y2 = line.end_point
        if x2 - x1 == 0:  # Vertical line
            return 1, 0, -x1
        else:
            slope = (y2 - y1) / (x2 - x1)
            return -slope, 1, slope * x1 - y1
    
    a1, b1, c1 = line_to_general_form(line1)
    a2, b2, c2 = line_to_general_form(line2)
    
    # Solve system of equations
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:  # Lines are parallel
        return None
    
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    
    return (x, y)

def compute_extended_line_intersection(line1: LineSegment, line2: LineSegment) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of two lines extended beyond their segments.
    """
    # This is a simplified version - in practice, you'd want more sophisticated logic
    return compute_line_intersection(line1, line2)

def sort_corners_standard_order(corners: np.ndarray) -> np.ndarray:
    """
    Sort corners in standard order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    
    Args:
        corners: Array of corner points
    
    Returns:
        Sorted array of corners
    """
    if len(corners) == 0:
        return corners
    
    # Find center
    center = np.mean(corners, axis=0)
    
    # Sort by angle from center
    angles = []
    for corner in corners:
        angle = np.arctan2(corner[1] - center[1], corner[0] - center[0])
        angles.append(angle)
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    sorted_corners = corners[sorted_indices]
    
    return sorted_corners

def mask_based_corner_detection(field_mask: np.ndarray, 
                               num_samples: int = 200,
                               min_inliers: float = 0.6,
                               internal_line_threshold: float = 0.5,
                               min_boundary_inliers: int = 20,
                               debug: bool = False) -> CornerDetectionResult:
    """
    Main function for mask-based corner detection with refined filtering and scoring logic.
    
    Args:
        field_mask: Binary field mask (field=255, background=0)
        num_samples: Number of boundary points to sample
        min_inliers: Minimum fraction of inliers for line fitting
        internal_line_threshold: Threshold for detecting internal lines (0.5 = 50% of line must have boundary on both sides)
        min_boundary_inliers: Minimum number of boundary-aligned points required for a line
        debug: Whether to include debug information
    
    Returns:
        CornerDetectionResult with corners and metadata
    """
    debug_info = {}
    
    # Ensure mask is binary
    if field_mask.dtype != np.uint8:
        field_mask = field_mask.astype(np.uint8)
    
    # Convert to binary if needed
    if field_mask.max() > 1:
        field_mask = (field_mask > 0).astype(np.uint8) * 255
    
    # Get frame shape
    frame_shape = field_mask.shape
    height, width = frame_shape
    debug_info['frame_shape'] = frame_shape
    
    # Find field center
    field_pixels = np.argwhere(field_mask > 0)
    if len(field_pixels) == 0:
        logger.warning("No field pixels found in mask")
        return CornerDetectionResult(None, [], 0.0, debug_info)
    
    field_center = np.mean(field_pixels, axis=0)[::-1]  # Convert to (x, y)
    debug_info['field_center'] = field_center
    
    # Sample boundary points with improved filtering
    boundary_points, boundary_angles = sample_field_boundary_improved(field_mask, num_samples, min_boundary_inliers)
    debug_info['boundary_points_count'] = len(boundary_points)
    
    if len(boundary_points) == 0:
        logger.warning("No boundary points found after filtering")
        return CornerDetectionResult(None, [], 0.0, debug_info)
    
    # Group points by side using image-relative positioning
    side_groups = group_boundary_points_by_side_improved(boundary_points, boundary_angles, frame_shape)
    debug_info['side_groups'] = {side: len(points) for side, points in side_groups.items()}
    
    # Fit lines to each side with comprehensive scoring
    fitted_lines = []
    rejected_lines = []
    
    for side, points in side_groups.items():
        if len(points) >= 3:  # Require at least 3 points for better line fitting
            # Fit line with RANSAC
            line = fit_line_ransac_improved(points, min_inliers=0.4, threshold=8.0, frame_shape=frame_shape)
            if line is not None:
                # Check if this is an internal line (like halfway line)
                if is_internal_line(line, field_mask, internal_line_threshold):
                    logger.debug(f"Rejecting internal line for {side}: likely field marking")
                    rejected_lines.append((line, f"Internal line detected for {side}"))
                    continue
                
                # Calculate comprehensive confidence score
                confidence_score = calculate_line_confidence(line, field_mask, frame_shape, side)
                line.confidence = confidence_score
                
                # Extend line to frame boundaries
                extended_line = extend_line_to_frame_boundaries(line, frame_shape)
                fitted_lines.append(extended_line)
                logger.debug(f"Fitted line for {side}: angle={extended_line.angle:.1f}°, confidence={extended_line.confidence:.2f}")
            else:
                logger.debug(f"Failed to fit line for {side}: insufficient inliers")
    
    debug_info['fitted_lines_count'] = len(fitted_lines)
    debug_info['rejected_lines_count'] = len(rejected_lines)
    debug_info['rejected_lines'] = [f"{line.side}: {reason}" for line, reason in rejected_lines]
    
    # Ensure we have both horizontal and vertical lines
    horizontal_lines = [line for line in fitted_lines if line.side in ['top', 'bottom']]
    vertical_lines = [line for line in fitted_lines if line.side in ['left', 'right']]
    
    debug_info['horizontal_lines'] = len(horizontal_lines)
    debug_info['vertical_lines'] = len(vertical_lines)
    
    logger.debug(f"Initial line detection: {len(fitted_lines)} total lines")
    logger.debug(f"  Horizontal: {len(horizontal_lines)} lines")
    logger.debug(f"  Vertical: {len(vertical_lines)} lines")
    logger.debug(f"  Rejected: {len(rejected_lines)} lines")
    
    for line in fitted_lines:
        logger.debug(f"  Line {line.side}: angle={line.angle:.1f}°, confidence={line.confidence:.2f}")
    
    # If we don't have both types, try to fit additional lines with more relaxed parameters
    if len(horizontal_lines) == 0 or len(vertical_lines) == 0:
        logger.debug("Missing horizontal or vertical lines, trying with more relaxed parameters")
        
        for side, points in side_groups.items():
            if len(points) >= 2:  # Lower threshold for additional lines
                # Check if we already have a line for this side
                existing_lines = [line for line in fitted_lines if line.side == side]
                if not existing_lines:
                    # Try with very relaxed parameters
                    line = fit_line_ransac_improved(points, min_inliers=0.3, threshold=12.0, frame_shape=frame_shape)
                    if line is not None:
                        # Still check for internal lines
                        if is_internal_line(line, field_mask, internal_line_threshold):
                            logger.debug(f"Rejecting additional internal line for {side}")
                            rejected_lines.append((line, f"Additional internal line for {side}"))
                            continue
                        
                        # Calculate confidence
                        confidence_score = calculate_line_confidence(line, field_mask, frame_shape, side)
                        line.confidence = confidence_score
                        
                        extended_line = extend_line_to_frame_boundaries(line, frame_shape)
                        fitted_lines.append(extended_line)
                        logger.debug(f"Added additional line for {side}: angle={extended_line.angle:.1f}°, confidence={extended_line.confidence:.2f}")
    
    # Compute corner intersections
    corners = compute_line_intersections(fitted_lines)
    
    # Calculate overall confidence
    if fitted_lines:
        avg_confidence = np.mean([line.confidence for line in fitted_lines])
    else:
        avg_confidence = 0.0
    
    debug_info['avg_line_confidence'] = avg_confidence
    debug_info['corners_found'] = corners is not None
    debug_info['final_lines_count'] = len(fitted_lines)
    
    if corners is not None:
        logger.info(f"Detected {len(corners)} corners with confidence {avg_confidence:.2f}")
    else:
        logger.warning("Failed to detect corners")
    
    return CornerDetectionResult(
        corners=corners,
        lines=fitted_lines,
        confidence=avg_confidence,
        debug_info=debug_info
    )

def create_corner_visualization(field_mask: np.ndarray, 
                               result: CornerDetectionResult,
                               original_frame: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create visualization of detected corners and fitted lines with extended lines and rejection info.
    
    Args:
        field_mask: Original field mask
        result: CornerDetectionResult
        original_frame: Optional original frame for overlay
    
    Returns:
        Visualization image
    """
    # Create visualization image
    if original_frame is not None:
        vis_img = original_frame.copy()
    else:
        # Create RGB image from mask
        vis_img = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
    
    # Draw fitted lines (now extended to frame boundaries)
    for line in result.lines:
        color = {
            'top': (0, 255, 0),      # Green
            'bottom': (0, 255, 255),  # Yellow
            'left': (255, 0, 0),      # Blue
            'right': (255, 0, 255)    # Magenta
        }.get(line.side, (255, 255, 255))
        
        # Draw extended line segment
        cv2.line(vis_img, 
                (int(line.start_point[0]), int(line.start_point[1])),
                (int(line.end_point[0]), int(line.end_point[1])),
                color, 3)  # Thicker lines to show they're extended
        
        # Add label with angle and confidence information
        mid_point = ((line.start_point[0] + line.end_point[0]) // 2,
                    (line.start_point[1] + line.end_point[1]) // 2)
        cv2.putText(vis_img, f"{line.side} ({line.angle:.1f}°, {line.confidence:.2f})",
                   (int(mid_point[0]), int(mid_point[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw detected corners
    if result.corners is not None:
        for i, corner in enumerate(result.corners):
            cv2.circle(vis_img, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), -1)
            cv2.putText(vis_img, f"C{i}", (int(corner[0]) + 15, int(corner[1]) + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add detailed information with rejection details
    debug_info = result.debug_info
    y_offset = 30
    
    # Basic stats
    cv2.putText(vis_img, f"Lines: {debug_info.get('final_lines_count', 0)} (H:{debug_info.get('horizontal_lines', 0)} V:{debug_info.get('vertical_lines', 0)})",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Rejection information
    rejected_count = debug_info.get('rejected_lines_count', 0)
    if rejected_count > 0:
        cv2.putText(vis_img, f"Rejected: {rejected_count} lines", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)  # Orange
        y_offset += 30
        
        # Show rejection reasons
        rejected_lines = debug_info.get('rejected_lines', [])
        for i, reason in enumerate(rejected_lines[:3]):  # Show first 3 reasons
            cv2.putText(vis_img, f"  - {reason}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            y_offset += 20
    
    # Confidence information
    cv2.putText(vis_img, f"Confidence: {result.confidence:.2f}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(vis_img, f"Corners: {len(result.corners) if result.corners is not None else 0}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Frame shape and boundary info
    frame_shape = debug_info.get('frame_shape', 'N/A')
    cv2.putText(vis_img, f"Frame: {frame_shape}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 20
    cv2.putText(vis_img, f"Boundary points: {debug_info.get('boundary_points_count', 0)}",
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return vis_img

def sample_field_boundary_improved(field_mask: np.ndarray, num_samples: int = 200, min_boundary_inliers: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample points along the field boundary with improved filtering for continuous clusters.
    
    Args:
        field_mask: Binary field mask (field=255, background=0)
        num_samples: Number of points to sample along boundary
        min_boundary_inliers: Minimum number of boundary-aligned points required
    
    Returns:
        Tuple of (boundary_points, boundary_angles)
    """
    # Find contours - use all contours for now, don't filter by area
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([]), np.array([])
    
    # Use the largest contour (main field boundary)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour to reduce noise
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    simplified_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Sample points at regular intervals along the contour
    boundary_points = []
    boundary_angles = []
    
    contour_len = len(simplified_contour)
    if contour_len == 0:
        return np.array([]), np.array([])
    
    # Sample points at regular intervals
    step = max(1, contour_len // num_samples)
    
    for i in range(0, contour_len, step):
        point = simplified_contour[i][0]
        
        # Calculate angle (tangent) at this point
        if i > 0 and i < contour_len - 1:
            prev_point = simplified_contour[i-1][0]
            next_point = simplified_contour[i+1][0]
            dx = next_point[0] - prev_point[0]
            dy = next_point[1] - prev_point[1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
        else:
            angle = 0
        
        # Filter out points with extreme angles (likely noise) - less restrictive
        if abs(angle) < 85:  # Allow more vertical lines
            boundary_points.append(point)
            boundary_angles.append(angle)
    
    # Only check minimum if we have very few points
    if len(boundary_points) < min_boundary_inliers and len(boundary_points) > 0:
        logger.warning(f"Low boundary points: {len(boundary_points)} < {min_boundary_inliers}, but continuing")
    
    return np.array(boundary_points), np.array(boundary_angles)

def group_boundary_points_by_side_improved(boundary_points: np.ndarray, boundary_angles: np.ndarray, 
                                          frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
    """
    Group boundary points by which side of the field they belong to using image-relative positioning.
    
    Args:
        boundary_points: Array of boundary points
        boundary_angles: Array of angles at boundary points
        frame_shape: (height, width) of the frame
    
    Returns:
        Dictionary mapping side names to arrays of points
    """
    if len(boundary_points) == 0:
        return {'top': np.array([]), 'bottom': np.array([]), 'left': np.array([]), 'right': np.array([])}
    
    height, width = frame_shape
    sides = {'top': [], 'bottom': [], 'left': [], 'right': []}
    
    # First pass: group by angle clusters
    angle_clusters = {'horizontal': [], 'vertical': []}
    
    for i, (point, angle) in enumerate(zip(boundary_points, boundary_angles)):
        # Normalize angle to 0-180 range
        norm_angle = abs(angle) % 180
        
        # Classify as horizontal or vertical based on angle
        if norm_angle < 45 or norm_angle > 135:  # Horizontal-ish
            angle_clusters['horizontal'].append((point, angle))
        else:  # Vertical-ish
            angle_clusters['vertical'].append((point, angle))
    
    # Second pass: subdivide horizontal and vertical clusters by image-relative position
    for point, angle in angle_clusters['horizontal']:
        x, y = point
        # Use image-relative positioning: top 60% vs bottom 40%
        if y < height * 0.6:  # Top 60% of frame
            sides['top'].append(point)
        else:  # Bottom 40% of frame
            sides['bottom'].append(point)
    
    for point, angle in angle_clusters['vertical']:
        x, y = point
        # Use image-relative positioning: left 50% vs right 50%
        if x < width * 0.5:  # Left half of frame
            sides['left'].append(point)
        else:  # Right half of frame
            sides['right'].append(point)
    
    # Convert to numpy arrays
    return {side: np.array(points) if points else np.array([]) for side, points in sides.items()}

def fit_line_ransac_improved(points: np.ndarray, min_inliers: float = 0.6, 
                            max_iterations: int = 100, threshold: float = 5.0,
                            frame_shape: Tuple[int, int] = None) -> Optional[LineSegment]:
    """
    Fit a line to points using RANSAC with improved side labeling.
    
    Args:
        points: Array of points to fit line to
        min_inliers: Minimum fraction of points that must be inliers
        max_iterations: Maximum RANSAC iterations
        threshold: Distance threshold for inlier classification
        frame_shape: Frame shape for position-based side labeling
    
    Returns:
        LineSegment if successful, None otherwise
    """
    if len(points) < 2:
        return None
    
    best_line = None
    best_inliers = 0
    
    for _ in range(max_iterations):
        # Randomly sample 2 points
        if len(points) == 2:
            idx1, idx2 = 0, 1
        else:
            idx1, idx2 = np.random.choice(len(points), 2, replace=False)
        
        p1, p2 = points[idx1], points[idx2]
        
        # Fit line through these points
        if p2[0] - p1[0] == 0:  # Vertical line
            a, b, c = 1, 0, -p1[0]
        else:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
            a, b, c = -slope, 1, slope * p1[0] - p1[1]
        
        # Count inliers
        inliers = 0
        for point in points:
            distance = abs(a * point[0] + b * point[1] + c) / np.sqrt(a**2 + b**2)
            if distance < threshold:
                inliers += 1
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_line = (a, b, c, p1, p2)
    
    if best_line is None or best_inliers < min_inliers * len(points):
        return None
    
    a, b, c, p1, p2 = best_line
    
    # Calculate line properties
    angle = np.arctan2(-a, b) * 180 / np.pi
    confidence = best_inliers / len(points)
    
    # Determine side based on angle and image-relative position
    if frame_shape is not None:
        height, width = frame_shape
        avg_y = np.mean(points[:, 1])
        avg_x = np.mean(points[:, 0])
        
        if abs(angle) < 45:  # Horizontal line
            side = 'top' if avg_y < height * 0.6 else 'bottom'
        else:  # Vertical line
            side = 'left' if avg_x < width * 0.5 else 'right'
    else:
        # Fallback to old method
        if abs(angle) < 45:  # Horizontal line
            side = 'top' if p1[1] < np.mean(points[:, 1]) else 'bottom'
        else:  # Vertical line
            side = 'left' if p1[0] < np.mean(points[:, 0]) else 'right'
    
    return LineSegment(
        start_point=tuple(p1),
        end_point=tuple(p2),
        angle=angle,
        confidence=confidence,
        side=side
    )

def is_internal_line(line: LineSegment, field_mask: np.ndarray, threshold: float = 0.5) -> bool:
    """
    Check if a line is likely an internal field marking (like halfway line) by testing
    if it has boundary points on both sides.
    
    Args:
        line: LineSegment to test
        field_mask: Binary field mask
        threshold: Threshold for internal line detection (0.5 = 50% of line must have boundary on both sides)
    
    Returns:
        True if line is likely internal, False otherwise
    """
    height, width = field_mask.shape
    
    # Sample points along the line
    num_samples = 50
    line_points = []
    
    x1, y1 = line.start_point
    x2, y2 = line.end_point
    
    for i in range(num_samples):
        t = i / (num_samples - 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        # Ensure point is within frame bounds
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        line_points.append((x, y))
    
    # Check each point for boundary presence on both sides
    points_with_boundary_on_both_sides = 0
    
    for x, y in line_points:
        # Check perpendicular to line direction
        if abs(line.angle) < 45:  # Horizontal line, check vertical direction
            # Check above and below the line
            above_y = max(0, y - 10)
            below_y = min(height - 1, y + 10)
            
            above_is_field = field_mask[above_y, x] > 0
            below_is_field = field_mask[below_y, x] > 0
            
            # If both above and below are field, this point has boundary on both sides
            if above_is_field and below_is_field:
                points_with_boundary_on_both_sides += 1
        else:  # Vertical line, check horizontal direction
            # Check left and right of the line
            left_x = max(0, x - 10)
            right_x = min(width - 1, x + 10)
            
            left_is_field = field_mask[y, left_x] > 0
            right_is_field = field_mask[y, right_x] > 0
            
            # If both left and right are field, this point has boundary on both sides
            if left_is_field and right_is_field:
                points_with_boundary_on_both_sides += 1
    
    # Calculate fraction of points with boundary on both sides
    fraction_with_boundary_on_both_sides = points_with_boundary_on_both_sides / len(line_points)
    
    return fraction_with_boundary_on_both_sides > threshold

def calculate_line_confidence(line: LineSegment, field_mask: np.ndarray, 
                             frame_shape: Tuple[int, int], expected_side: str) -> float:
    """
    Calculate comprehensive confidence score for a line based on multiple criteria.
    
    Args:
        line: LineSegment to score
        field_mask: Binary field mask
        frame_shape: Frame shape for position validation
        expected_side: Expected side label for position validation
    
    Returns:
        Confidence score between 0 and 1
    """
    height, width = frame_shape
    
    # 1. Position match score (0-1)
    position_score = 0.0
    avg_y = (line.start_point[1] + line.end_point[1]) / 2
    avg_x = (line.start_point[0] + line.end_point[0]) / 2
    
    if expected_side == 'top':
        # Should be in top 60% of frame
        if avg_y < height * 0.6:
            position_score = 1.0
        else:
            position_score = max(0, 1.0 - (avg_y - height * 0.6) / (height * 0.4))
    elif expected_side == 'bottom':
        # Should be in bottom 40% of frame
        if avg_y > height * 0.6:
            position_score = 1.0
        else:
            position_score = max(0, avg_y / (height * 0.6))
    elif expected_side == 'left':
        # Should be in left 50% of frame
        if avg_x < width * 0.5:
            position_score = 1.0
        else:
            position_score = max(0, 1.0 - (avg_x - width * 0.5) / (width * 0.5))
    elif expected_side == 'right':
        # Should be in right 50% of frame
        if avg_x > width * 0.5:
            position_score = 1.0
        else:
            position_score = max(0, avg_x / (width * 0.5))
    
    # 2. Boundary alignment score (0-1)
    boundary_score = 0.0
    num_samples = 20
    total_distance = 0.0
    
    x1, y1 = line.start_point
    x2, y2 = line.end_point
    
    for i in range(num_samples):
        t = i / (num_samples - 1)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        # Find nearest boundary point
        x, y = int(x), int(y)
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        # Simple distance to boundary (could be improved with actual boundary detection)
        # For now, use distance to nearest non-field pixel
        min_dist = float('inf')
        for dy in range(-10, 11):
            for dx in range(-10, 11):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    if field_mask[ny, nx] == 0:  # Background pixel
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = min(min_dist, dist)
        
        if min_dist != float('inf'):
            total_distance += min_dist
    
    if num_samples > 0:
        avg_distance = total_distance / num_samples
        # Convert distance to score (closer = higher score)
        boundary_score = max(0, 1.0 - avg_distance / 20.0)  # Normalize by max expected distance
    
    # 3. Segment consistency score (0-1)
    consistency_score = 1.0  # Default to perfect consistency
    
    # Check for sharp inflections by comparing angles at different points
    if len(line.start_point) == 2 and len(line.end_point) == 2:
        dx = line.end_point[0] - line.start_point[0]
        dy = line.end_point[1] - line.start_point[1]
        line_angle = np.arctan2(dy, dx) * 180 / np.pi
        
        # If angle is very different from expected for the side, penalize
        if expected_side in ['top', 'bottom'] and abs(line_angle) > 30:
            consistency_score *= 0.5
        elif expected_side in ['left', 'right'] and abs(line_angle) < 60:
            consistency_score *= 0.5
    
    # 4. Combine scores with weights
    weights = {
        'position': 0.3,
        'boundary': 0.4,
        'consistency': 0.3
    }
    
    final_score = (weights['position'] * position_score + 
                   weights['boundary'] * boundary_score + 
                   weights['consistency'] * consistency_score)
    
    return min(1.0, max(0.0, final_score)) 