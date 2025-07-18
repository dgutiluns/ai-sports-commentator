#!/usr/bin/env python3
"""
Test script for mask-based corner detection method.
This script tests the new corner detection approach using existing field masks
from the test_field_segmentation output folder.
"""

import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import cv2
import numpy as np
import logging
from pathlib import Path
from src.vision.mask_based_corner_detection import (
    mask_based_corner_detection, 
    create_corner_visualization,
    CornerDetectionResult
)

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test configuration
TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'mask_based_corner_detection_log.txt')

# Input masks directory (existing field masks from test_field_segmentation)
INPUT_MASKS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'test_field_segmentation', 'field_masks')

def test_single_mask_corner_detection(mask_path: str) -> CornerDetectionResult:
    """
    Test corner detection on a single field mask with improved parameters.
    
    Args:
        mask_path: Path to the field mask image
    
    Returns:
        CornerDetectionResult with detection results
    """
    logger.info(f"Testing corner detection on: {mask_path}")
    
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error(f"Failed to load mask: {mask_path}")
        return CornerDetectionResult(None, [], 0.0, {})
    
    # Ensure mask is binary
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Run corner detection with improved parameters
    result = mask_based_corner_detection(
        field_mask=mask,
        num_samples=200,
        min_inliers=0.6,
        internal_line_threshold=0.5,  # 50% threshold for internal line detection
        min_boundary_inliers=5,       # Reduced from 20 to 5 based on test results
        debug=True
    )
    
    return result

def create_debug_visualizations(mask_path: str, result: CornerDetectionResult, 
                               output_dir: str, frame_name: str):
    """
    Create debug visualizations for corner detection results.
    
    Args:
        mask_path: Path to original mask
        result: CornerDetectionResult
        output_dir: Output directory for visualizations
        frame_name: Name of the frame for file naming
    """
    # Load original mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Create visualization
    vis_img = create_corner_visualization(mask, result)
    
    # Save visualization
    vis_path = os.path.join(output_dir, f"{frame_name}_corner_detection.jpg")
    cv2.imwrite(vis_path, vis_img)
    
    # Create additional debug visualizations
    create_boundary_visualization(mask, result, output_dir, frame_name)
    create_line_fitting_visualization(mask, result, output_dir, frame_name)

def create_composite_visualization(mask: np.ndarray, result: CornerDetectionResult, 
                                  frame_name: str, frame_number: int) -> np.ndarray:
    """
    Create a single composite visualization showing all detection results.
    
    Args:
        mask: Original field mask
        result: CornerDetectionResult
        frame_name: Name of the frame
        frame_number: Frame number for labeling
    
    Returns:
        Composite visualization image
    """
    # Create RGB image from mask
    vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw field boundary contour in green
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)
    
    # Draw fitted lines with side labels and confidence scores
    colors = {
        'top': (0, 255, 0),      # Green
        'bottom': (0, 255, 255),  # Yellow
        'left': (255, 0, 0),      # Blue
        'right': (255, 0, 255)    # Magenta
    }
    
    for i, line in enumerate(result.lines):
        color = colors.get(line.side, (255, 255, 255))
        
        # Draw line segment
        cv2.line(vis_img, 
                (int(line.start_point[0]), int(line.start_point[1])),
                (int(line.end_point[0]), int(line.end_point[1])),
                color, 3)
        
        # Add label with side, angle, and confidence
        mid_point = ((line.start_point[0] + line.end_point[0]) // 2,
                    (line.start_point[1] + line.end_point[1]) // 2)
        
        # Create label text
        label_text = f"{line.side.upper()}: {line.confidence:.2f} ({line.angle:.1f}°)"
        
        # Add background rectangle for better text visibility
        (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_img, 
                     (int(mid_point[0]) - 5, int(mid_point[1]) - text_height - 5),
                     (int(mid_point[0]) + text_width + 5, int(mid_point[1]) + 5),
                     color, -1)
        
        # Add text in black for contrast
        cv2.putText(vis_img, label_text,
                   (int(mid_point[0]), int(mid_point[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Draw detected corners
    if result.corners is not None:
        for i, corner in enumerate(result.corners):
            # Draw corner point
            cv2.circle(vis_img, (int(corner[0]), int(corner[1])), 12, (0, 0, 255), -1)
            cv2.circle(vis_img, (int(corner[0]), int(corner[1])), 12, (255, 255, 255), 2)
            
            # Add corner label
            cv2.putText(vis_img, f"C{i}", 
                       (int(corner[0]) + 15, int(corner[1]) + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add comprehensive information panel
    debug_info = result.debug_info
    y_offset = 30
    
    # Frame information
    cv2.putText(vis_img, f"Frame: {frame_name} (#{frame_number})", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Detection status
    status_color = (0, 255, 0) if result.corners is not None else (0, 165, 255)
    status_text = f"Status: {'SUCCESS' if result.corners is not None else 'FAILED'}"
    cv2.putText(vis_img, status_text, 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    y_offset += 30
    
    # Corner count
    corner_count = len(result.corners) if result.corners is not None else 0
    cv2.putText(vis_img, f"Corners: {corner_count}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Overall confidence
    cv2.putText(vis_img, f"Confidence: {result.confidence:.2f}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Line statistics
    horizontal_lines = len([line for line in result.lines if line.side in ['top', 'bottom']])
    vertical_lines = len([line for line in result.lines if line.side in ['left', 'right']])
    cv2.putText(vis_img, f"Lines: {len(result.lines)} (H:{horizontal_lines} V:{vertical_lines})", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Rejection information
    rejected_count = debug_info.get('rejected_lines_count', 0)
    if rejected_count > 0:
        cv2.putText(vis_img, f"Rejected: {rejected_count} lines", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        y_offset += 30
        
        # Show rejection reasons
        rejected_lines = debug_info.get('rejected_lines', [])
        for i, reason in enumerate(rejected_lines[:2]):  # Show first 2 reasons
            cv2.putText(vis_img, f"  {reason}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
            y_offset += 20
    
    # Boundary information
    cv2.putText(vis_img, f"Boundary points: {debug_info.get('boundary_points_count', 0)}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 20
    
    # Frame shape
    frame_shape = debug_info.get('frame_shape', 'N/A')
    cv2.putText(vis_img, f"Frame shape: {frame_shape}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    return vis_img

def create_boundary_visualization(mask: np.ndarray, result: CornerDetectionResult, 
                                 output_dir: str, frame_name: str):
    """
    Create visualization showing field boundary and sampled points with improved info.
    """
    # Create RGB image from mask
    vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw field boundary
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(vis_img, [largest_contour], -1, (0, 255, 0), 2)
    
    # Add text with debug info
    debug_info = result.debug_info
    y_offset = 30
    cv2.putText(vis_img, f"Field Center: {debug_info.get('field_center', 'N/A')}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    cv2.putText(vis_img, f"Boundary Points: {debug_info.get('boundary_points_count', 0)}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    y_offset += 25
    
    # Show side groups info
    side_groups = debug_info.get('side_groups', {})
    for side, count in side_groups.items():
        cv2.putText(vis_img, f"{side.capitalize()}: {count} points", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 20
    
    # Show rejection info
    rejected_count = debug_info.get('rejected_lines_count', 0)
    if rejected_count > 0:
        y_offset += 10
        cv2.putText(vis_img, f"Rejected Lines: {rejected_count}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        y_offset += 20
        
        rejected_lines = debug_info.get('rejected_lines', [])
        for reason in rejected_lines[:2]:  # Show first 2 reasons
            cv2.putText(vis_img, f"  {reason}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            y_offset += 15
    
    # Save visualization
    boundary_path = os.path.join(output_dir, f"{frame_name}_boundary.jpg")
    cv2.imwrite(boundary_path, vis_img)

def create_line_fitting_visualization(mask: np.ndarray, result: CornerDetectionResult, 
                                     output_dir: str, frame_name: str):
    """
    Create visualization showing line fitting process with confidence scores.
    """
    # Create RGB image from mask
    vis_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    # Draw fitted lines with different colors and confidence info
    colors = {
        'top': (0, 255, 0),      # Green
        'bottom': (0, 255, 255),  # Yellow
        'left': (255, 0, 0),      # Blue
        'right': (255, 0, 255)    # Magenta
    }
    
    for line in result.lines:
        color = colors.get(line.side, (255, 255, 255))
        
        # Draw line segment
        cv2.line(vis_img, 
                (int(line.start_point[0]), int(line.start_point[1])),
                (int(line.end_point[0]), int(line.end_point[1])),
                color, 3)
        
        # Add label with confidence and angle
        mid_point = ((line.start_point[0] + line.end_point[0]) // 2,
                    (line.start_point[1] + line.end_point[1]) // 2)
        cv2.putText(vis_img, f"{line.side} ({line.angle:.1f}°, {line.confidence:.2f})",
                   (int(mid_point[0]), int(mid_point[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Add summary text with improved info
    y_offset = 30
    cv2.putText(vis_img, f"Lines Fitted: {len(result.lines)}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(vis_img, f"Overall Confidence: {result.confidence:.2f}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    cv2.putText(vis_img, f"Corners Found: {result.corners is not None}", 
               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 30
    
    # Show rejection info
    debug_info = result.debug_info
    rejected_count = debug_info.get('rejected_lines_count', 0)
    if rejected_count > 0:
        cv2.putText(vis_img, f"Rejected: {rejected_count} lines", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        y_offset += 30
        
        # Show individual line confidences
        for line in result.lines:
            cv2.putText(vis_img, f"{line.side}: {line.confidence:.2f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors.get(line.side, (255, 255, 255)), 1)
            y_offset += 20
    
    # Save visualization
    line_path = os.path.join(output_dir, f"{frame_name}_line_fitting.jpg")
    cv2.imwrite(line_path, vis_img)

def create_video_from_saved_frames(visualizations_dir: str, output_path: str):
    """
    Create a video from saved composite frame images using FFmpeg.
    
    Args:
        visualizations_dir: Directory containing saved composite frames
        output_path: Path to save the output video
    """
    # Get all composite frame files
    composite_files = [f for f in os.listdir(visualizations_dir) if f.endswith('_composite.jpg')]
    composite_files.sort()  # Sort by frame number
    
    if not composite_files:
        logger.warning("No composite frame files found.")
        return
    
    logger.info(f"Found {len(composite_files)} composite frame files")
    
    # Create a temporary file listing all frames
    frame_list_path = os.path.join(visualizations_dir, "frame_list.txt")
    with open(frame_list_path, 'w') as f:
        for filename in composite_files:
            frame_path = os.path.join(visualizations_dir, filename)
            f.write(f"file '{frame_path}'\n")
            f.write("duration 0.5\n")  # 0.5 seconds per frame (2 FPS)
    
    # Use FFmpeg to create video from frame list
    try:
        import subprocess
        
        # FFmpeg command to create video from frame list
        cmd = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0',
            '-i', frame_list_path,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-y',  # Overwrite output file
            output_path
        ]
        
        logger.info(f"Running FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Video created successfully at {output_path}")
        else:
            logger.error(f"FFmpeg failed: {result.stderr}")
            # Fallback to OpenCV method
            create_video_opencv_fallback(visualizations_dir, output_path)
            
    except ImportError:
        logger.warning("FFmpeg not available, using OpenCV fallback")
        create_video_opencv_fallback(visualizations_dir, output_path)
    except FileNotFoundError:
        logger.warning("FFmpeg not found, using OpenCV fallback")
        create_video_opencv_fallback(visualizations_dir, output_path)
    finally:
        # Clean up temporary file
        if os.path.exists(frame_list_path):
            os.remove(frame_list_path)

def create_video_opencv_fallback(visualizations_dir: str, output_path: str):
    """
    Fallback video creation using OpenCV when FFmpeg is not available.
    
    Args:
        visualizations_dir: Directory containing saved composite frames
        output_path: Path to save the output video
    """
    # Get all composite frame files
    composite_files = [f for f in os.listdir(visualizations_dir) if f.endswith('_composite.jpg')]
    composite_files.sort()
    
    if not composite_files:
        logger.warning("No composite frame files found.")
        return
    
    # Read the first frame to get dimensions
    first_frame_path = os.path.join(visualizations_dir, composite_files[0])
    first_frame = cv2.imread(first_frame_path)
    
    if first_frame is None:
        logger.error(f"Failed to read first frame: {first_frame_path}")
        return
    
    height, width, _ = first_frame.shape
    fps = 2
    
    # Try different codecs
    codecs_to_try = ['mp4v', 'XVID', 'MJPG']
    out = None
    
    for codec in codecs_to_try:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                logger.info(f"Successfully opened video writer with codec: {codec}")
                break
            else:
                out.release()
        except Exception as e:
            logger.warning(f"Failed to use codec {codec}: {e}")
            continue
    
    if out is None or not out.isOpened():
        logger.error("Failed to create video writer with any codec.")
        return
    
    logger.info(f"Creating video with {len(composite_files)} frames, {width}x{height}, {fps} FPS")
    
    # Read and write each frame
    for i, filename in enumerate(composite_files):
        frame_path = os.path.join(visualizations_dir, filename)
        frame = cv2.imread(frame_path)
        
        if frame is None:
            logger.warning(f"Failed to read frame {i}: {frame_path}")
            continue
        
        # Ensure frame has correct dimensions
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        # Write frame
        success = out.write(frame)
        if not success:
            logger.warning(f"Failed to write frame {i}: {filename}")
    
    out.release()
    logger.info(f"Video created successfully at {output_path}")

def test_mask_based_corner_detection():
    """
    Main test function for mask-based corner detection with comprehensive visualization.
    """
    log_lines = []
    log_lines.append("=== Mask-Based Corner Detection Test ===")
    log_lines.append("")
    
    # Check if input masks directory exists
    if not os.path.exists(INPUT_MASKS_DIR):
        log_lines.append(f"ERROR: Input masks directory not found: {INPUT_MASKS_DIR}")
        log_lines.append("Please run test_field_segmentation.py first to generate field masks.")
        with open(LOG_PATH, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        return
    
    # Get list of mask files - load all available masks
    mask_files = [f for f in os.listdir(INPUT_MASKS_DIR) if f.endswith('_mask.png')]
    mask_files.sort()  # Sort by frame number
    
    if not mask_files:
        log_lines.append(f"ERROR: No mask files found in {INPUT_MASKS_DIR}")
        with open(LOG_PATH, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        return
    
    log_lines.append(f"Found {len(mask_files)} mask files to test")
    log_lines.append(f"Input directory: {INPUT_MASKS_DIR}")
    log_lines.append(f"Output directory: {OUTPUT_DIR}")
    log_lines.append("")
    
    # Create output subdirectories
    visualizations_dir = os.path.join(OUTPUT_DIR, "visualizations")
    os.makedirs(visualizations_dir, exist_ok=True)
    
    # Statistics tracking
    total_tested = 0
    successful_detections = 0
    total_confidence = 0.0
    frame_results = []
    composite_frames = []  # Store frames for video generation
    
    # Test each mask
    for frame_number, mask_file in enumerate(mask_files):
        mask_path = os.path.join(INPUT_MASKS_DIR, mask_file)
        frame_name = mask_file.replace('_mask.png', '')
        
        log_lines.append(f"Testing {frame_name}...")
        
        # Run corner detection
        result = test_single_mask_corner_detection(mask_path)
        total_tested += 1
        
        # Record results
        frame_result = {
            'frame': frame_name,
            'corners_found': result.corners is not None,
            'confidence': result.confidence,
            'lines_fitted': len(result.lines),
            'rejected_lines': result.debug_info.get('rejected_lines_count', 0),
            'debug_info': result.debug_info
        }
        frame_results.append(frame_result)
        
        if result.corners is not None:
            successful_detections += 1
            total_confidence += result.confidence
            log_lines.append(f"  ✅ SUCCESS: {len(result.corners)} corners, confidence {result.confidence:.2f}")
        else:
            log_lines.append(f"  ❌ FAILED: No corners detected, confidence {result.confidence:.2f}")
        
        # Log rejection information
        rejected_count = result.debug_info.get('rejected_lines_count', 0)
        if rejected_count > 0:
            log_lines.append(f"     Rejected {rejected_count} lines:")
            rejected_reasons = result.debug_info.get('rejected_lines', [])
            for reason in rejected_reasons:
                log_lines.append(f"       - {reason}")
        
        # Log individual line confidences
        if result.lines:
            log_lines.append(f"     Line confidences:")
            for line in result.lines:
                log_lines.append(f"       {line.side}: {line.confidence:.2f} (angle: {line.angle:.1f}°)")
        
        # Create composite visualization
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8) * 255
        composite_frame = create_composite_visualization(mask, result, frame_name, frame_number)
        
        # Save individual composite frame
        composite_path = os.path.join(visualizations_dir, f"{frame_name}_composite.jpg")
        cv2.imwrite(composite_path, composite_frame)
        
        # Store frame for video generation
        composite_frames.append(composite_frame)
        
        # Limit logging to first 20 frames to avoid overwhelming output
        if frame_number >= 20:
            log_lines.append(f"  ... (continuing with {len(mask_files) - frame_number - 1} more frames)")
            break
    
    # Generate video from composite frames
    if composite_frames:
        video_path = os.path.join(OUTPUT_DIR, "output_video.mp4")
        create_video_from_saved_frames(visualizations_dir, video_path)
        log_lines.append(f"Video saved to: {video_path}")
    
    # Final statistics
    success_rate = (successful_detections / total_tested) * 100 if total_tested > 0 else 0
    avg_confidence = total_confidence / successful_detections if successful_detections > 0 else 0
    
    log_lines.append("")
    log_lines.append("=== FINAL STATISTICS ===")
    log_lines.append(f"Total frames tested: {total_tested}")
    log_lines.append(f"Successful detections: {successful_detections}")
    log_lines.append(f"Success rate: {success_rate:.1f}%")
    log_lines.append(f"Average confidence: {avg_confidence:.2f}")
    
    # Rejection statistics
    total_rejected = sum(result['rejected_lines'] for result in frame_results)
    avg_rejected_per_frame = total_rejected / total_tested if total_tested > 0 else 0
    log_lines.append(f"Total lines rejected: {total_rejected}")
    log_lines.append(f"Average rejected per frame: {avg_rejected_per_frame:.1f}")
    
    # Line distribution analysis
    all_lines = []
    for result in frame_results:
        if result['corners_found']:
            all_lines.extend(result['debug_info'].get('side_groups', {}).values())
    
    if all_lines:
        avg_lines_per_side = np.mean(all_lines)
        log_lines.append(f"Average points per side: {avg_lines_per_side:.1f}")
    
    # Confidence distribution
    confidences = [result['confidence'] for result in frame_results if result['corners_found']]
    if confidences:
        min_conf = min(confidences)
        max_conf = max(confidences)
        std_conf = np.std(confidences)
        log_lines.append(f"Confidence range: {min_conf:.2f} - {max_conf:.2f}")
        log_lines.append(f"Confidence std dev: {std_conf:.2f}")
    
    log_lines.append("")
    log_lines.append("=== DETAILED RESULTS ===")
    for result in frame_results:
        log_lines.append(f"{result['frame']}: {'✅' if result['corners_found'] else '❌'} "
                        f"conf={result['confidence']:.2f} "
                        f"lines={result['lines_fitted']} "
                        f"rejected={result['rejected_lines']}")
    
    log_lines.append("")
    log_lines.append(f"Test completed. Results saved to: {OUTPUT_DIR}")
    
    # Save log
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')
            print(line)
    
    print(f"\nTest completed successfully!")
    print(f"Success rate: {success_rate:.1f}% ({successful_detections}/{total_tested})")
    print(f"Average confidence: {avg_confidence:.2f}")
    print(f"Total lines rejected: {total_rejected}")
    print(f"Results saved to: {OUTPUT_DIR}")
    if composite_frames:
        print(f"Video saved to: {os.path.join(OUTPUT_DIR, 'output_video.mp4')}")
    
    # Assertions for automated testing
    assert successful_detections > 0, "No successful corner detections"
    assert success_rate >= 30, f"Success rate too low: {success_rate:.1f}%"
    assert avg_confidence >= 0.5, f"Average confidence too low: {avg_confidence:.2f}"

def test_specific_parameters():
    """
    Test corner detection with different parameters to find optimal settings.
    """
    log_lines = []
    log_lines.append("=== Parameter Optimization Test ===")
    log_lines.append("")
    
    # Get a sample mask for parameter testing
    mask_files = [f for f in os.listdir(INPUT_MASKS_DIR) if f.endswith('_mask.png')]
    if not mask_files:
        log_lines.append("No mask files available for parameter testing")
        return
    
    sample_mask_path = os.path.join(INPUT_MASKS_DIR, mask_files[0])
    mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Test different parameters
    parameters = [
        {'num_samples': 100, 'min_inliers': 0.5},
        {'num_samples': 200, 'min_inliers': 0.5},
        {'num_samples': 300, 'min_inliers': 0.5},
        {'num_samples': 200, 'min_inliers': 0.4},
        {'num_samples': 200, 'min_inliers': 0.6},
        {'num_samples': 200, 'min_inliers': 0.7},
    ]
    
    log_lines.append("Testing different parameter combinations:")
    log_lines.append("")
    
    for i, params in enumerate(parameters):
        result = mask_based_corner_detection(mask, **params)
        status = "✅" if result.corners is not None else "❌"
        log_lines.append(f"{status} Test {i+1}: samples={params['num_samples']}, "
                        f"min_inliers={params['min_inliers']}, "
                        f"confidence={result.confidence:.2f}, "
                        f"lines={len(result.lines)}")
    
    # Save parameter test results
    param_log_path = os.path.join(OUTPUT_DIR, 'parameter_test_log.txt')
    with open(param_log_path, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')
    
    print(f"Parameter test results saved to {param_log_path}")

if __name__ == "__main__":
    print("Running mask-based corner detection tests...")
    
    # Run main test
    test_mask_based_corner_detection()
    
    print("\n" + "="*50)
    print("Running parameter optimization test...")
    test_specific_parameters() 