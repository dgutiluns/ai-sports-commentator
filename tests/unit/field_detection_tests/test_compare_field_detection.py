#!/usr/bin/env python3
"""
Side-by-side comparison test for current field segmentation vs Roboflow field segmentation.
This test processes video frames and compares field detection results from both approaches.
"""

import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import numpy as np
import json
from pathlib import Path
from src.vision.field_segmentation import segment_field, get_field_corners_from_mask
from inference_sdk import InferenceHTTPClient
from src.vision.detector import RoboflowFieldDetector

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'field_detection_comparison_log.txt')

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="rF2UaYsyVRscRHH14q4u"
)

# Update Roboflow model ID for segmentation
ROBOFLOW_MODEL_ID = "football-field-detection-f07vi/15"

# Initialize RoboflowFieldDetector with the correct model ID
roboflow_field_detector = RoboflowFieldDetector(model_id=ROBOFLOW_MODEL_ID)

def process_roboflow_result(result, image_shape, debug=False):
    """
    Process Roboflow inference result to extract field mask.
    Returns a binary mask and coverage percentage.
    """
    height, width = image_shape
    field_mask = np.zeros((height, width), dtype=np.uint8)
    if debug:
        print("\n[DEBUG] Roboflow result (first frame):\n", result)
    try:
        # Roboflow segmentation models typically return a 'predictions' list with 'mask' (RLE or array)
        if result and 'predictions' in result:
            for pred in result['predictions']:
                if 'class' in pred and pred['class'].lower() in ['field', 'pitch', 'football field', 'soccer field']:
                    if 'mask' in pred:
                        mask_data = pred['mask']
                        # Try pycocotools RLE decode first
                        try:
                            from pycocotools import mask as maskUtils
                            mask = maskUtils.decode(mask_data)
                            if mask.shape != (height, width):
                                mask = mask.squeeze()
                            field_mask = np.logical_or(field_mask, mask)
                        except Exception as e:
                            print(f"[DEBUG] pycocotools decode failed: {e}")
                            # Fallback: try to decode as array
                            if isinstance(mask_data, dict) and 'data' in mask_data:
                                arr = np.array(mask_data['data']).reshape((height, width))
                                field_mask = np.logical_or(field_mask, arr)
                    elif 'data' in pred:
                        arr = np.array(pred['data']).reshape((height, width))
                        field_mask = np.logical_or(field_mask, arr)
        # Fallback: try 'segmentation' key
        elif result and 'segmentation' in result:
            seg = result['segmentation']
            if isinstance(seg, list):
                for segment in seg:
                    if 'mask' in segment:
                        mask_data = segment['mask']
                        arr = np.array(mask_data['data']).reshape((height, width))
                        field_mask = np.logical_or(field_mask, arr)
        field_mask = (field_mask > 0).astype(np.uint8)
        field_pixels = np.sum(field_mask)
        field_percentage = (field_pixels / (width * height)) * 100
        return field_mask, field_percentage
    except Exception as e:
        print(f"[ERROR] Exception in process_roboflow_result: {e}")
        return np.zeros((height, width), dtype=np.uint8), 0.0

def create_comparison_frame(frame, current_mask, roboflow_mask, current_percentage, roboflow_percentage, roboflow_contours=None):
    """
    Create a side-by-side comparison frame.
    
    Args:
        frame: Original video frame
        current_mask: Field mask from current method
        roboflow_mask: Field mask from Roboflow
        current_percentage: Field coverage from current method
        roboflow_percentage: Field coverage from Roboflow
        roboflow_contours: Contours from Roboflow
    
    Returns:
        comparison_frame: Side-by-side comparison image
    """
    height, width = frame.shape[:2]
    if roboflow_contours is None:
        roboflow_contours = []
    # Defensive: ensure masks are valid numpy arrays with correct shape
    if not isinstance(current_mask, np.ndarray) or current_mask.shape != (height, width):
        current_mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(roboflow_mask, np.ndarray) or roboflow_mask.shape != (height, width):
        roboflow_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Ensure masks are binary (0 or 1)
    current_mask = (current_mask > 0).astype(np.uint8)
    roboflow_mask = (roboflow_mask > 0).astype(np.uint8)
    
    # Create comparison frame (side by side)
    comparison_width = width * 2
    comparison_frame = np.zeros((height, comparison_width, 3), dtype=np.uint8)
    
    # Left side: Current method
    left_frame = frame.copy()
    if np.any(current_mask > 0):
        mask_indices = current_mask > 0
        left_frame[mask_indices] = cv2.addWeighted(
            left_frame[mask_indices], 0.7,
            np.full_like(left_frame[mask_indices], [0, 255, 0]), 0.3, 0
        )
    
    # Add text overlay for current method
    cv2.putText(left_frame, f"Current Method: {current_percentage:.1f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(left_frame, "Field Coverage", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Right side: Roboflow method
    right_frame = frame.copy()
    if np.any(roboflow_mask > 0):
        mask_indices = roboflow_mask > 0
        right_frame[mask_indices] = cv2.addWeighted(
            right_frame[mask_indices], 0.7,
            np.full_like(right_frame[mask_indices], [255, 0, 0]), 0.3, 0
        )
    
    # Draw Roboflow contours in blue
    if roboflow_contours:
        cv2.drawContours(right_frame, roboflow_contours, -1, (0, 0, 255), 2)
    
    # Add text overlay for Roboflow method
    cv2.putText(right_frame, f"Roboflow Model: {roboflow_percentage:.1f}%", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(right_frame, "Field Coverage", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Combine frames side by side
    comparison_frame[:, :width] = left_frame
    comparison_frame[:, width:] = right_frame
    
    # Add separator line
    cv2.line(comparison_frame, (width, 0), (width, height), (255, 255, 255), 2)
    
    # Add title
    cv2.putText(comparison_frame, "Field Detection Comparison", 
                (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return comparison_frame

def test_compare_field_detection():
    """Compare current field segmentation with Roboflow field segmentation."""
    print("Starting field detection comparison test...")
    
    # Initialize logging
    log_lines = []
    log_lines.append("=== Field Detection Comparison Test ===")
    log_lines.append(f"Video: data/pipelineV1/ROMVER.mp4")
    log_lines.append("=" * 50)
    
    # Create output directories
    comparison_dir = os.path.join(OUTPUT_DIR, 'comparison_frames')
    masks_dir = os.path.join(OUTPUT_DIR, 'field_masks')
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Open video
    video_path = "data/pipelineV1/ROMVER.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        log_lines.append("ERROR: Could not open video file")
        with open(LOG_PATH, 'w') as f:
            f.write('\n'.join(log_lines))
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log_lines.append(f"Video Properties:")
    log_lines.append(f"  - Total frames: {total_frames}")
    log_lines.append(f"  - FPS: {fps}")
    log_lines.append(f"  - Resolution: {width}x{height}")
    log_lines.append("")
    
    # Initialize video writer for comparison video
    comparison_video_path = os.path.join(OUTPUT_DIR, 'field_comparison.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    comparison_writer = cv2.VideoWriter(
        comparison_video_path, fourcc, fps, (width * 2, height)
    )
    
    # Process every 10th frame for efficiency
    sample_interval = 10
    processed_frames = 0
    
    frame_results = []
    
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        processed_frames += 1
        log_lines.append(f"Processing frame {frame_idx}...")

        # ==== Current Field Segmentation Approach ====
        try:
            current_mask = segment_field(frame)
            current_pixels = np.sum(current_mask > 0)
            current_percentage = (current_pixels / (width * height)) * 100
            log_lines.append(f"  - Current method: {current_pixels:,} pixels ({current_percentage:.1f}%)")
            current_corners = None
            try:
                corners_result = get_field_corners_from_mask(current_mask)
                if corners_result is not None:
                    current_corners, fallback_used = corners_result
                    if current_corners is not None:
                        log_lines.append(f"    - Corners detected: {len(current_corners)} points")
                        if fallback_used:
                            log_lines.append(f"    - Used fallback rectangle")
                        else:
                            log_lines.append(f"    - Used polygon approximation")
            except Exception as e:
                log_lines.append(f"    - Corner detection failed: {str(e)}")
        except Exception as e:
            log_lines.append(f"  - ERROR in current method: {str(e)}")
            current_mask = np.zeros((height, width), dtype=np.uint8)
            current_percentage = 0.0
            current_corners = None

        # ==== Roboflow Field Segmentation Approach ====
        try:
            roboflow_mask, roboflow_contours = roboflow_field_detector.detect_field(frame)
            roboflow_pixels = int(np.sum(roboflow_mask > 0))
            roboflow_percentage = (roboflow_pixels / (width * height)) * 100
            log_lines.append(f"  - Roboflow method: {roboflow_pixels:,} pixels ({roboflow_percentage:.1f}%)")
            roboflow_corners = None
            try:
                corners_result = get_field_corners_from_mask(roboflow_mask)
                if corners_result is not None:
                    roboflow_corners, fallback_used = corners_result
                    if roboflow_corners is not None:
                        log_lines.append(f"    - Corners detected: {len(roboflow_corners)} points")
                        if fallback_used:
                            log_lines.append(f"    - Used fallback rectangle")
                        else:
                            log_lines.append(f"    - Used polygon approximation")
            except Exception as e:
                log_lines.append(f"    - Corner detection failed: {str(e)}")
        except Exception as e:
            log_lines.append(f"  - ERROR in Roboflow method: {str(e)}")
            roboflow_mask = np.zeros((height, width), dtype=np.uint8)
            roboflow_percentage = 0.0
            roboflow_corners = None
            roboflow_pixels = 0
            roboflow_contours = []

        # --- Visualization ---
        comparison_frame = create_comparison_frame(
            frame, current_mask, roboflow_mask, 
            current_percentage, roboflow_percentage,
            roboflow_contours=roboflow_contours
        )
        
        # Save comparison frame
        comparison_frame_path = os.path.join(comparison_dir, f"comparison_frame_{frame_idx:04d}.jpg")
        cv2.imwrite(comparison_frame_path, comparison_frame)
        
        # Add frame to video
        comparison_writer.write(comparison_frame)
        
        # Save individual masks
        current_mask_path = os.path.join(masks_dir, f"current_mask_{frame_idx:04d}.png")
        roboflow_mask_path = os.path.join(masks_dir, f"roboflow_mask_{frame_idx:04d}.png")
        cv2.imwrite(current_mask_path, current_mask * 255)
        cv2.imwrite(roboflow_mask_path, roboflow_mask * 255)
        
        # Store frame results
        frame_result = {
            "frame": frame_idx,
            "current_method": {
                "pixels": int(current_pixels),
                "percentage": float(current_percentage),
                "corners_detected": current_corners is not None,
                "corner_count": len(current_corners) if current_corners is not None else 0
            },
            "roboflow_method": {
                "pixels": int(roboflow_pixels),
                "percentage": float(roboflow_percentage),
                "corners_detected": roboflow_corners is not None,
                "corner_count": len(roboflow_corners) if roboflow_corners is not None else 0
            }
        }
        frame_results.append(frame_result)
        
        # Log comparison summary
        coverage_diff = current_percentage - roboflow_percentage
        log_lines.append(f"  - Coverage difference: {coverage_diff:+.1f}% (Current - Roboflow)")
        
        if current_corners is not None and roboflow_corners is not None:
            log_lines.append(f"  - Both methods detected corners successfully")
        elif current_corners is not None:
            log_lines.append(f"  - Only current method detected corners")
        elif roboflow_corners is not None:
            log_lines.append(f"  - Only Roboflow method detected corners")
        else:
            log_lines.append(f"  - Neither method detected corners")
        
        log_lines.append("")
    
    cap.release()
    comparison_writer.release()
    
    # Summary statistics
    log_lines.append("=== SUMMARY ===")
    log_lines.append(f"Total frames processed: {processed_frames}")
    
    if frame_results:
        # Calculate averages
        current_avg_percentage = np.mean([r["current_method"]["percentage"] for r in frame_results])
        roboflow_avg_percentage = np.mean([r["roboflow_method"]["percentage"] for r in frame_results])
        
        current_corner_success = sum(1 for r in frame_results if r["current_method"]["corners_detected"])
        roboflow_corner_success = sum(1 for r in frame_results if r["roboflow_method"]["corners_detected"])
        
        log_lines.append(f"Average field coverage:")
        log_lines.append(f"  - Current method: {current_avg_percentage:.1f}%")
        log_lines.append(f"  - Roboflow method: {roboflow_avg_percentage:.1f}%")
        log_lines.append(f"  - Difference: {current_avg_percentage - roboflow_avg_percentage:+.1f}%")
        
        log_lines.append(f"Corner detection success:")
        log_lines.append(f"  - Current method: {current_corner_success}/{processed_frames} frames ({current_corner_success/processed_frames*100:.1f}%)")
        log_lines.append(f"  - Roboflow method: {roboflow_corner_success}/{processed_frames} frames ({roboflow_corner_success/processed_frames*100:.1f}%)")
    
    log_lines.append("")
    log_lines.append("=== OUTPUT FILES ===")
    log_lines.append(f"Comparison video: {comparison_video_path}")
    log_lines.append(f"Comparison frames: {comparison_dir}")
    log_lines.append(f"Field masks: {masks_dir}")
    log_lines.append(f"Detailed log: {LOG_PATH}")
    
    # Save detailed log
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))
    
    # Save frame results as JSON
    results_path = os.path.join(OUTPUT_DIR, 'frame_results.json')
    with open(results_path, 'w') as f:
        json.dump(frame_results, f, indent=2)
    
    print(f"Field detection comparison test completed!")
    print(f"Processed {processed_frames} frames")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    test_compare_field_detection() 