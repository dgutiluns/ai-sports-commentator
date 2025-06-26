#!/usr/bin/env python3
"""
Side-by-side comparison test for custom ball detector vs Roboflow ball detector.
This test processes video frames and compares detection results from both models.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import numpy as np
import supervision as sv
from src.vision.detector import VisionDetector, RoboflowBallDetector

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'ball_comparison_log.txt')

VIDEO_PATH = "data/pipelineV1/ROMVER.mp4"
CUSTOM_MODEL_PATH = "runs/detect/model2/weights/best.pt"
ROBOFLOW_MODEL_ID = "football-ball-detection-rejhg/4"

def draw_ball_boxes(frame, bboxes, confidences, color=(0,255,255), label_prefix="Ball"):
    """Draw bounding boxes around detected balls with confidence scores."""
    for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
        if len(bbox) >= 4:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_prefix}_{i+1}: {conf:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            # Draw center point
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
    return frame

def create_side_by_side_frame(frame, custom_bboxes, custom_confs, roboflow_bboxes, roboflow_confs):
    """Create a side-by-side comparison frame."""
    h, w = frame.shape[:2]
    combined_frame = np.zeros((h, w * 2, 3), dtype=np.uint8)
    # Left: Custom detector (green)
    left_frame = frame.copy()
    left_frame = draw_ball_boxes(left_frame, custom_bboxes, custom_confs, color=(0, 255, 0), label_prefix="Custom")
    cv2.putText(left_frame, "Custom Ball Detector", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(left_frame, f"Balls: {len(custom_bboxes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Right: Roboflow detector (red)
    right_frame = frame.copy()
    right_frame = draw_ball_boxes(right_frame, roboflow_bboxes, roboflow_confs, color=(0, 0, 255), label_prefix="RF")
    cv2.putText(right_frame, "Roboflow Ball Detector", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(right_frame, f"Balls: {len(roboflow_bboxes)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Combine
    combined_frame[:, :w] = left_frame
    combined_frame[:, w:] = right_frame
    cv2.line(combined_frame, (w, 0), (w, h), (255, 255, 255), 2)
    return combined_frame

def test_ball_detector_comparison():
    log_lines = []
    log_lines.append("=== Ball Detector Comparison Test ===")
    log_lines.append(f"Custom Model: {CUSTOM_MODEL_PATH}")
    log_lines.append(f"Roboflow Model: {ROBOFLOW_MODEL_ID}")
    log_lines.append("")
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log_lines.append(f"ERROR: Could not open video {VIDEO_PATH}")
        with open(LOG_PATH, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        return
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log_lines.append(f"Video properties: {total_frames} frames, {original_fps} fps, {width}x{height}")
    # Detectors
    custom_detector = VisionDetector(sport="soccer")
    roboflow_detector = RoboflowBallDetector(model_id=ROBOFLOW_MODEL_ID)
    # Output video
    output_video_path = os.path.join(OUTPUT_DIR, "ball_comparison.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (width * 2, height))
    log_lines.append(f"Output video FPS: {original_fps:.2f} (real-time)")
    log_lines.append(f"Output video size: {width * 2}x{height} (side-by-side)")
    # Process all frames
    processed_frames = 0
    custom_total_detections = 0
    roboflow_total_detections = 0
    frame_comparison_summary = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Custom ball detection
        try:
            ball_results = custom_detector.ball_model(frame)
            ball_detections = sv.Detections.from_ultralytics(ball_results[0])
            ball_mask = (ball_detections.confidence > custom_detector.ball_confidence_threshold)
            ball_detections = ball_detections[ball_mask]
            custom_bboxes = ball_detections.xyxy.tolist() if len(ball_detections) > 0 else []
            custom_confs = ball_detections.confidence.tolist() if len(ball_detections) > 0 else []
        except Exception as e:
            log_lines.append(f"ERROR in custom detection frame {frame_idx}: {str(e)}")
            custom_bboxes, custom_confs = [], []
        # Roboflow ball detection
        try:
            roboflow_bboxes = roboflow_detector.detect_balls(frame)
            if roboflow_bboxes:
                roboflow_confs = [bbox[4] for bbox in roboflow_bboxes]
                roboflow_bboxes = [bbox[:4] for bbox in roboflow_bboxes]
            else:
                roboflow_bboxes, roboflow_confs = [], []
        except Exception as e:
            log_lines.append(f"ERROR in Roboflow detection frame {frame_idx}: {str(e)}")
            roboflow_bboxes, roboflow_confs = [], []
        # Log frame results
        frame_summary = f"Frame {frame_idx}: Custom={len(custom_bboxes)} balls | Roboflow={len(roboflow_bboxes)} balls"
        if custom_bboxes:
            for i, (bbox, conf) in enumerate(zip(custom_bboxes, custom_confs)):
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                frame_summary += f" | Custom_Ball_{i+1}: center=({center_x},{center_y}), conf={conf:.3f}"
        if roboflow_bboxes:
            for i, (bbox, conf) in enumerate(zip(roboflow_bboxes, roboflow_confs)):
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                frame_summary += f" | RF_Ball_{i+1}: center=({center_x},{center_y}), conf={conf:.3f}"
        log_lines.append(frame_summary)
        frame_comparison_summary.append({
            'frame': frame_idx,
            'custom_detections': len(custom_bboxes),
            'roboflow_detections': len(roboflow_bboxes),
            'custom_bboxes': custom_bboxes,
            'custom_confs': custom_confs,
            'roboflow_bboxes': roboflow_bboxes,
            'roboflow_confs': roboflow_confs
        })
        custom_total_detections += len(custom_bboxes)
        roboflow_total_detections += len(roboflow_bboxes)
        # Create side-by-side comparison frame
        combined_frame = create_side_by_side_frame(
            frame, custom_bboxes, custom_confs, roboflow_bboxes, roboflow_confs
        )
        cv2.putText(combined_frame, f"Frame: {frame_idx}", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(combined_frame)
        processed_frames += 1
        frame_idx += 1
    # Cleanup
    cap.release()
    out.release()
    # Summary statistics
    log_lines.append("\n=== Detection Comparison Summary ===")
    log_lines.append(f"Total frames processed: {processed_frames}")
    log_lines.append(f"Custom detector total detections: {custom_total_detections}")
    log_lines.append(f"Roboflow detector total detections: {roboflow_total_detections}")
    log_lines.append(f"Custom average detections per frame: {custom_total_detections/processed_frames:.2f}")
    log_lines.append(f"Roboflow average detections per frame: {roboflow_total_detections/processed_frames:.2f}")
    frames_with_custom = [f for f in frame_comparison_summary if f['custom_detections'] > 0]
    frames_with_roboflow = [f for f in frame_comparison_summary if f['roboflow_detections'] > 0]
    log_lines.append(f"Frames with custom detections: {len(frames_with_custom)}/{processed_frames}")
    log_lines.append(f"Frames with Roboflow detections: {len(frames_with_roboflow)}/{processed_frames}")
    log_lines.append("\n=== Detailed Frame Comparison (First 10 with detections) ===")
    frames_with_any_detections = [f for f in frame_comparison_summary if f['custom_detections'] > 0 or f['roboflow_detections'] > 0]
    for frame_info in frames_with_any_detections[:10]:
        log_lines.append(f"Frame {frame_info['frame']}:")
        if frame_info['custom_detections'] > 0:
            log_lines.append(f"  Custom: {frame_info['custom_detections']} balls")
            for i, (bbox, conf) in enumerate(zip(frame_info['custom_bboxes'], frame_info['custom_confs'])):
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                log_lines.append(f"    Ball {i+1}: center=({center_x},{center_y}), conf={conf:.3f}")
        if frame_info['roboflow_detections'] > 0:
            log_lines.append(f"  Roboflow: {frame_info['roboflow_detections']} balls")
            for i, (bbox, conf) in enumerate(zip(frame_info['roboflow_bboxes'], frame_info['roboflow_confs'])):
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                log_lines.append(f"    Ball {i+1}: center=({center_x},{center_y}), conf={conf:.3f}")
    log_lines.append(f"\nSide-by-side comparison video saved to: {output_video_path}")
    log_lines.append("Ball detector comparison test completed successfully!")
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            print(line)
            f.write(line + '\n')
    print(f"\nLog saved to {LOG_PATH}")
    print(f"Comparison video saved to {output_video_path}")
    assert processed_frames > 0, "No frames were processed"
    print(f"\nComparison Summary:")
    print(f"  Custom detector: {custom_total_detections} total detections")
    print(f"  Roboflow detector: {roboflow_total_detections} total detections")

if __name__ == "__main__":
    test_ball_detector_comparison() 