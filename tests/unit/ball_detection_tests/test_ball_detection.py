#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import numpy as np
import supervision as sv
from src.vision.detector import VisionDetector

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'ball_detection_log.txt')

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

def test_custom_ball_detection():
    """Test ball detection using the custom trained model on video frames."""
    log_lines = []
    log_lines.append("=== Custom Ball Detection Test ===")
    log_lines.append(f"Using custom trained model: runs/detect/model2/weights/best.pt")
    
    # Video path
    video_path = "data/pipelineV1/ROMVER.mp4"
    log_lines.append(f"Processing video: {video_path}")
    
    # Initialize detector with custom ball model
    detector = VisionDetector(sport="soccer")
    log_lines.append(f"Initialized VisionDetector with custom ball model")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_lines.append(f"ERROR: Could not open video {video_path}")
        with open(LOG_PATH, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log_lines.append(f"Video properties: {total_frames} frames, {original_fps} fps, {width}x{height}")
    
    # Process frames (sample every 5th frame to keep test fast)
    frame_skip = 5
    processed_frames = 0
    total_detections = 0
    frame_detection_summary = []
    
    log_lines.append(f"Processing every {frame_skip}th frame...")
    
    # Setup output video with adjusted FPS to maintain real-time playback
    output_video_path = os.path.join(OUTPUT_DIR, "ball_detection_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Adjust output FPS to maintain real-time playback: original_fps / frame_skip
    output_fps = original_fps / frame_skip
    out = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
    
    log_lines.append(f"Output video FPS: {output_fps:.2f} (adjusted for frame skipping)")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every nth frame
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
            
        # Run ball detection using custom model
        try:
            ball_results = detector.ball_model(frame)
            ball_detections = sv.Detections.from_ultralytics(ball_results[0])
            
            # Filter by confidence threshold
            ball_mask = (ball_detections.confidence > detector.ball_confidence_threshold)
            ball_detections = ball_detections[ball_mask]
            
            # Extract bounding boxes and confidences
            bboxes = ball_detections.xyxy.tolist() if len(ball_detections) > 0 else []
            confidences = ball_detections.confidence.tolist() if len(ball_detections) > 0 else []
            
            # Log frame results
            frame_summary = f"Frame {frame_idx}: {len(bboxes)} balls detected"
            if bboxes:
                for i, (bbox, conf) in enumerate(zip(bboxes, confidences)):
                    center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                    frame_summary += f" | Ball {i+1}: center=({center_x},{center_y}), conf={conf:.3f}"
            log_lines.append(frame_summary)
            frame_detection_summary.append({
                'frame': frame_idx,
                'num_detections': len(bboxes),
                'bboxes': bboxes,
                'confidences': confidences
            })
            
            total_detections += len(bboxes)
            
            # Annotate frame
            annotated_frame = frame.copy()
            annotated_frame = draw_ball_boxes(annotated_frame, bboxes, confidences)
            
            # Add frame info
            cv2.putText(annotated_frame, f"Frame: {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Balls: {len(bboxes)}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Write to output video
            out.write(annotated_frame)
            processed_frames += 1
            
        except Exception as e:
            log_lines.append(f"ERROR in frame {frame_idx}: {str(e)}")
        
        frame_idx += 1
        
        # Limit to first 50 processed frames for quick testing
        if processed_frames >= 50:
            break
    
    # Cleanup
    cap.release()
    out.release()
    
    # Summary statistics
    log_lines.append("\n=== Detection Summary ===")
    log_lines.append(f"Total frames processed: {processed_frames}")
    log_lines.append(f"Total ball detections: {total_detections}")
    log_lines.append(f"Average detections per frame: {total_detections/processed_frames:.2f}")
    
    # Find frames with detections
    frames_with_detections = [f for f in frame_detection_summary if f['num_detections'] > 0]
    log_lines.append(f"Frames with detections: {len(frames_with_detections)}/{processed_frames}")
    
    if frames_with_detections:
        log_lines.append("\n=== Frames with Ball Detections ===")
        for frame_info in frames_with_detections[:10]:  # Show first 10
            log_lines.append(f"Frame {frame_info['frame']}: {frame_info['num_detections']} balls")
            for i, (bbox, conf) in enumerate(zip(frame_info['bboxes'], frame_info['confidences'])):
                center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
                log_lines.append(f"  Ball {i+1}: center=({center_x},{center_y}), conf={conf:.3f}")
    
    log_lines.append(f"\nAnnotated video saved to: {output_video_path}")
    log_lines.append("Custom ball detection test completed successfully!")
    
    # Save detailed log
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            print(line)
            f.write(line + '\n')
    
    print(f"\nLog saved to {LOG_PATH}")
    print(f"Annotated video saved to {output_video_path}")
    
    # Basic assertion
    assert total_detections > 0, "No ball detections found in video"
    assert processed_frames > 0, "No frames were processed"

if __name__ == "__main__":
    test_custom_ball_detection() 