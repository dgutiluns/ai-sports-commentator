#!/usr/bin/env python3
"""
AI Commentator - Main Pipeline (Roboflow+ByteTrack for Player Tracking + Custom Ball Detection)
This script runs the main pipeline using:
- Roboflow for player detection and ByteTrack for player tracking
- Custom-trained ball detector (VisionDetector) for ball detection
Field detection, event detection, and commentary are left as placeholders for future integration.
"""

import argparse
import logging
import json
import pickle
import sys
import os
from pathlib import Path
import cv2
import numpy as np
from dotenv import load_dotenv

# Add src directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.vision.detector import RoboflowPlayerDetector, VisionDetector
import supervision as sv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_environment():
    """Initialize environment variables and setup."""
    load_dotenv()
    # Add any additional environment setup here

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI Commentator - Main Pipeline (Roboflow+ByteTrack + Custom Ball)')
    parser.add_argument('--input', type=str, required=True, help='Input video file path or "camera" for live feed')
    parser.add_argument('--output', type=str, help='Output file path for annotated video (optional)')
    parser.add_argument('--data-log', type=str, help='Output file path for frame results data (JSON or pickle)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def detect_players(frame, player_detector, player_tracker):
    """
    Detect and track players using Roboflow + ByteTrack.
    
    Args:
        frame: Input video frame
        player_detector: RoboflowPlayerDetector instance
        player_tracker: ByteTrack instance
    
    Returns:
        dict: Player detection results with tracking IDs and positions
    """
    roboflow_bboxes = player_detector.detect_players(frame)
    players = []
    
    if roboflow_bboxes:
        xyxy = np.array([b[:4] for b in roboflow_bboxes])
        conf = np.array([b[4] for b in roboflow_bboxes])
        class_id = np.zeros(len(roboflow_bboxes), dtype=int)
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id
        )
        tracked = player_tracker.update_with_detections(detections)
        
        for bbox, tid in zip(tracked.xyxy, tracked.tracker_id):
            x1, y1, x2, y2 = bbox
            players.append({
                'id': int(tid),
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'x': int((x1 + x2) // 2),
                'y': int((y1 + y2) // 2),
                'img_x': int((x1 + x2) // 2),
                'img_y': int((y1 + y2) // 2),
                'team': None,  # TODO: Add team classification
                'number': None  # TODO: Add jersey number detection
            })
    
    return players, tracked

def detect_ball(frame, ball_detector):
    """
    Detect ball using custom-trained model.
    
    Args:
        frame: Input video frame
        ball_detector: VisionDetector instance with custom ball model
    
    Returns:
        list: Ball detection results with bounding boxes and confidence scores
    """
    try:
        ball_results = ball_detector.ball_model(frame)
        ball_detections = sv.Detections.from_ultralytics(ball_results[0])
        
        # Filter by confidence threshold
        ball_mask = (ball_detections.confidence > ball_detector.ball_confidence_threshold)
        ball_detections = ball_detections[ball_mask]
        
        balls = []
        if len(ball_detections) > 0:
            for bbox, conf in zip(ball_detections.xyxy, ball_detections.confidence):
                x1, y1, x2, y2 = bbox
                balls.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'x': int((x1 + x2) // 2),
                    'y': int((y1 + y2) // 2)
                })
        
        return balls
    except Exception as e:
        logger.warning(f"Ball detection error: {e}")
        return []

def detect_field(frame):
    """
    TODO: Field detection and homography calculation.
    Placeholder for future field segmentation module.
    
    Args:
        frame: Input video frame
    
    Returns:
        dict: Field detection results (placeholder)
    """
    # TODO: Implement field detection
    # - Field boundary detection
    # - Homography matrix calculation
    # - Field coordinate mapping
    return {
        'field_detected': False,
        'homography_matrix': None,
        'field_boundaries': None
    }

def detect_events(frame_results_history):
    """
    TODO: Event detection using player and ball tracking data.
    Placeholder for future event detection module.
    
    Args:
        frame_results_history: List of previous frame results
    
    Returns:
        list: Detected events (placeholder)
    """
    # TODO: Implement event detection
    # - Goal detection
    # - Pass detection
    # - Shot detection
    # - Foul detection
    # - etc.
    return []

def create_frame_results(frame_idx, players, balls, field_info=None, events=None):
    """
    Create standardized frame results data structure.
    
    Args:
        frame_idx: Current frame index
        players: Player detection results
        balls: Ball detection results
        field_info: Field detection results (optional)
        events: Event detection results (optional)
    
    Returns:
        dict: Standardized frame results
    """
    return {
        "frame": frame_idx,
        "timestamp": frame_idx / 25.0,  # Assuming 25 FPS, TODO: get actual FPS
        "players": players,
        "ball": balls if balls else None,
        "field": field_info or {"field_detected": False},
        "events": events or []
    }

def draw_annotations(frame, players, balls, frame_idx):
    """
    Draw annotations on frame for visualization.
    
    Args:
        frame: Input video frame
        players: Player detection results
        balls: Ball detection results
        frame_idx: Current frame index
    
    Returns:
        numpy.ndarray: Annotated frame
    """
    overlay = frame.copy()
    
    # Draw player boxes and tracking IDs
    COLORS = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in range(100)]
    
    for player in players:
        x, y = player['img_x'], player['img_y']
        bbox = player['bbox']
        player_id = player['id']
        color = COLORS[player_id % len(COLORS)]
        
        # Draw bounding box
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Draw center point and ID
        cv2.circle(overlay, (int(x), int(y)), 5, color, -1)
        cv2.putText(overlay, f"P{player_id}", (int(x), int(y)-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw ball detections
    for ball in balls:
        bbox = ball['bbox']
        conf = ball['confidence']
        x, y = ball['x'], ball['y']
        
        # Draw ball bounding box (green for ball)
        cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Draw ball center and confidence
        cv2.circle(overlay, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(overlay, f"Ball: {conf:.2f}", (bbox[0], bbox[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw frame info
    cv2.putText(overlay, f"Frame: {frame_idx}", (10, overlay.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(overlay, f"Players: {len(players)} | Balls: {len(balls)}", 
               (10, overlay.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay

def save_frame_results(frame_results_history, output_path):
    """
    Save frame results to file (JSON or pickle).
    
    Args:
        frame_results_history: List of frame results
        output_path: Output file path
    """
    if not output_path:
        return
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix.lower() == '.json':
        # Convert numpy arrays to lists for JSON serialization
        json_data = []
        for result in frame_results_history:
            json_result = result.copy()
            # Ensure all data is JSON serializable
            if json_result['ball']:
                for ball in json_result['ball']:
                    ball['bbox'] = [int(x) for x in ball['bbox']]
            json_data.append(json_result)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    else:
        # Use pickle for binary format
        with open(output_path, 'wb') as f:
            pickle.dump(frame_results_history, f)
    
    logger.info(f"Frame results saved to: {output_path}")

def main():
    """Main pipeline execution."""
    args = parse_arguments()
    setup_environment()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize detectors
    logger.info("Initializing detectors...")
    player_detector = RoboflowPlayerDetector()
    player_tracker = sv.ByteTrack()
    ball_detector = VisionDetector(sport="soccer")  # Uses custom ball model
    
    # Setup video capture
    if args.input.lower() == 'camera':
        cap = cv2.VideoCapture(0)
    else:
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            return
        cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        logger.error("Failed to open video source")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")

    # Setup video writer if output is specified
    out = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        logger.info(f"Output video will be saved to: {output_path}")

    # Initialize data storage
    frame_results_history = []
    frame_idx = 0

    try:
        logger.info("Starting pipeline processing...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Player detection and tracking
            players, tracked = detect_players(frame, player_detector, player_tracker)
            
            # 2. Ball detection using custom model
            balls = detect_ball(frame, ball_detector)
            
            # 3. TODO: Field detection (placeholder)
            field_info = detect_field(frame)
            
            # 4. TODO: Event detection (placeholder)
            events = detect_events(frame_results_history)
            
            # 5. Create frame results
            frame_results = create_frame_results(frame_idx, players, balls, field_info, events)
            frame_results_history.append(frame_results)
            
            # 6. Visualization
            annotated_frame = draw_annotations(frame, players, balls, frame_idx)
            
            if out:
                out.write(annotated_frame)
            if args.debug:
                cv2.imshow('AI Commentator - Main Pipeline', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Log progress
            if frame_idx % 50 == 0:
                logger.info(f"Processed frame {frame_idx}/{total_frames} - Players: {len(players)}, Balls: {len(balls)}")
            
            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Save frame results
        if args.data_log:
            save_frame_results(frame_results_history, args.data_log)
        
        logger.info(f"Pipeline completed. Processed {frame_idx} frames.")
        logger.info(f"Total detections - Players: {sum(len(fr['players']) for fr in frame_results_history)}, "
                   f"Balls: {sum(len(fr['ball']) if fr['ball'] else 0 for fr in frame_results_history)}")

if __name__ == '__main__':
    main() 