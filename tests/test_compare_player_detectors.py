import cv2
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.vision.detector import VisionDetector, RoboflowPlayerDetector
from src.vision.homography import get_homography_for_frame, map_to_field
import supervision as sv
from dotenv import load_dotenv

load_dotenv()

VIDEO_PATH = 'data/pipelineV1/ROMVER.mp4'
OUTPUT_YOLO = 'data/ROMVER_yolov8_tracked.mp4'
OUTPUT_ROBOFLOW = 'data/ROMVER_roboflow_tracked.mp4'

# Ensure output directory exists
os.makedirs('data', exist_ok=True)

# Initialize detectors
vision = VisionDetector(sport='soccer')
roboflow_detector = RoboflowPlayerDetector()

# Initialize tracker (ByteTrack)
tracker_yolo = sv.ByteTrack()
tracker_roboflow = sv.ByteTrack()

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Video writers
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_yolo = cv2.VideoWriter(OUTPUT_YOLO, fourcc, fps, (frame_width, frame_height))
out_roboflow = cv2.VideoWriter(OUTPUT_ROBOFLOW, fourcc, fps, (frame_width, frame_height))

frame_idx = 0

# Color palette for tracker IDs
COLORS = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in range(100)]

def draw_boxes(frame, detections, tracker_ids):
    for bbox, tid in zip(detections.xyxy, tracker_ids):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS[int(tid) % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID {tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    print(f'\n[Frame {frame_idx}]')
    # --- YOLOv8 Detection ---
    yolo_detections = vision.player_model(frame)
    yolo_sv_detections = sv.Detections.from_ultralytics(yolo_detections[0])
    # Filter for players (class_id == 0)
    player_mask = (yolo_sv_detections.confidence > vision.confidence_threshold) & \
                  (yolo_sv_detections.class_id == 0)
    yolo_sv_detections = yolo_sv_detections[player_mask]
    # Track
    yolo_tracked = tracker_yolo.update_with_detections(yolo_sv_detections)
    print('YOLOv8+ByteTrack:')
    for bbox, tid in zip(yolo_tracked.xyxy, yolo_tracked.tracker_id):
        print(f'  ID {tid}: bbox={bbox}')
    # Draw and save
    yolo_frame = frame.copy()
    yolo_frame = draw_boxes(yolo_frame, yolo_tracked, yolo_tracked.tracker_id)
    out_yolo.write(yolo_frame)

    # --- Roboflow Detection ---
    roboflow_bboxes = roboflow_detector.detect_players(frame)
    if roboflow_bboxes:
        # Convert to sv.Detections format
        xyxy = np.array([b[:4] for b in roboflow_bboxes])
        conf = np.array([b[4] for b in roboflow_bboxes])
        class_id = np.zeros(len(roboflow_bboxes), dtype=int)  # Assume all are 'person'
        roboflow_sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id
        )
        roboflow_tracked = tracker_roboflow.update_with_detections(roboflow_sv_detections)
        print('Roboflow+ByteTrack:')
        for bbox, tid in zip(roboflow_tracked.xyxy, roboflow_tracked.tracker_id):
            print(f'  ID {tid}: bbox={bbox}')
        # Draw and save
        roboflow_frame = frame.copy()
        roboflow_frame = draw_boxes(roboflow_frame, roboflow_tracked, roboflow_tracked.tracker_id)
        out_roboflow.write(roboflow_frame)
    else:
        print('Roboflow+ByteTrack: No detections')
        out_roboflow.write(frame)

    frame_idx += 1
    if frame_idx >= 50:
        break  # Limit to first 50 frames for quick comparison

cap.release()
out_yolo.release()
out_roboflow.release() 