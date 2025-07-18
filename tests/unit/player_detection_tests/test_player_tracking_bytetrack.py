import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import cv2
import numpy as np
import supervision as sv
from src.vision.detector import VisionDetector, RoboflowPlayerDetector

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
VIDEO_PATH = 'data/Bundesliga_Games/B1.mp4'
OUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'player_tracking_bytetrack_comparison.mp4')
LOG_PATH = os.path.join(OUTPUT_DIR, 'player_tracking_bytetrack_log.txt')
FRAME_LIMIT = 400

COLORS = [(np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)) for _ in range(100)]

def draw_boxes(frame, detections, tracker_ids, label_prefix="TID_"):
    for bbox, tid in zip(detections.xyxy, tracker_ids):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = COLORS[int(tid) % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{label_prefix}{tid}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def test_player_tracking_bytetrack():
    log_lines = []
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(OUT_VIDEO_PATH, fourcc, fps, (frame_width*2, frame_height))

    yolo_detector = VisionDetector(sport="soccer")
    roboflow_detector = RoboflowPlayerDetector()
    tracker_yolo = sv.ByteTrack()
    tracker_roboflow = sv.ByteTrack()

    yolo_id_history = set()
    roboflow_id_history = set()
    frame_idx = 0
    while frame_idx < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLOv8+ByteTrack
        yolo_result = yolo_detector.player_model(frame)
        yolo_sv_detections = sv.Detections.from_ultralytics(yolo_result[0])
        player_mask = (yolo_sv_detections.confidence > yolo_detector.confidence_threshold) & \
                      (yolo_sv_detections.class_id == 0)
        yolo_sv_detections = yolo_sv_detections[player_mask]
        yolo_tracked = tracker_yolo.update_with_detections(yolo_sv_detections)
        yolo_id_history.update(yolo_tracked.tracker_id)
        yolo_frame = draw_boxes(frame.copy(), yolo_tracked, yolo_tracked.tracker_id, label_prefix="YOLO_")
        # Roboflow+ByteTrack
        roboflow_bboxes = roboflow_detector.detect_players(frame)
        if roboflow_bboxes:
            xyxy = np.array([b[:4] for b in roboflow_bboxes])
            conf = np.array([b[4] for b in roboflow_bboxes])
            class_id = np.zeros(len(roboflow_bboxes), dtype=int)
            roboflow_sv_detections = sv.Detections(
                xyxy=xyxy,
                confidence=conf,
                class_id=class_id
            )
            roboflow_tracked = tracker_roboflow.update_with_detections(roboflow_sv_detections)
            roboflow_id_history.update(roboflow_tracked.tracker_id)
            roboflow_frame = draw_boxes(frame.copy(), roboflow_tracked, roboflow_tracked.tracker_id, label_prefix="RF_")
        else:
            roboflow_frame = frame.copy()
        # Side-by-side
        combined = np.concatenate([yolo_frame, roboflow_frame], axis=1)
        out_video.write(combined)
        # Log
        log_lines.append(f"Frame {frame_idx}: YOLOv8+ByteTrack IDs: {list(yolo_tracked.tracker_id)} | Roboflow+ByteTrack IDs: {list(roboflow_tracked.tracker_id) if roboflow_bboxes else []}")
        frame_idx += 1
    cap.release()
    out_video.release()
    # Summary
    summary = [
        f"Processed {frame_idx} frames.",
        f"YOLOv8+ByteTrack: {len(yolo_id_history)} unique IDs tracked.",
        f"Roboflow+ByteTrack: {len(roboflow_id_history)} unique IDs tracked.",
        f"Output video saved as {OUT_VIDEO_PATH}"
    ]
    for line in summary:
        print(line)
        log_lines.append(line)
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')
    print(f"\nLog saved to {LOG_PATH}")

if __name__ == "__main__":
    test_player_tracking_bytetrack() 