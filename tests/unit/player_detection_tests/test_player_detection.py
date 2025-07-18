import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
import cv2
import numpy as np
from src.vision.detector import VisionDetector, RoboflowPlayerDetector

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'player_detection_log.txt')

VIDEO_PATH = 'data/Bundesliga_Games/B1.mp4'
YOLO_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'yolo_player_detection_video.mp4')
ROBOFLOW_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'roboflow_player_detection_video.mp4')

FRAME_LIMIT = 250  # Limit to first 50 frames for speed

def draw_boxes(frame, bboxes, color=(0,255,0), label_prefix=""):
    for i, bbox in enumerate(bboxes):
        if len(bbox) == 5:
            x1, y1, x2, y2, conf = bbox
        else:
            x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label_prefix}{i+1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def test_video_player_detection():
    log_lines = []
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_yolo = cv2.VideoWriter(YOLO_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))
    out_roboflow = cv2.VideoWriter(ROBOFLOW_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

    yolo_detector = VisionDetector(sport="soccer")
    roboflow_detector = RoboflowPlayerDetector()

    yolo_counts = []
    roboflow_counts = []
    frame_idx = 0
    while frame_idx < FRAME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            break
        # YOLOv8 detection
        yolo_result = yolo_detector.player_model(frame)
        yolo_bboxes = [box.xyxy[0].tolist() for box in yolo_result[0].boxes]
        yolo_counts.append(len(yolo_bboxes))
        yolo_vis = draw_boxes(frame.copy(), yolo_bboxes, color=(0,255,0), label_prefix="YOLO_")
        out_yolo.write(yolo_vis)
        log_lines.append(f"Frame {frame_idx}: YOLOv8 detected {len(yolo_bboxes)} players.")
        # Roboflow detection
        roboflow_bboxes = roboflow_detector.detect_players(frame)
        roboflow_counts.append(len(roboflow_bboxes))
        roboflow_vis = draw_boxes(frame.copy(), roboflow_bboxes, color=(255,0,0), label_prefix="RF_")
        out_roboflow.write(roboflow_vis)
        log_lines.append(f"Frame {frame_idx}: Roboflow detected {len(roboflow_bboxes)} players.")
        frame_idx += 1
    cap.release()
    out_yolo.release()
    out_roboflow.release()
    # Summary stats
    avg_yolo = np.mean(yolo_counts) if yolo_counts else 0
    avg_roboflow = np.mean(roboflow_counts) if roboflow_counts else 0
    summary = [
        f"Processed {frame_idx} frames.",
        f"YOLOv8: avg detections/frame = {avg_yolo:.2f}",
        f"Roboflow: avg detections/frame = {avg_roboflow:.2f}",
        f"YOLOv8 video saved as {YOLO_VIDEO_PATH}",
        f"Roboflow video saved as {ROBOFLOW_VIDEO_PATH}"
    ]
    for line in summary:
        print(line)
        log_lines.append(line)
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            f.write(line + '\n')
    print(f"\nLog saved to {LOG_PATH}")

if __name__ == "__main__":
    test_video_player_detection() 