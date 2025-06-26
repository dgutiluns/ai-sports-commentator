import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
from src.vision.detector import VisionDetector

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'ball_detection_log.txt')

def draw_boxes(frame, bboxes, color=(0,255,255), label_prefix="Ball_"):
    for i, bbox in enumerate(bboxes):
        if len(bbox) == 5:
            x1, y1, x2, y2, conf = bbox
        else:
            x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"{label_prefix}{i+1}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame

def test_yolo_ball_detection():
    log_lines = []
    image_path = "data/detector_test_images/ball_detection1.jpg"
    frame = cv2.imread(image_path)
    detector = VisionDetector(sport="soccer")
    result = detector.ball_model(frame)
    bboxes = [box.xyxy[0].tolist() for box in result[0].boxes]
    log_lines.append(f"YOLOv8 detected {len(bboxes)} balls.")
    for i, bbox in enumerate(bboxes):
        log_lines.append(f"  Ball {i+1}: bbox={bbox}")
    vis = draw_boxes(frame.copy(), bboxes, color=(0,255,255), label_prefix="Ball_")
    out_path = os.path.join(OUTPUT_DIR, "yolo_ball_detection_result.jpg")
    cv2.imwrite(out_path, vis)
    log_lines.append(f"YOLOv8 ball detection result saved as {out_path}")
    assert len(bboxes) > 0, "YOLOv8 ball detection failed"
    log_lines.append("YOLOv8 ball detection passed.")
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            print(line)
            f.write(line + '\n')
    print(f"\nLog saved to {LOG_PATH}")

if __name__ == "__main__":
    test_yolo_ball_detection() 