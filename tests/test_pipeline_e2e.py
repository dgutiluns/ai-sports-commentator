import cv2
from src.vision.detector import VisionDetector
from src.game_engine.event_detector import EventDetector
from src.vision.homography import get_homography_for_frame, map_to_field

VIDEO_PATH = 'data/pipelineV1/ROMVER.mp4'

# Initialize detectors
vision = VisionDetector(sport='soccer')
event_detector = EventDetector()

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Run vision detector
    detections = vision.detect(frame)

    # 2. Map all player and ball positions to field coordinates
    H = get_homography_for_frame(frame_idx, total_frames)
    for player in detections['players']:
        player['x'], player['y'] = map_to_field(player['x'], player['y'], H)
    if detections['ball'] is not None:
        detections['ball']['x'], detections['ball']['y'] = map_to_field(
            detections['ball']['x'], detections['ball']['y'], H
        )
    # Print mapped coordinates for debug
    print(f"\n[Frame {frame_idx}] Ball field coordinates: {detections['ball'] if detections['ball'] is not None else 'None'}")
    for p in detections['players']:
        print(f"[Frame {frame_idx}] Player {p['id']} field coordinates: (x={p['x']:.2f}, y={p['y']:.2f}, team={p.get('team')})")

    # 3. Run event detector
    events = event_detector.update(frame_idx, detections)

    # 4. Print events
    for event in events:
        print(f"Frame {frame_idx}: Detected event: {event}")

    frame_idx += 1

cap.release() 