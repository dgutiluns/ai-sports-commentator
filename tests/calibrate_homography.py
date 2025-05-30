import cv2
import numpy as np

VIDEO_PATH = 'data/pipelineV1/ROMVER.mp4'
NUM_FRAMES = 3  # first, 1/3, 2/3

# Open video and get total frames
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = [0, total_frames // 3, (2 * total_frames) // 3]
frame_labels = ['start', 'one_third', 'two_third']

all_points = {}

def click_event(event, x, y, flags, param):
    frame_label, frame, clicked_points = param
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"{frame_label} - Point {len(clicked_points)}: ({x}, {y})")
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow(f'Frame: {frame_label}', frame)

for idx, frame_idx in enumerate(frame_indices):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {frame_idx}")
        continue

    frame_label = frame_labels[idx]
    print(f"\nSelect points for {frame_label} frame (frame {frame_idx}).")
    print("Click 4 or more field reference points (corners, line intersections, center, etc.).")
    print("Close the window when done.")

    clicked_points = []
    clone = frame.copy()
    cv2.imshow(f'Frame: {frame_label}', frame)
    cv2.setMouseCallback(f'Frame: {frame_label}', click_event, [frame_label, clone, clicked_points])

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            break
        if len(clicked_points) >= 4:
            # Optionally, break automatically after 4 points
            pass

    cv2.destroyAllWindows()
    all_points[frame_label] = clicked_points

cap.release()

print("\nCalibration points for each frame:")
for label, points in all_points.items():
    print(f"{label}: {points}")

print("\nNow, record the corresponding real-world field coordinates for these points for each frame.") 