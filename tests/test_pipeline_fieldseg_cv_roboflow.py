import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from src.vision.detector import VisionDetector, RoboflowPlayerDetector
from src.game_engine.event_detector import EventDetector
from src.vision.field_segmentation import segment_field
from collections import deque
from gtts import gTTS
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip
from src.nlp.templates import event_to_commentary
from src.tts.gtts_speaker import save_tts_audio
from src.utils.overlay import overlay_audio_on_video
from src.vision.ball_smoothing import filter_and_smooth_ball_positions
from src.player_tracking import merge_player_ids
from src.utils.audio_queue import queue_audio_clips
from src.utils.event_postprocessing import deduplicate_events
from src.utils.event_utils import filter_self_passes
from src.vision.homography_auto import HomographyEstimator

VIDEO_PATH = 'data/pipelineV2/115.mp4'
# Use Roboflow for player detection, keep YOLO for ball
yolo_vision = VisionDetector(sport='soccer')
roboflow_detector = RoboflowPlayerDetector()
event_detector = EventDetector()

FIELD_COORDS = np.array([
    [0, 0],        # top-left
    [0, 68],       # bottom-left
    [105, 68],     # bottom-right
    [105, 0]       # top-right
], dtype=np.float32)

homography_estimator = HomographyEstimator(FIELD_COORDS, smoothing=0)

all_events = []

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = 'data/pipelineV2/fieldseg_cv_roboflow_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
all_ball_positions = []
all_player_tracks = []
ball_buffer = deque(maxlen=6)

print("\n=== Starting Event Detection with Field Segmentation (CV) + Roboflow Players ===")
print("Press Ctrl+C to stop and see summary\n")
print(f"Annotated video will be saved to: {output_path}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run player detection with Roboflow, ball detection with YOLO
        # --- Player detection ---
        roboflow_bboxes = roboflow_detector.detect_players(frame)
        players = []
        for i, bbox in enumerate(roboflow_bboxes):
            x1, y1, x2, y2, conf = bbox
            # Use center of bbox for player position
            x = (x1 + x2) // 2
            y = (y1 + y2) // 2
            players.append({
                'id': i+1,  # Assign a unique ID per frame (tracking can be improved)
                'x': x,
                'y': y,
                'img_x': x,
                'img_y': y,
                'team': None,  # Team assignment can be added if needed
                'number': None
            })
        # --- Ball detection (use YOLO from VisionDetector) ---
        yolo_ball = yolo_vision.ball_model(frame)
        try:
            ball_detections = sv.Detections.from_ultralytics(yolo_ball[0])
        except Exception as e:
            ball_detections = None
        ball = None
        if ball_detections is not None:
            ball_mask = (ball_detections.confidence > yolo_vision.ball_confidence_threshold)
            ball_detections = ball_detections[ball_mask]
            if len(ball_detections) > 0:
                best_ball_idx = np.argmax(ball_detections.confidence)
                bbox = ball_detections.xyxy[best_ball_idx].astype(int)
                x1, y1, x2, y2 = bbox
                ball = {'img_x': (x1 + x2) // 2, 'img_y': (y1 + y2) // 2}

        detections = {'players': players, 'ball': ball}

        # Add current ball detection to buffer
        ball_buffer.append((frame_idx, ball.copy() if ball is not None else None))

        # Linear interpolation for missing ball detections (same as before)
        if detections['ball'] is None:
            prev_idx, prev_ball = None, None
            next_idx, next_ball = None, None
            for i in range(len(ball_buffer)-2, -1, -1):
                idx, b = ball_buffer[i]
                if b is not None:
                    prev_idx, prev_ball = idx, b
                    break
            lookahead = 0
            temp_buffer = []
            while lookahead < 5:
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                # Ball detection only
                yolo_ball2 = yolo_vision.ball_model(frame2)
                try:
                    ball_detections2 = sv.Detections.from_ultralytics(yolo_ball2[0])
                except Exception as e:
                    ball_detections2 = None
                b2 = None
                if ball_detections2 is not None:
                    ball_mask2 = (ball_detections2.confidence > yolo_vision.ball_confidence_threshold)
                    ball_detections2 = ball_detections2[ball_mask2]
                    if len(ball_detections2) > 0:
                        best_ball_idx2 = np.argmax(ball_detections2.confidence)
                        bbox2 = ball_detections2.xyxy[best_ball_idx2].astype(int)
                        x1, y1, x2, y2 = bbox2
                        b2 = {'img_x': (x1 + x2) // 2, 'img_y': (y1 + y2) // 2}
                temp_buffer.append((frame_idx+lookahead+1, b2.copy() if b2 is not None else None))
                if b2 is not None:
                    next_idx, next_ball = frame_idx+lookahead+1, b2
                    break
                lookahead += 1
            for item in temp_buffer:
                ball_buffer.append(item)
            if prev_ball is not None and next_ball is not None and next_idx != prev_idx:
                alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
                interp_x = prev_ball['img_x'] + alpha * (next_ball['img_x'] - prev_ball['img_x'])
                interp_y = prev_ball['img_y'] + alpha * (next_ball['img_y'] - prev_ball['img_y'])
                interp_ball = {'img_x': interp_x, 'img_y': interp_y}
                detections['ball'] = interp_ball
            elif prev_ball is not None:
                detections['ball'] = prev_ball
            elif next_ball is not None:
                detections['ball'] = next_ball

        if detections['ball'] is not None:
            if 'img_x' not in detections['ball']:
                detections['ball']['img_x'] = detections['ball'].get('x', 0)
            if 'img_y' not in detections['ball']:
                detections['ball']['img_y'] = detections['ball'].get('y', 0)

        # 2. Segment the field using the new technique
        field_mask = segment_field(frame)

        # --- Automatic homography estimation ---
        H = homography_estimator.estimate(field_mask)
        # --- End homography estimation ---

        # 3. Filter player and ball detections to those inside the field mask
        def is_on_field(x, y):
            xi, yi = int(round(x)), int(round(y))
            return 0 <= xi < frame_width and 0 <= yi < frame_height and field_mask[yi, xi] > 0
        filtered_players = [p for p in detections['players'] if is_on_field(p['img_x'], p['img_y'])]
        filtered_ball = detections['ball'] if (detections['ball'] is not None and is_on_field(detections['ball']['img_x'], detections['ball']['img_y'])) else None
        detections['players'] = filtered_players
        detections['ball'] = filtered_ball

        # Map image coordinates to field coordinates using homography estimator
        if detections['ball'] is not None and H is not None:
            x, y = homography_estimator.map_point(detections['ball']['img_x'], detections['ball']['img_y'], H)
            detections['ball']['x'] = x
            detections['ball']['y'] = y

        for player in detections['players']:
            if H is not None:
                x, y = homography_estimator.map_point(player['img_x'], player['img_y'], H)
                player['x'] = x
                player['y'] = y

        # Collect for smoothing/merging
        all_ball_positions.append({'frame': frame_idx, 'x': detections['ball']['img_x'], 'y': detections['ball']['img_y']} if detections['ball'] is not None else None)
        all_player_tracks.extend([{**p, 'frame': frame_idx} for p in detections['players']])

        # 4. Run event detector
        events = event_detector.update(frame_idx, {
            'ball': detections['ball'],
            'players': detections['players']
        })
        events = filter_self_passes(events)

        # 5. Visualize detections, events, and field mask
        overlay = frame.copy()
        overlay[field_mask > 0] = (0.5 * overlay[field_mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        for player in detections['players']:
            x, y = player['img_x'], player['img_y']
            cv2.circle(overlay, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(overlay, f"P{player['id']}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if detections['ball'] is not None:
            x, y = detections['ball']['img_x'], detections['ball']['img_y']
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), -1)
        if events:
            y_offset = 30
            for event in events:
                event_text = f"{event['event'].upper()}"
                if 'by_player' in event:
                    event_text += f" by P{event['by_player']}"
                if 'from_player' in event:
                    event_text += f" from P{event['from_player']}"
                if 'to_player' in event:
                    event_text += f" to P{event['to_player']}"
                if 'team' in event:
                    event_text += f" ({event['team']})"
                cv2.putText(overlay, event_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
        cv2.putText(overlay, f"Frame: {frame_idx}", (10, frame_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out.write(overlay)
        cv2.imshow('Event Detection (FieldSeg CV) + Roboflow', overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if events:
            print(f"\n{'='*50}")
            print(f"Frame {frame_idx}: Detected {len(events)} events:")
            for event in events:
                print(f"  - {event['event'].upper()}")
                if 'by_player' in event:
                    print(f"    By player: {event['by_player']}")
                if 'from_player' in event:
                    print(f"    From player: {event['from_player']}")
                if 'to_player' in event:
                    print(f"    To player: {event['to_player']}")
                if 'team' in event:
                    print(f"    Team: {event['team']}")
            print(f"{'='*50}\n")
            all_events.extend(events)
        frame_idx += 1

except KeyboardInterrupt:
    print("\nStopping video processing...")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("\n\n=== Event Detection Summary ===")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total events detected: {len(all_events)}")
    event_types = {}
    for event in all_events:
        event_type = event['event']
        event_types[event_type] = event_types.get(event_type, 0) + 1
    print("\nEvents by type:")
    for event_type, count in event_types.items():
        print(f"  - {event_type}: {count}")
    print("\nDetailed event log:")
    for event in all_events:
        print(f"Frame {event['frame']}: {event['event']}")

    # === Post-processing and Commentary Generation ===
    print("\nApplying post-processing (smoothing, merging, deduplication)...")
    smoothed_ball_positions = filter_and_smooth_ball_positions(all_ball_positions)
    print("Sample player track:", all_player_tracks[0] if all_player_tracks else "No tracks")
    merged_player_tracks = merge_player_ids(all_player_tracks, distance_threshold=7)
    all_events = []
    for idx in range(len(smoothed_ball_positions)):
        events = event_detector.update(idx, {
            'ball': smoothed_ball_positions[idx],
            'players': [p for p in merged_player_tracks if p['frame'] == idx]
        })
        if events:
            all_events.extend(events)
    all_events = deduplicate_events(all_events)
    print("\nGenerating commentary audio and overlaying on video...")
    fps = int(fps) if isinstance(fps, float) else fps
    audio_dir = "event_audio"
    audio_clips = []
    for event in all_events:
        text = event_to_commentary(event)
        audio_path = os.path.join(audio_dir, f"event_{event['frame']}.mp3")
        os.makedirs(audio_dir, exist_ok=True)
        save_tts_audio(text, audio_path)
        event['audio_path'] = audio_path
        event['audio_time'] = event['frame'] / fps
        audio_clips.append((audio_path, event['audio_time']))
    overlay_audio_on_video(output_path, audio_clips, output_path.replace("_annotated.mp4", "_commentary.mp4"))
    print("Done! Commentary video saved as:", output_path.replace("_annotated.mp4", "_commentary.mp4")) 