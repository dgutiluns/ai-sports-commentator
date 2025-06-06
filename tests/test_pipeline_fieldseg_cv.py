import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
import numpy as np
from src.vision.detector import VisionDetector
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

VIDEO_PATH = 'data/pipelineV2/115.mp4'
vision = VisionDetector(sport='soccer')
event_detector = EventDetector()

all_events = []

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

output_path = 'data/pipelineV2/fieldseg_cv_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
all_ball_positions = []
all_player_tracks = []
ball_buffer = deque(maxlen=6)

print("\n=== Starting Event Detection with Field Segmentation (CV) ===")
print("Press Ctrl+C to stop and see summary\n")
print(f"Annotated video will be saved to: {output_path}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run vision detector
        detections = vision.detect(frame)

        # Store original image coordinates for visualization
        for player in detections['players']:
            player['img_x'] = player['x']
            player['img_y'] = player['y']
        if detections['ball'] is not None:
            if 'img_x' not in detections['ball']:
                detections['ball']['img_x'] = detections['ball'].get('x', 0)
            if 'img_y' not in detections['ball']:
                detections['ball']['img_y'] = detections['ball'].get('y', 0)

        # Add current ball detection to buffer
        ball_buffer.append((frame_idx, detections['ball'].copy() if detections['ball'] is not None else None))

        # Linear interpolation for missing ball detections (same as before)
        if detections['ball'] is None:
            prev_idx, prev_ball = None, None
            next_idx, next_ball = None, None
            for i in range(len(ball_buffer)-2, -1, -1):
                idx, ball = ball_buffer[i]
                if ball is not None:
                    prev_idx, prev_ball = idx, ball
                    break
            lookahead = 0
            temp_buffer = []
            while lookahead < 5:
                ret2, frame2 = cap.read()
                if not ret2:
                    break
                temp_detections = vision.detect(frame2)
                temp_ball = temp_detections['ball']
                if temp_ball is not None:
                    if 'img_x' not in temp_ball:
                        temp_ball['img_x'] = temp_ball.get('x', 0)
                    if 'img_y' not in temp_ball:
                        temp_ball['img_y'] = temp_ball.get('y', 0)
                temp_buffer.append((frame_idx+lookahead+1, temp_ball.copy() if temp_ball is not None else None))
                if temp_ball is not None:
                    next_idx, next_ball = frame_idx+lookahead+1, temp_ball
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

        # 3. Filter player and ball detections to those inside the field mask
        def is_on_field(x, y):
            xi, yi = int(round(x)), int(round(y))
            return 0 <= xi < frame_width and 0 <= yi < frame_height and field_mask[yi, xi] > 0
        filtered_players = [p for p in detections['players'] if is_on_field(p['img_x'], p['img_y'])]
        filtered_ball = detections['ball'] if (detections['ball'] is not None and is_on_field(detections['ball']['img_x'], detections['ball']['img_y'])) else None
        detections['players'] = filtered_players
        detections['ball'] = filtered_ball

        # Map image coordinates to field coordinates
        if detections['ball'] is not None:
            # Map ball coordinates to field space (0-1000 x, 0-600 y)
            detections['ball']['x'] = (detections['ball']['img_x'] / frame_width) * 1000
            detections['ball']['y'] = (detections['ball']['img_y'] / frame_height) * 600

        for player in detections['players']:
            # Map player coordinates to field space
            player['x'] = (player['img_x'] / frame_width) * 1000
            player['y'] = (player['img_y'] / frame_height) * 600

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
        cv2.imshow('Event Detection (FieldSeg CV)', overlay)
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