import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from src.vision.detector import VisionDetector
from src.game_engine.event_detector import EventDetector
from src.vision.homography import get_homography_for_frame, map_to_field
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

# Use the ROMVER.mp4 video that we calibrated homography for
VIDEO_PATH = 'data/pipelineV1/ROMVER.mp4'

# Initialize detectors
vision = VisionDetector(sport='soccer')
event_detector = EventDetector()

# Keep track of all events for summary
all_events = []

# Setup video capture and writer
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create video writer for annotated output
output_path = 'data/pipelineV1/ROMVER_annotated.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
all_ball_positions = []
all_player_tracks = []

# Buffer for ball detections (frame_idx, ball dict or None)
ball_buffer = deque(maxlen=6)  # 5 lookahead + current

print("\n=== Starting Event Detection ===")
print("Press Ctrl+C to stop and see summary\n")
print(f"Annotated video will be saved to: {output_path}")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Run vision detector with our new model
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

        # Linear interpolation for missing ball detections
        if detections['ball'] is None:
            # Find last known and next known ball positions in buffer
            prev_idx, prev_ball = None, None
            next_idx, next_ball = None, None
            # Search backwards for previous detection
            for i in range(len(ball_buffer)-2, -1, -1):
                idx, ball = ball_buffer[i]
                if ball is not None:
                    prev_idx, prev_ball = idx, ball
                    break
            # Search forwards for next detection (need to read ahead)
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
            # Add lookahead frames to buffer
            for item in temp_buffer:
                ball_buffer.append(item)
            # If both found, interpolate
            if prev_ball is not None and next_ball is not None and next_idx != prev_idx:
                alpha = (frame_idx - prev_idx) / (next_idx - prev_idx)
                interp_x = prev_ball['img_x'] + alpha * (next_ball['img_x'] - prev_ball['img_x'])
                interp_y = prev_ball['img_y'] + alpha * (next_ball['img_y'] - prev_ball['img_y'])
                interp_ball = {'img_x': interp_x, 'img_y': interp_y}
                # Also interpolate field coordinates for event logic
                interp_field_x = prev_ball['x'] + alpha * (next_ball['x'] - prev_ball['x'])
                interp_field_y = prev_ball['y'] + alpha * (next_ball['y'] - prev_ball['y'])
                interp_ball['x'] = interp_field_x
                interp_ball['y'] = interp_field_y
                detections['ball'] = interp_ball
            elif prev_ball is not None:
                # Hold last known position
                detections['ball'] = prev_ball
            elif next_ball is not None:
                detections['ball'] = next_ball
            # else: leave as None

        # Ensure every ball detection has 'img_x' and 'img_y' for visualization
        if detections['ball'] is not None:
            if 'img_x' not in detections['ball']:
                detections['ball']['img_x'] = detections['ball'].get('x', 0)
            if 'img_y' not in detections['ball']:
                detections['ball']['img_y'] = detections['ball'].get('y', 0)

        # 2. Map all player and ball positions to field coordinates
        H = get_homography_for_frame(frame_idx, total_frames)
        for player in detections['players']:
            player['x'], player['y'] = map_to_field(player['x'], player['y'], H)
        if detections['ball'] is not None:
            detections['ball']['x'], detections['ball']['y'] = map_to_field(
                detections['ball']['img_x'], detections['ball']['img_y'], H
            )

        # Collect for smoothing/merging
        all_ball_positions.append({'frame': frame_idx, 'x': detections['ball']['img_x'], 'y': detections['ball']['img_y']} if detections['ball'] is not None else None)
        all_player_tracks.extend([{**p, 'frame': frame_idx} for p in detections['players']])

        # 3. Run event detector
        events = event_detector.update(frame_idx, {
            'ball': detections['ball'],
            'players': detections['players']
        })
        
        # Filter out self-passes
        events = filter_self_passes(events)
        
        # 4. Visualize detections and events
        # Draw player circles at original image coordinates
        for player in detections['players']:
            x, y = player['img_x'], player['img_y']
            cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(frame, f"P{player['id']}", (int(x), int(y)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw ball at original image coordinates
        if detections['ball'] is not None:
            x, y = detections['ball']['img_x'], detections['ball']['img_y']
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Draw events
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
                
                cv2.putText(frame, event_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30

        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, frame_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Write frame to output video
        out.write(frame)

        # Display frame
        cv2.imshow('Event Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Print events to console
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
    
    # Print summary regardless of how the script ends
    print("\n\n=== Event Detection Summary ===")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total events detected: {len(all_events)}")
    
    # Count events by type
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

    # === Commentary Audio Generation and Overlay ===
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

    # After the frame loop, apply smoothing/merging
    smoothed_ball_positions = filter_and_smooth_ball_positions(all_ball_positions)
    merged_player_tracks = merge_player_ids(all_player_tracks)

    # Run event detection using smoothed/merged data
    all_events = []
    for idx in range(len(smoothed_ball_positions)):
        events = event_detector.update(idx, {
            'ball': smoothed_ball_positions[idx],
            'players': [p for p in merged_player_tracks if p['frame'] == idx]
        })
        if events:
            all_events.extend(events)

    # Deduplicate dribble events before generating commentary/audio
    all_events = deduplicate_events(all_events)

    # For audio commentary, generate audio clips for deduplicated events
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

    # Overlay audio clips directly (no queuing)
    overlay_audio_on_video(output_path, audio_clips, output_path.replace("_annotated.mp4", "_commentary.mp4"))
    print("Done! Commentary video saved as:", output_path.replace("_annotated.mp4", "_commentary.mp4")) 