import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.game_engine.event_detector import EventDetector

def test_event_logic():
    mock_frames = [
        {"players": [
            {"id": 1, "x": 100, "y": 200, "team": "red"},
            {"id": 2, "x": 300, "y": 400, "team": "blue"}
        ], "ball": {"x": 105, "y": 205}},
        {"players": [
            {"id": 1, "x": 120, "y": 220, "team": "red"},
            {"id": 2, "x": 300, "y": 400, "team": "blue"}
        ], "ball": {"x": 125, "y": 225}},
        {"players": [
            {"id": 1, "x": 140, "y": 240, "team": "red"},
            {"id": 2, "x": 300, "y": 400, "team": "blue"}
        ], "ball": {"x": 145, "y": 245}},
        {"players": [
            {"id": 1, "x": 150, "y": 250, "team": "red"},
            {"id": 2, "x": 160, "y": 260, "team": "blue"}
        ], "ball": {"x": 160, "y": 260}},
        {"players": [
            {"id": 2, "x": 180, "y": 280, "team": "blue"},
            {"id": 3, "x": 185, "y": 285, "team": "blue"}
        ], "ball": {"x": 185, "y": 285}},
    ]
    detector = EventDetector()
    all_events = []
    for idx, detections in enumerate(mock_frames):
        events = detector.update(idx, detections)
        print(f"Frame {idx}: {len(events)} events detected.")
        for event in events:
            print(f"  - {event['event'].upper()} | by: {event.get('by_player', '-')}, from: {event.get('from_player', '-')}, to: {event.get('to_player', '-')}, team: {event.get('team', '-')} ")
        all_events.extend(events)
    assert len(all_events) > 0, "No events detected in event logic test"
    print(f"Total events detected: {len(all_events)}")
    print("Event logic test passed.")

if __name__ == "__main__":
    test_event_logic() 