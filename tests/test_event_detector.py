#!/usr/bin/env python3
"""
Test script for the EventDetector using mock soccer data.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.game_engine.event_detector import EventDetector

# Mock detection data for several frames
mock_frames = [
    # Frame 0: Ball near red player 1
    {"players": [
        {"id": 1, "x": 100, "y": 200, "team": "red"},
        {"id": 2, "x": 300, "y": 400, "team": "blue"}
    ], "ball": {"x": 105, "y": 205}},
    # Frame 1: Ball still near red player 1 (dribble starts)
    {"players": [
        {"id": 1, "x": 120, "y": 220, "team": "red"},
        {"id": 2, "x": 300, "y": 400, "team": "blue"}
    ], "ball": {"x": 125, "y": 225}},
    # Frame 2: Ball still near red player 1 (dribble continues)
    {"players": [
        {"id": 1, "x": 140, "y": 240, "team": "red"},
        {"id": 2, "x": 300, "y": 400, "team": "blue"}
    ], "ball": {"x": 145, "y": 245}},
    # Frame 3: Ball moves to blue player 2 (turnover)
    {"players": [
        {"id": 1, "x": 150, "y": 250, "team": "red"},
        {"id": 2, "x": 160, "y": 260, "team": "blue"}
    ], "ball": {"x": 160, "y": 260}},
    # Frame 4: Ball moves to another blue player (pass within blue team)
    {"players": [
        {"id": 2, "x": 180, "y": 280, "team": "blue"},
        {"id": 3, "x": 185, "y": 285, "team": "blue"}
    ], "ball": {"x": 185, "y": 285}},
]

def main():
    detector = EventDetector()
    for idx, detections in enumerate(mock_frames):
        events = detector.update(idx, detections)
        if events:
            for event in events:
                print(f"Frame {idx}: Detected event: {event}")

if __name__ == "__main__":
    main() 