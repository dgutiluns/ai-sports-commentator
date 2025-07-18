# AI Sports Commentator

An AI-powered sports commentary system focused on soccer, using computer vision and natural language generation to provide real-time commentary on matches.

## Features

- Real-time object detection for players and ball using YOLOv8
- Team color clustering for player identification
- Homography mapping for field coordinate tracking
- Event detection logic for game events
- Natural language generation for commentary

## Project Structure

```
.
├── src/                    # Source code
│   ├── detection/         # Object detection modules
│   ├── event_logic/      # Game event detection
│   └── nlg/              # Natural language generation
├── images/               # Dataset images
│   ├── train/           # Training images
│   └── val/             # Validation images
├── labels/              # YOLO format labels
│   ├── train/          # Training labels
│   └── val/            # Validation labels
└── models/             # Trained models
```

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[Add usage instructions here]

## Development

[Add development instructions here]

## License

[Add license information here] 

## New Main Pipeline: Roboflow + ByteTrack for Player Tracking

The new `src/main_pipeline.py` script runs the AI Commentator pipeline using Roboflow for player detection and ByteTrack for player tracking. This is now the recommended approach for player tracking, replacing YOLOv8 and merge_player_ids. Ball tracking, event detection, and commentary are not yet integrated in this script—they are left as TODOs for future updates.

### Usage

```bash
python3 src/main_pipeline.py --input path/to/video.mp4 --output path/to/annotated_output.mp4 --debug
```