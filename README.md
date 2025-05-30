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
python -m venv venv
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