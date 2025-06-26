"""
Script to download required models for the AI Commentator.
"""

import os
import requests
from pathlib import Path
import torch
from ultralytics import YOLO

def download_base_model():
    """Download the base YOLOv8 model."""
    print("Downloading base YOLOv8 model...")
    model = YOLO('yolov8m.pt')  # Using medium size for better accuracy
    return model

def setup_training_config():
    """Create training configuration for soccer ball detection."""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create data.yaml for training
    data_yaml = """
path: data/soccer_ball  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
names:
  0: soccer_ball
    """
    
    with open("data/soccer_ball/data.yaml", "w") as f:
        f.write(data_yaml.strip())
    
    print("Created training configuration at data/soccer_ball/data.yaml")

def main():
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("data/soccer_ball/images/train").mkdir(parents=True, exist_ok=True)
    Path("data/soccer_ball/images/val").mkdir(parents=True, exist_ok=True)
    
    # Download base model
    model = download_base_model()
    
    # Setup training configuration
    setup_training_config()
    
    print("\nNext steps:")
    print("1. Place your soccer ball training images in data/soccer_ball/images/train")
    print("2. Place your validation images in data/soccer_ball/images/val")
    print("3. Run the training script: python3 scripts/train_ball_detector.py")

if __name__ == "__main__":
    main() 