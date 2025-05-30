"""
Script to train YOLOv8 model for soccer ball detection.
"""

from pathlib import Path
from ultralytics import YOLO
import torch

def train_model():
    """Train YOLOv8 model on soccer ball dataset."""
    print("Loading base model...")
    model = YOLO('yolov8m.pt')
    
    print("\nStarting training...")
    results = model.train(
        data='data/soccer_ball/data.yaml',
        epochs=50,
        imgsz=960,  # Higher resolution for better small object detection
        batch=8,
        name='soccer_ball_detector',
        patience=10,  # Early stopping patience
        save=True,  # Save best model
        device='0' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    print("\nTraining completed!")
    print(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    # Copy the best model to our models directory
    best_model = Path(results.save_dir) / "weights" / "best.pt"
    target_path = Path("models") / "soccernet_yolov8.pt"
    if best_model.exists():
        import shutil
        shutil.copy(best_model, target_path)
        print(f"Best model copied to: {target_path}")

if __name__ == "__main__":
    train_model() 