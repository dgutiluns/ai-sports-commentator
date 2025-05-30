"""
Script to extract frames from video for training data.
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import time

def extract_frames(video_path: str, output_dir: Path, num_frames: int = 1000):
    """Extract frames from video for training data.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save extracted frames
        num_frames: Number of frames to extract
    """
    train_dir = output_dir / "images" / "train"
    val_dir = output_dir / "images" / "val"
    label_train_dir = output_dir / "labels" / "train"
    label_val_dir = output_dir / "labels" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    label_train_dir.mkdir(parents=True, exist_ok=True)
    label_val_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nProcessing video: {video_path}")
    print(f"Total frames: {total_frames}")
    print("\nControls:")
    print("  SPACE: Save current frame")
    print("  ESC: Skip to next video")
    print("  Q: Quit")
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Display frame
        cv2.imshow('Frame Selection (SPACE=Save, ESC=Skip, Q=Quit)', frame)
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):  # Quit
            break
        elif key == 27:  # ESC - Skip to next video
            break
        elif key == 32:  # SPACE - Save frame
            # Ask user if ball is visible
            print("\nIs the ball clearly visible in this frame? (y/n)")
            while True:
                yn_key = cv2.waitKey(0) & 0xFF
                output_path = train_dir if random.random() < 0.8 else val_dir
                label_path = label_train_dir if output_path == train_dir else label_val_dir
                frame_path = output_path / f"frame_{frame_count:06d}.jpg"
                label_file = label_path / f"frame_{frame_count:06d}.txt"
                if yn_key == ord('y'):
                    cv2.imwrite(str(frame_path), frame)
                    # No label file yet; will be created during annotation
                    saved_count += 1
                    print(f"Saved frame {frame_count} ({saved_count}/{num_frames}) [ball visible]")
                    break
                elif yn_key == ord('n'):
                    cv2.imwrite(str(frame_path), frame)
                    # Create an empty label file for YOLO (no objects)
                    with open(label_file, 'w') as f:
                        pass
                    saved_count += 1
                    print(f"Saved frame {frame_count} ({saved_count}/{num_frames}) [NO BALL]")
                    break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nExtracted {saved_count} frames:")
    print(f"Training frames: {len(list(train_dir.glob('*.jpg')))}")
    print(f"Validation frames: {len(list(val_dir.glob('*.jpg')))}")

def process_videos(video_paths: list, output_dir: Path, frames_per_video: int = 200, start_from: str = None):
    """Process multiple videos to extract training frames.
    
    Args:
        video_paths: List of paths to video files
        output_dir: Directory to save extracted frames
        frames_per_video: Number of frames to extract per video
        start_from: Filename to start from (inclusive)
    """
    total_saved = 0
    start = False if start_from else True
    for video_path in video_paths:
        if not Path(video_path).exists():
            print(f"Warning: Video not found: {video_path}")
            continue
        if not start:
            if Path(video_path).name == start_from:
                start = True
            else:
                print(f"Skipping {video_path}")
                continue
        extract_frames(video_path, output_dir, frames_per_video)
        total_saved += len(list((output_dir / "images" / "train").glob('*.jpg')))
        total_saved += len(list((output_dir / "images" / "val").glob('*.jpg')))
        if total_saved >= 1000:  # Stop if we have enough frames
            break

def main():
    # Automatically find all .mp4 videos in data/ball_det_train_data_b2/
    video_dir = Path("data/ball_det_train_data_b2")
    video_paths = sorted([str(p) for p in video_dir.glob("*.mp4")])
    if not video_paths:
        print("No videos found in data/ball_det_train_data_b2/")
        return
    print(f"Found {len(video_paths)} videos:")
    for v in video_paths:
        print(f"  {v}")
    # Save to the same output directory as before
    output_dir = Path("data/soccer_ball")
    # Start from 112.mp4
    process_videos(video_paths, output_dir, start_from="112.mp4")
    print("\nNext steps:")
    print("1. Label the extracted frames using a tool like LabelImg or makesense.ai")
    print("2. Export YOLO labels to data/soccer_ball/labels/train/")
    print("3. Retrain or fine-tune your model!")

if __name__ == "__main__":
    main() 