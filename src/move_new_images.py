import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Define paths
IMAGES_TRAIN_DIR = Path('images/train')
IMAGES_VAL_DIR = Path('images/val')
BATCH2_DIR = Path('images/batch2')
LABELS_TRAIN_DIR = Path('labels/train')
LABELS_VAL_DIR = Path('labels/val')

# Create batch2 directory if it doesn't exist
BATCH2_DIR.mkdir(parents=True, exist_ok=True)

# Get today's date at midnight
today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

# Function to move files added today
def move_new_files(image_dir, label_dir):
    for img_file in image_dir.glob('*.*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Check if the file was modified today
            if datetime.fromtimestamp(img_file.stat().st_mtime) >= today:
                # Move image to batch2
                shutil.move(str(img_file), str(BATCH2_DIR / img_file.name))
                print(f"Moved image: {img_file.name}")

                # Check for corresponding label file
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.move(str(label_file), str(BATCH2_DIR / label_file.name))
                    print(f"Moved label: {label_file.name}")

# Process train and val directories
move_new_files(IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR)
move_new_files(IMAGES_VAL_DIR, LABELS_VAL_DIR)

print("Done moving new files to batch2.") 