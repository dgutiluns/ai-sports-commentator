import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

# Define paths
IMAGES_TRAIN_DIR = Path('data/soccer_ball/images/train')
IMAGES_VAL_DIR = Path('data/soccer_ball/images/val')
BATCH2_DIR = Path('data/soccer_ball/images/batch2')
LABELS_TRAIN_DIR = Path('data/soccer_ball/labels/train')
LABELS_VAL_DIR = Path('data/soccer_ball/labels/val')

# Create all necessary directories
for directory in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, BATCH2_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Get today's date at midnight
today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
print(f"Today's midnight: {today}")

# Function to move files added today
def move_new_files(image_dir, label_dir):
    if not image_dir.exists():
        print(f"Directory {image_dir} does not exist. Skipping...")
        return
    
    for img_file in image_dir.glob('*.*'):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            mtime = datetime.fromtimestamp(img_file.stat().st_mtime)
            print(f"{img_file.name} - Modified: {mtime}")
            if mtime >= today:
                print(f"  -> {img_file.name} considered added today.")
                # Move image to batch2
                shutil.move(str(img_file), str(BATCH2_DIR / img_file.name))
                print(f"Moved image: {img_file.name}")

                # Check for corresponding label file
                label_file = label_dir / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.move(str(label_file), str(BATCH2_DIR / label_file.name))
                    print(f"Moved label: {label_file.name}")
            else:
                print(f"  -> {img_file.name} NOT considered added today.")

# Process train and val directories
move_new_files(IMAGES_TRAIN_DIR, LABELS_TRAIN_DIR)
move_new_files(IMAGES_VAL_DIR, LABELS_VAL_DIR)

print("Done moving new files to batch2.") 