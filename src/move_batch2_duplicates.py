import os
import shutil
from pathlib import Path

# Paths
BATCH2_DIR = Path('data/soccer_ball/images/batch2')
TRAIN_IMG_DIR = Path('data/soccer_ball/images/train')
VAL_IMG_DIR = Path('data/soccer_ball/images/val')
TRAIN_LABEL_DIR = Path('data/soccer_ball/labels/train')
VAL_LABEL_DIR = Path('data/soccer_ball/labels/val')
BATCH2_LABELS = BATCH2_DIR  # .txt files are in batch2 for now

# Read duplicate filenames (without path)
duplicate_files = set()
with open('/tmp/duplicate_images.txt', 'r') as f:
    for line in f:
        duplicate_files.add(line.strip())

# Helper to find original location (train or val)
def find_original_label_dir(filename):
    if (TRAIN_IMG_DIR / filename).exists():
        return TRAIN_LABEL_DIR
    elif (VAL_IMG_DIR / filename).exists():
        return VAL_LABEL_DIR
    else:
        return None

for filename in duplicate_files:
    img_path = BATCH2_DIR / filename
    label_path = BATCH2_LABELS / (Path(filename).stem + '.txt')
    # 1. Move .txt back if it exists
    if label_path.exists():
        orig_label_dir = find_original_label_dir(filename)
        if orig_label_dir:
            orig_label_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label_path), str(orig_label_dir / label_path.name))
            print(f"Moved label {label_path.name} back to {orig_label_dir}")
    # 2. Rename image (and .txt if it exists)
    if img_path.exists():
        new_img_name = img_path.stem + '_b2' + img_path.suffix
        new_img_path = BATCH2_DIR / new_img_name
        img_path.rename(new_img_path)
        print(f"Renamed {img_path.name} to {new_img_name}")
        # If .txt was not moved (e.g., doesn't exist), rename it if present
        batch2_label = BATCH2_LABELS / (img_path.stem + '.txt')
        if batch2_label.exists():
            new_label_name = img_path.stem + '_b2.txt'
            batch2_label.rename(BATCH2_LABELS / new_label_name)
            print(f"Renamed label {batch2_label.name} to {new_label_name}")

print("Done processing duplicates in batch2.") 