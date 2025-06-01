import os
import random
from pathlib import Path

BATCH2_DIR = Path('data/soccer_ball/images/batch2')
TRAIN_DIR = Path('data/soccer_ball/images/batch2_train')
VAL_DIR = Path('data/soccer_ball/images/batch2_val')

TRAIN_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR.mkdir(parents=True, exist_ok=True)

# List all image files in batch2
glob_patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_files = []
for pattern in glob_patterns:
    image_files.extend(BATCH2_DIR.glob(pattern))

random.shuffle(image_files)

split_idx = int(0.8 * len(image_files))
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

for f in train_files:
    dest = TRAIN_DIR / f.name
    f.rename(dest)
    print(f"Moved {f.name} to {TRAIN_DIR}")

for f in val_files:
    dest = VAL_DIR / f.name
    f.rename(dest)
    print(f"Moved {f.name} to {VAL_DIR}")

print(f"Done splitting {len(image_files)} images: {len(train_files)} train, {len(val_files)} val.") 