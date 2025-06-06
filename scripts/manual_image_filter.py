import cv2
import os
from pathlib import Path
import shutil
from tqdm import tqdm

def manual_filter_images(source_dir, output_dir):
    """
    Manually filter images by showing each one and letting user decide.
    Press 'y' to copy, 'n' to skip, 'q' to quit.
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files from the specific directory
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(Path(source_dir).glob(f"*{ext}")))
    
    print(f"Found {len(image_files)} images to review")
    print("\nControls:")
    print("  'y' - Copy image to output directory")
    print("  'n' - Skip image")
    print("  'q' - Quit program")
    print("\nPress any key to start...")
    
    # Create a window and wait for key press
    cv2.namedWindow('Image Review')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    copied_count = 0
    skipped_count = 0
    
    for img_path in image_files:
        # Read and display image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue
            
        # Resize if too large for screen
        max_height = 800
        if img.shape[0] > max_height:
            scale = max_height / img.shape[0]
            img = cv2.resize(img, None, fx=scale, fy=scale)
        
        # Show image with filename
        cv2.imshow('Image Review', img)
        print(f"\nReviewing: {img_path.name}")
        
        # Wait for key press
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('y'):
            # Copy image to output directory
            new_path = output_dir / img_path.name
            shutil.copy2(img_path, new_path)
            copied_count += 1
            print(f"Copied to: {new_path}")
        elif key == ord('n'):
            skipped_count += 1
            print("Skipped")
    
    cv2.destroyAllWindows()
    
    print(f"\nReview complete!")
    print(f"Copied: {copied_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Remaining: {len(image_files) - copied_count - skipped_count} images")

if __name__ == "__main__":
    # Ask user which set to process
    print("Which set do you want to process?")
    print("1. Training set")
    print("2. Validation set")
    choice = input("Enter 1 or 2: ")
    
    if choice == "1":
        source_dir = "data/soccer_ball/images/train"
        output_dir = "data/field_lines/images/train"
    else:
        source_dir = "data/soccer_ball/images/val"
        output_dir = "data/field_lines/images/val"
    
    # Start manual filtering
    manual_filter_images(source_dir, output_dir) 