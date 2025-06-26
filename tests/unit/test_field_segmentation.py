import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import cv2
import numpy as np
from src.vision.field_segmentation import segment_field

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'field_segmentation_log.txt')

def test_field_segmentation():
    """Test field segmentation on a single image."""
    log_lines = []
    image_path = "data/detector_test_images/ball_detection1.jpg"
    frame = cv2.imread(image_path)
    mask = segment_field(frame)
    field_pixels = np.count_nonzero(mask)
    log_lines.append(f"Field segmentation: {field_pixels} field pixels detected.")
    # Overlay mask on image
    overlay = frame.copy()
    overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0,255,0])).astype(np.uint8)
    out_path = os.path.join(OUTPUT_DIR, "field_segmentation_result.jpg")
    cv2.imwrite(out_path, overlay)
    log_lines.append(f"Field segmentation result saved as {out_path}")
    assert field_pixels > 0, "Field segmentation failed (no field detected)"
    log_lines.append("Field segmentation passed.")
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            print(line)
            f.write(line + '\n')
    print(f"\nLog saved to {LOG_PATH}")

def test_field_segmentation_on_video():
    """Test field segmentation on video frames with detailed logging and visualization."""
    log_lines = []
    log_lines.append("=== Field Segmentation Video Test ===")
    log_lines.append("")
    
    # Video path
    video_path = "data/pipelineV1/ROMVER.mp4"
    log_lines.append(f"Processing video: {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_lines.append(f"ERROR: Could not open video {video_path}")
        with open(LOG_PATH, 'w') as f:
            for line in log_lines:
                f.write(line + '\n')
        return
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_pixels = width * height
    
    log_lines.append(f"Video properties: {total_frames} frames, {fps} fps, {width}x{height}")
    log_lines.append(f"Total pixels per frame: {total_pixels}")
    log_lines.append("")
    
    # Create subdirectories for organized output
    masks_dir = os.path.join(OUTPUT_DIR, "field_masks")
    overlays_dir = os.path.join(OUTPUT_DIR, "field_overlays")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Statistics tracking
    frame_stats = []
    processed_frames = 0
    total_field_pixels = 0
    
    # Process frames (sample every 10th frame to keep test manageable)
    frame_skip = 10
    log_lines.append(f"Processing every {frame_skip}th frame for efficiency...")
    log_lines.append("")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Only process every nth frame
        if frame_idx % frame_skip != 0:
            frame_idx += 1
            continue
        
        # Run field segmentation
        try:
            field_mask = segment_field(frame)
            field_pixels = np.count_nonzero(field_mask)
            field_percentage = (field_pixels / total_pixels) * 100
            
            # Store statistics
            frame_stats.append({
                'frame': frame_idx,
                'field_pixels': field_pixels,
                'field_percentage': field_percentage,
                'total_pixels': total_pixels
            })
            
            total_field_pixels += field_pixels
            processed_frames += 1
            
            # Log frame results
            log_lines.append(f"Frame {frame_idx}: {field_pixels} field pixels ({field_percentage:.1f}% coverage)")
            
            # Save field mask (binary image)
            mask_filename = f"frame_{frame_idx:04d}_mask.png"
            mask_path = os.path.join(masks_dir, mask_filename)
            cv2.imwrite(mask_path, field_mask * 255)  # Convert to 0-255 range
            
            # Create overlay visualization
            overlay = frame.copy()
            # Apply green tint to field areas
            overlay[field_mask > 0] = (0.5 * overlay[field_mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
            
            # Add text overlay with statistics
            cv2.putText(overlay, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(overlay, f"Field: {field_pixels} pixels ({field_percentage:.1f}%)", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Save overlay
            overlay_filename = f"frame_{frame_idx:04d}_overlay.jpg"
            overlay_path = os.path.join(overlays_dir, overlay_filename)
            cv2.imwrite(overlay_path, overlay)
            
        except Exception as e:
            log_lines.append(f"ERROR in frame {frame_idx}: {str(e)}")
        
        frame_idx += 1
        
        # Limit to first 50 processed frames for quick testing
        if processed_frames >= 50:
            break
    
    # Cleanup
    cap.release()
    
    # Summary statistics
    log_lines.append("")
    log_lines.append("=== Field Segmentation Summary ===")
    log_lines.append(f"Total frames processed: {processed_frames}")
    log_lines.append(f"Total field pixels detected: {total_field_pixels}")
    log_lines.append(f"Average field pixels per frame: {total_field_pixels/processed_frames:.0f}")
    log_lines.append(f"Average field coverage: {(total_field_pixels/(processed_frames*total_pixels))*100:.1f}%")
    
    # Frame-by-frame statistics
    log_lines.append("")
    log_lines.append("=== Detailed Frame Statistics ===")
    for stat in frame_stats:
        log_lines.append(f"Frame {stat['frame']}: {stat['field_pixels']} pixels ({stat['field_percentage']:.1f}%)")
    
    # Output file paths
    log_lines.append("")
    log_lines.append("=== Output Files ===")
    log_lines.append(f"Field masks saved to: {masks_dir}")
    log_lines.append(f"Field overlays saved to: {overlays_dir}")
    log_lines.append(f"Detailed log saved to: {LOG_PATH}")
    
    # Validation
    log_lines.append("")
    log_lines.append("=== Validation ===")
    if processed_frames > 0:
        avg_coverage = (total_field_pixels/(processed_frames*total_pixels))*100
        if avg_coverage > 10:  # Expect at least 10% field coverage
            log_lines.append("✅ Field segmentation test PASSED - reasonable field coverage detected")
        else:
            log_lines.append("⚠️ Field segmentation test WARNING - low field coverage detected")
    else:
        log_lines.append("❌ Field segmentation test FAILED - no frames processed")
    
    # Save log
    with open(LOG_PATH, 'w') as f:
        for line in log_lines:
            print(line)
            f.write(line + '\n')
    
    print(f"\nLog saved to {LOG_PATH}")
    print(f"Field masks saved to {masks_dir}")
    print(f"Field overlays saved to {overlays_dir}")
    
    # Assertions for automated testing
    assert processed_frames > 0, "No frames were processed"
    assert total_field_pixels > 0, "No field pixels were detected"
    
    return frame_stats

if __name__ == "__main__":
    # Run both tests
    print("Running single image field segmentation test...")
    test_field_segmentation()
    
    print("\n" + "="*50)
    print("Running video-based field segmentation test...")
    test_field_segmentation_on_video() 