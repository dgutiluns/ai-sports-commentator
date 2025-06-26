import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import cv2
import json
from pathlib import Path
from src.vision.homography_auto import HomographyEstimator
from src.vision.field_segmentation import segment_field

TEST_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs', TEST_NAME)
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_DIR, 'homography_log.txt')

def test_homography_estimation():
    """Test homography estimation with dummy mask."""
    FIELD_COORDS = np.array([
        [0, 0],
        [0, 68],
        [105, 68],
        [105, 0]
    ], dtype=np.float32)
    # Dummy mask: just a white rectangle
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[100:600, 200:1000] = 255
    estimator = HomographyEstimator(FIELD_COORDS, smoothing=0)
    H = estimator.estimate(mask)
    print("Estimated homography matrix:\n", H)
    assert H is not None, "Homography estimation failed (no matrix returned)"
    # Test mapping a point
    x, y = estimator.map_point(300, 200, H)
    print(f"Mapped point (300,200) -> ({x:.2f}, {y:.2f})")
    assert x is not None and y is not None, "Homography mapping failed"
    assert isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating)), "Homography mapping returned invalid types"
    print("Homography estimation and mapping passed.")

def test_homography_with_field_segmentation():
    """Test homography estimation using field segmentation on video frames."""
    print("Starting homography integration test with field segmentation...")
    
    # Initialize logging
    log_lines = []
    log_lines.append("=== Homography Integration Test with Field Segmentation ===")
    log_lines.append(f"Video: data/pipelineV1/ROMVER.mp4")
    log_lines.append("=" * 60)
    
    # Create output directories
    homography_dir = os.path.join(OUTPUT_DIR, 'homography_matrices')
    overlay_dir = os.path.join(OUTPUT_DIR, 'field_overlays')
    warped_dir = os.path.join(OUTPUT_DIR, 'warped_grids')
    
    os.makedirs(homography_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(warped_dir, exist_ok=True)
    
    # Define field coordinates (standard soccer field dimensions in meters)
    FIELD_COORDS = np.array([
        [0, 0],      # Top-left
        [0, 68],     # Bottom-left  
        [105, 68],   # Bottom-right
        [105, 0]     # Top-right
    ], dtype=np.float32)
    
    # Initialize homography estimator
    estimator = HomographyEstimator(FIELD_COORDS, smoothing=0.1)
    
    # Open video
    video_path = "data/pipelineV1/ROMVER.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        log_lines.append("ERROR: Could not open video file")
        with open(LOG_PATH, 'w') as f:
            f.write('\n'.join(log_lines))
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    log_lines.append(f"Video Properties:")
    log_lines.append(f"  - Total frames: {total_frames}")
    log_lines.append(f"  - FPS: {fps}")
    log_lines.append(f"  - Resolution: {width}x{height}")
    log_lines.append("")
    
    # Process every 10th frame for efficiency
    sample_interval = 10
    processed_frames = 0
    successful_homographies = 0
    
    frame_results = []
    
    for frame_idx in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        processed_frames += 1
        log_lines.append(f"Processing frame {frame_idx}...")
        
        # Step 1: Field segmentation
        try:
            field_mask = segment_field(frame)
            field_pixels = np.sum(field_mask > 0)
            field_percentage = (field_pixels / (width * height)) * 100
            
            log_lines.append(f"  - Field pixels: {field_pixels:,} ({field_percentage:.1f}%)")
            
            # Skip if field coverage is too low
            if field_percentage < 30:
                log_lines.append(f"  - SKIPPED: Field coverage too low (< 30%)")
                frame_results.append({
                    "frame": frame_idx,
                    "field_pixels": field_pixels,
                    "field_percentage": field_percentage,
                    "homography_success": False,
                    "reason": "low_field_coverage"
                })
                continue
                
        except Exception as e:
            log_lines.append(f"  - ERROR in field segmentation: {str(e)}")
            frame_results.append({
                "frame": frame_idx,
                "homography_success": False,
                "reason": f"field_segmentation_error: {str(e)}"
            })
            continue
        
        # Step 2: Homography estimation
        try:
            H = estimator.estimate(field_mask)
            
            if H is not None:
                successful_homographies += 1
                log_lines.append(f"  - Homography SUCCESS")
                
                # Save homography matrix
                matrix_path = os.path.join(homography_dir, f"homography_frame_{frame_idx:04d}.npy")
                np.save(matrix_path, H)
                
                # Create field overlay visualization
                overlay = frame.copy()
                overlay[field_mask > 0] = cv2.addWeighted(
                    overlay[field_mask > 0], 0.7, 
                    np.full_like(overlay[field_mask > 0], [0, 255, 0]), 0.3, 0
                )
                
                # Draw field corners if available
                try:
                    corners = estimator.get_field_corners(field_mask)
                    if corners is not None:
                        for i, corner in enumerate(corners):
                            cv2.circle(overlay, (int(corner[0]), int(corner[1])), 10, (0, 0, 255), -1)
                            cv2.putText(overlay, str(i), (int(corner[0])+15, int(corner[1])+5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except:
                    pass  # Corner detection might fail
                
                overlay_path = os.path.join(overlay_dir, f"overlay_frame_{frame_idx:04d}.jpg")
                cv2.imwrite(overlay_path, overlay)
                
                # Create warped grid visualization
                try:
                    warped_grid = estimator.create_warped_grid(H, (width, height))
                    if warped_grid is not None:
                        warped_path = os.path.join(warped_dir, f"warped_frame_{frame_idx:04d}.jpg")
                        cv2.imwrite(warped_path, warped_grid)
                except:
                    pass  # Grid warping might fail
                
                # Test point mapping
                test_x, test_y = estimator.map_point(width//2, height//2, H)
                log_lines.append(f"  - Center point ({width//2}, {height//2}) -> ({test_x:.2f}, {test_y:.2f})")
                
                frame_results.append({
                    "frame": frame_idx,
                    "field_pixels": field_pixels,
                    "field_percentage": field_percentage,
                    "homography_success": True,
                    "homography_matrix": H.tolist(),
                    "mapped_center": [test_x, test_y]
                })
                
            else:
                log_lines.append(f"  - Homography FAILED")
                frame_results.append({
                    "frame": frame_idx,
                    "field_pixels": field_pixels,
                    "field_percentage": field_percentage,
                    "homography_success": False,
                    "reason": "homography_estimation_failed"
                })
                
        except Exception as e:
            log_lines.append(f"  - ERROR in homography estimation: {str(e)}")
            frame_results.append({
                "frame": frame_idx,
                "field_pixels": field_pixels,
                "field_percentage": field_percentage,
                "homography_success": False,
                "reason": f"homography_error: {str(e)}"
            })
    
    cap.release()
    
    # Summary statistics
    log_lines.append("")
    log_lines.append("=== SUMMARY ===")
    log_lines.append(f"Total frames processed: {processed_frames}")
    log_lines.append(f"Successful homographies: {successful_homographies}")
    log_lines.append(f"Success rate: {(successful_homographies/processed_frames)*100:.1f}%")
    
    if successful_homographies > 0:
        successful_results = [r for r in frame_results if r.get("homography_success", False)]
        avg_field_percentage = np.mean([r["field_percentage"] for r in successful_results])
        log_lines.append(f"Average field coverage (successful): {avg_field_percentage:.1f}%")
    
    log_lines.append("")
    log_lines.append("=== OUTPUT FILES ===")
    log_lines.append(f"Homography matrices: {homography_dir}")
    log_lines.append(f"Field overlays: {overlay_dir}")
    log_lines.append(f"Warped grids: {warped_dir}")
    log_lines.append(f"Detailed log: {LOG_PATH}")
    
    # Save detailed log
    with open(LOG_PATH, 'w') as f:
        f.write('\n'.join(log_lines))
    
    # Save frame results as JSON
    results_path = os.path.join(OUTPUT_DIR, 'frame_results.json')
    # Convert all numpy types to native Python types for JSON serialization
    def to_python_type(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    frame_results_py = [
        {k: to_python_type(v) for k, v in r.items()} for r in frame_results
    ]
    with open(results_path, 'w') as f:
        json.dump(frame_results_py, f, indent=2)
    
    print(f"Homography integration test completed!")
    print(f"Processed {processed_frames} frames, {successful_homographies} successful homographies")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    # Run both tests
    print("Running basic homography test...")
    test_homography_estimation()
    print("\n" + "="*50 + "\n")
    test_homography_with_field_segmentation() 