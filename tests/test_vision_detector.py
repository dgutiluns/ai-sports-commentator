import cv2
from src.vision.detector import VisionDetector


def main():
    # Path to your test image (update this to your actual image path)
    image_path = "data/detector_test_images/ball_detection1.jpg"
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return

    # Initialize the vision detector for soccer
    detector = VisionDetector(sport="soccer")

    # Run detection
    result = detector.detect(frame)

    # Print the structured output
    print("Detection result:")
    print(result)

if __name__ == "__main__":
    main() 