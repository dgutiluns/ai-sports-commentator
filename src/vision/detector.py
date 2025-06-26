"""
Vision detector module for object detection in sports videos.
"""

import logging
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import easyocr
from sklearn.cluster import KMeans
import os
from inference_sdk import InferenceHTTPClient
import tempfile

from config.config import VISION_CONFIG

logger = logging.getLogger(__name__)

# Define your team colors in BGR (OpenCV uses BGR, not RGB)
TEAM_COLORS = {
    "red": np.array([40, 40, 200]),   # Example: Red jersey
    "blue": np.array([200, 40, 40]),  # Example: Blue jersey
    # Add more as needed
}

reader = easyocr.Reader(['en'], gpu=False)  # Set gpu=True if you have a GPU

def get_dominant_color(image):
    pixels = image.reshape(-1, 3)
    k = 2  # number of teams
    _, labels, centers = cv2.kmeans(np.float32(pixels), k, None,
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    dominant = centers[np.argmax(np.bincount(labels.flatten()))]
    return dominant

def assign_team(dominant_color):
    min_dist = float('inf')
    assigned_team = None
    for team, color in TEAM_COLORS.items():
        dist = np.linalg.norm(dominant_color - color)
        if dist < min_dist:
            min_dist = dist
            assigned_team = team
    return assigned_team

def detect_jersey_number(player_crop):
    result = reader.readtext(player_crop)
    for (bbox, text, prob) in result:
        if prob > 0.5 and text.isdigit():
            return int(text)
    return None

# Example function to process detections and enrich player data
def enrich_player_data(frame, player_bboxes, tracker_ids):
    players = []
    dominant_colors = []
    crops = []

    # 1. Extract dominant color for each player
    for bbox in player_bboxes:
        x1, y1, x2, y2 = bbox
        # Convert to integers and ensure valid bounds
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(x1+1, min(x2, w))  # Ensure x2 > x1
        y2 = max(y1+1, min(y2, h))  # Ensure y2 > y1
        player_crop = frame[y1:y2, x1:x2]
        # Skip if crop is too small (edge case)
        if player_crop.size == 0:
            continue
        crops.append(player_crop)
        dominant_color = get_dominant_color(player_crop)
        dominant_colors.append(dominant_color)
        print("Dominant color for player:", dominant_color)  # Debug print for dominant color

    if len(dominant_colors) >= 2:
        dominant_colors_np = np.array(dominant_colors)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dominant_colors_np)
        team_labels = kmeans.labels_  # 0 or 1 for each player
    else:
        team_labels = [0] * len(dominant_colors)

    # 3. Assign team label to each player
    crop_idx = 0  # Track which crop we're using
    for i, (bbox, tracker_id) in enumerate(zip(player_bboxes, tracker_ids)):
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if crop_idx >= len(team_labels):
            continue
        team = f"team_{team_labels[crop_idx]}"
        number = detect_jersey_number(crops[crop_idx]) if crop_idx < len(crops) else None
        players.append({
            "id": tracker_id,
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2,
            "team": team,
            "number": number
        })
        crop_idx += 1
    return players

class VisionDetector:
    """Handles object detection and tracking in sports videos."""
    
    def __init__(self, sport: str):
        """Initialize the vision detector.
        
        Args:
            sport: The sport type (e.g., 'soccer', 'basketball')
        """
        self.sport = sport
        # Initialize player detection model (original YOLOv8)
        self.player_model = YOLO(VISION_CONFIG["model"])
        
        # Initialize ball detection model (our new trained model) with robust path resolution
        # Get the project root directory (assuming this file is in src/vision/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        ball_model_path = os.path.join(project_root, "runs", "detect", "model2", "weights", "best.pt")
        self.ball_model = YOLO(ball_model_path)  # Using our newly trained model
        
        self.confidence_threshold = 0.2  # Lowered threshold for better detection in distant images
        self.ball_confidence_threshold = 0.15  # Lower threshold for ball detection to catch far-away balls
        self.target_classes = VISION_CONFIG["classes"][sport]
        
        # Initialize trackers
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        
        logger.info(f"Initialized VisionDetector for {sport}")
        logger.info(f"Ball model loaded from: {ball_model_path}")
    
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect objects in the given frame and enrich player data."""
        # 1. Detect players using original model
        player_results = self.player_model(frame)
        try:
            player_detections = sv.Detections.from_ultralytics(player_results[0])
        except Exception as e:
            logger.error(f"Error in player detection: {e}")
            raise

        # Filter player detections by confidence and class
        player_mask = (player_detections.confidence > self.confidence_threshold) & \
                     (player_detections.class_id == 0)  # 0 = person
        player_detections = player_detections[player_mask]

        # 2. Detect ball using our new model
        ball_results = self.ball_model(frame)
        try:
            ball_detections = sv.Detections.from_ultralytics(ball_results[0])
        except Exception as e:
            logger.error(f"Error in ball detection: {e}")
            ball_detections = None

        # Track players
        player_detections = self.tracker.update_with_detections(player_detections)

        # Extract player bboxes and tracker IDs after tracking
        player_bboxes = player_detections.xyxy if len(player_detections) > 0 else np.array([])
        tracker_ids = player_detections.tracker_id if len(player_detections) > 0 else np.array([])

        # Enrich player data (team, number, feet position, etc.)
        players = enrich_player_data(frame, player_bboxes, tracker_ids)

        # Process ball detection
        ball = None
        if ball_detections is not None:
            ball_mask = (ball_detections.confidence > self.ball_confidence_threshold)
            ball_detections = ball_detections[ball_mask]
            
            if len(ball_detections) > 0:
                # Take the highest confidence ball detection
                best_ball_idx = np.argmax(ball_detections.confidence)
                bbox = ball_detections.xyxy[best_ball_idx].astype(int)
                x1, y1, x2, y2 = bbox
                ball = {"x": (x1 + x2) // 2, "y": (y1 + y2) // 2}

        return {
            "players": players,
            "ball": ball
        }
    
    def _annotate_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Annotate the frame with detection boxes and labels.
        
        Args:
            frame: Input video frame
            detections: Detection results
            
        Returns:
            Annotated frame
        """
        labels = [
            f"{self.target_classes[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]
        
        return self.box_annotator.annotate(
            scene=frame.copy(),
            detections=detections,
            labels=labels
        )
    
    def get_game_state(self, detections: sv.Detections) -> Dict[str, Any]:
        """Analyze detections to determine the current game state.
        
        Args:
            detections: Detection results
            
        Returns:
            Dictionary containing game state information
        """
        # TODO: Implement game state analysis based on detections
        return {
            'ball_position': None,
            'player_positions': [],
            'game_events': []
        } 

# --- Roboflow Integration for Player Detection ---
class RoboflowPlayerDetector:
    """Detects players using the Roboflow hosted API."""
    def __init__(self, model_id="football-players-detection-3zvbc/12"):
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not set in environment.")
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id

    def detect_players(self, frame):
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Empty frame encountered, skipping.")
            return []
        # Save frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        # Double-check file exists and is non-empty
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print(f"Temp image {temp_path} is missing or empty! Skipping frame.")
            return []
        # Optionally, add a tiny delay (uncomment if needed)
        # import time; time.sleep(0.01)
        try:
            result = self.client.infer(temp_path, model_id=self.model_id)
        finally:
            os.remove(temp_path)
        # Parse results: return list of [x1, y1, x2, y2, confidence]
        bboxes = []
        for pred in result.get("predictions", []):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            conf = pred.get("confidence", 1.0)
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            bboxes.append([x1, y1, x2, y2, conf])
        return bboxes

# --- Roboflow Integration for Ball Detection ---
class RoboflowBallDetector:
    """Detects balls using the Roboflow hosted API."""
    def __init__(self, model_id=None):
        # Default to a placeholder - user should update this with real model ID
        if model_id is None:
            model_id = "soccer-ball-detection-xyz/1"  # Placeholder - needs real model ID
        
        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not set in environment.")
        
        self.client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key
        )
        self.model_id = model_id
        logger.info(f"Initialized RoboflowBallDetector with model: {model_id}")

    def detect_balls(self, frame):
        """Detect balls in the given frame using Roboflow API.
        
        Args:
            frame: Input video frame as numpy array
            
        Returns:
            List of [x1, y1, x2, y2, confidence] for each detected ball
        """
        # Check if frame is valid
        if frame is None or frame.size == 0:
            print("Warning: Empty frame encountered, skipping.")
            return []
        
        # Save frame to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_path = temp_file.name
            cv2.imwrite(temp_path, frame)
            temp_file.flush()
            os.fsync(temp_file.fileno())
        
        # Double-check file exists and is non-empty
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            print(f"Temp image {temp_path} is missing or empty! Skipping frame.")
            return []
        
        try:
            result = self.client.infer(temp_path, model_id=self.model_id)
        except Exception as e:
            logger.error(f"Roboflow API error: {e}")
            return []
        finally:
            os.remove(temp_path)
        
        # Parse results: return list of [x1, y1, x2, y2, confidence]
        bboxes = []
        for pred in result.get("predictions", []):
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            conf = pred.get("confidence", 1.0)
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            bboxes.append([x1, y1, x2, y2, conf])
        return bboxes 