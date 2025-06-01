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
        player_crop = frame[y1:y2, x1:x2]
        crops.append(player_crop)
        dominant_color = get_dominant_color(player_crop)
        dominant_colors.append(dominant_color)
        print("Dominant color for player:", dominant_color)  # Debug print for dominant color

    if len(dominant_colors) >= 2:
        dominant_colors_np = np.array(dominant_colors)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(dominant_colors_np)
        team_labels = kmeans.labels_  # 0 or 1 for each player
    else:
        # Not enough players to cluster, assign all to team_0
        team_labels = [0] * len(dominant_colors)

    # 3. Assign team label to each player
    for i, (bbox, tracker_id) in enumerate(zip(player_bboxes, tracker_ids)):
        team = f"team_{team_labels[i]}"
        number = detect_jersey_number(crops[i])
        x1, y1, x2, y2 = bbox
        players.append({
            "id": tracker_id,
            "x": (x1 + x2) // 2,
            "y": (y1 + y2) // 2,
            "team": team,
            "number": number
        })
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
        # Initialize ball detection model (our new trained model)
        self.ball_model = YOLO("runs/detect/model2/weights/best.pt")  # Using our newly trained model
        
        self.confidence_threshold = 0.2  # Lowered threshold for better detection in distant images
        self.ball_confidence_threshold = 0.15  # Lower threshold for ball detection to catch far-away balls
        self.target_classes = VISION_CONFIG["classes"][sport]
        
        # Initialize trackers
        self.tracker = sv.ByteTrack()
        self.box_annotator = sv.BoxAnnotator(
            thickness=2
        )
        
        logger.info(f"Initialized VisionDetector for {sport}")
    
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

        # Process player detections
        player_bboxes = []
        tracker_ids = []
        for i in range(len(player_detections)):
            bbox = player_detections.xyxy[i].astype(int)
            obj_id = player_detections.tracker_id[i] if player_detections.tracker_id is not None else i
            player_bboxes.append(bbox)
            tracker_ids.append(obj_id)

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

        # Enrich player data
        players = enrich_player_data(frame, player_bboxes, tracker_ids)

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