"""
Event detector module for interpreting soccer game events from vision output.
"""

from typing import List, Dict, Any, Optional
import numpy as np

# Field and goal boundaries (example values, adjust as needed)
FIELD_X_MIN = 0
FIELD_X_MAX = 1000
FIELD_Y_MIN = 0
FIELD_Y_MAX = 600
GOAL_X_MIN = 0
GOAL_X_MAX = 10
GOAL_Y_MIN = 250
GOAL_Y_MAX = 350
GOAL2_X_MIN = 990
GOAL2_X_MAX = 1000
GOAL2_Y_MIN = 250
GOAL2_Y_MAX = 350

DRIBBLE_FRAMES = 2  # Number of consecutive frames for a dribble
DRIBBLE_MIN_DIST = 5  # Lowered for MVP testing

class EventDetector:
    """Detects basic soccer events (e.g., pass, turnover, dribble, shot, goal, out of bounds) from object positions."""
    def __init__(self):
        self.prev_ball_position: Optional[np.ndarray] = None
        self.prev_player_positions: Optional[List[Dict[str, Any]]] = None
        self.event_log: List[Dict[str, Any]] = []
        self.last_possessing_player: Optional[Dict[str, Any]] = None
        self.dribble_start_pos: Optional[np.ndarray] = None
        self.dribble_start_frame: Optional[int] = None
        self.turnover_counter = 0
        self.turnover_threshold = 15  # Require 15 frames of opponent being closest
        self.last_ball_velocity = None

    def update(self, frame_idx: int, detections: Dict[str, Any]) -> List[Dict[str, Any]]:
        print(f"\n[EventDetector] Frame {frame_idx} called.")
        events = []
        ball = detections.get('ball')
        players = detections.get('players', [])
        
        # Print debug info
        print(f"  Ball field coordinates: {ball if ball is not None else 'None'}")
        for p in players:
            print(f"  Player {p['id']} field coordinates: (x={p['x']:.2f}, y={p['y']:.2f}, team={p.get('team')})")

        # Skip event detection if ball is not detected or missing coordinates
        if ball is None or 'x' not in ball or 'y' not in ball:
            # Update state with None for ball position
            self.prev_ball_position = None
            self.prev_player_positions = players
            return events

        # Calculate ball velocity if we have previous position
        if self.prev_ball_position is not None:
            ball_velocity = np.array([ball['x'], ball['y']]) - self.prev_ball_position
            self.last_ball_velocity = ball_velocity
        else:
            self.last_ball_velocity = None

        # Out of bounds detection
        if not (FIELD_X_MIN <= ball['x'] <= FIELD_X_MAX and FIELD_Y_MIN <= ball['y'] <= FIELD_Y_MAX):
            events.append({
                'event': 'out_of_bounds',
                'by_player': self._closest_player(ball, players)['id'] if players else None,
                'frame': frame_idx
            })

        # Goal detection (refined)
        in_goal1 = (GOAL_X_MIN <= ball['x'] <= GOAL_X_MAX and GOAL_Y_MIN <= ball['y'] <= GOAL_Y_MAX)
        in_goal2 = (GOAL2_X_MIN <= ball['x'] <= GOAL2_X_MAX and GOAL2_Y_MIN <= ball['y'] <= GOAL2_Y_MAX)
        if in_goal1 or in_goal2:
            events.append({
                'event': 'goal',
                'by_player': self._closest_player(ball, players)['id'] if players else None,
                'frame': frame_idx
            })

        # Shot on goal detection
        if self.prev_ball_position is not None:
            ball_vec = np.array([ball['x'], ball['y']])
            speed = np.linalg.norm(ball_vec - self.prev_ball_position)
            shot_zone1 = (GOAL_X_MAX < ball['x'] < GOAL_X_MAX + 10 and GOAL_Y_MIN <= ball['y'] <= GOAL_Y_MAX)
            shot_zone2 = (GOAL2_X_MIN - 10 < ball['x'] < GOAL2_X_MIN and GOAL2_Y_MIN <= ball['y'] <= GOAL2_Y_MAX)
            if (shot_zone1 or shot_zone2) and speed > 5:
                events.append({
                    'event': 'shot_on_goal',
                    'by_player': self._closest_player(ball, players)['id'] if players else None,
                    'frame': frame_idx
                })

        # Team-aware pass and turnover detection with improved logic
        if self.prev_ball_position is not None:
            ball_movement = np.linalg.norm(np.array([ball['x'], ball['y']]) - self.prev_ball_position)
            if ball_movement > 5:
                curr_closest = self._closest_player(ball, players)
                prev_closest = self._closest_player_by_pos(self.prev_ball_position, self.prev_player_positions or [])
                
                if curr_closest and prev_closest and curr_closest['id'] != prev_closest['id']:
                    if curr_closest.get('team') == prev_closest.get('team'):
                        events.append({
                            'event': 'pass',
                            'from_player': prev_closest['id'],
                            'to_player': curr_closest['id'],
                            'team': curr_closest.get('team'),
                            'frame': frame_idx
                        })
                    else:
                        # Check if ball is moving toward the new player
                        if self._is_ball_moving_toward_player(ball, curr_closest):
                            self.turnover_counter += 1
                            if self.turnover_counter >= self.turnover_threshold:
                                events.append({
                                    'event': 'turnover',
                                    'from_player': prev_closest['id'],
                                    'to_player': curr_closest['id'],
                                    'from_team': prev_closest.get('team'),
                                    'to_team': curr_closest.get('team'),
                                    'frame': frame_idx
                                })
                                self.turnover_counter = 0
                        else:
                            self.turnover_counter = 0

        # Dribble detection
        curr_closest = self._closest_player(ball, players)
        if curr_closest:
            if self.last_possessing_player and curr_closest['id'] == self.last_possessing_player['id']:
                # Continue dribble
                if self.dribble_start_pos is not None and self.dribble_start_frame is not None:
                    dribble_dist = np.linalg.norm(np.array([ball['x'], ball['y']]) - self.dribble_start_pos)
                    dribble_frames = frame_idx - self.dribble_start_frame
                    if dribble_frames >= DRIBBLE_FRAMES and dribble_dist >= DRIBBLE_MIN_DIST:
                        events.append({
                            'event': 'dribble',
                            'by_player': curr_closest['id'],
                            'team': curr_closest.get('team'),
                            'distance': dribble_dist,
                            'frames': dribble_frames,
                            'frame': frame_idx
                        })
                        # Reset dribble start
                        self.dribble_start_pos = np.array([ball['x'], ball['y']])
                        self.dribble_start_frame = frame_idx
            else:
                # New possession
                self.dribble_start_pos = np.array([ball['x'], ball['y']])
                self.dribble_start_frame = frame_idx
                self.last_possessing_player = curr_closest
        else:
            self.dribble_start_pos = None
            self.dribble_start_frame = None
            self.last_possessing_player = None

        # Update state
        self.prev_ball_position = np.array([ball['x'], ball['y']])
        self.prev_player_positions = players
        self.event_log.extend(events)
        return events

    def _closest_player(self, ball: Dict[str, Any], players: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not players:
            return None
        ball_pos = np.array([ball['x'], ball['y']])
        dists = [np.linalg.norm(np.array([p['x'], p['y']]) - ball_pos) for p in players]
        return players[int(np.argmin(dists))]

    def _closest_player_by_pos(self, pos: np.ndarray, players: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not players:
            return None
        dists = [np.linalg.norm(np.array([p['x'], p['y']]) - pos) for p in players]
        return players[int(np.argmin(dists))]

    def _is_ball_moving_toward_player(self, ball: Dict[str, Any], player: Dict[str, Any]) -> bool:
        """Check if the ball's movement is consistent with a turnover."""
        if self.last_ball_velocity is None:
            return True

        # Calculate direction from ball to player
        direction_to_player = np.array([player['x'] - ball['x'], player['y'] - ball['y']])
        
        # Normalize vectors for dot product
        ball_velocity_norm = self.last_ball_velocity / np.linalg.norm(self.last_ball_velocity)
        direction_norm = direction_to_player / np.linalg.norm(direction_to_player)
        
        # Check if the ball is moving toward the player (dot product > 0)
        dot_product = np.dot(ball_velocity_norm, direction_norm)
        return dot_product > 0 