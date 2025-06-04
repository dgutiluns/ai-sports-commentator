import numpy as np
from scipy.optimize import linear_sum_assignment

def merge_player_ids_hungarian(player_tracks, iou_threshold=0.5, distance_threshold=100):
    """
    Merge player IDs across frames using the Hungarian algorithm.
    Args:
        player_tracks: List of player track dictionaries with 'frame', 'id', 'x', 'y', 'team'
        iou_threshold: IoU threshold for considering a match
        distance_threshold: Distance threshold for considering a match
    Returns:
        List of player track dictionaries with merged IDs
    """
    if not player_tracks:
        return player_tracks

    # Group tracks by frame
    frames = {}
    for track in player_tracks:
        frame = track['frame']
        if frame not in frames:
            frames[frame] = []
        frames[frame].append(track)

    # Sort frames
    frame_numbers = sorted(frames.keys())
    if not frame_numbers:
        return player_tracks

    # Initialize merged tracks with the first frame
    merged_tracks = frames[frame_numbers[0]].copy()
    current_ids = {track['id']: track['id'] for track in merged_tracks}

    # Process subsequent frames
    debug_prints = 0
    for i in range(1, len(frame_numbers)):
        prev_frame = frame_numbers[i-1]
        curr_frame = frame_numbers[i]
        
        prev_tracks = frames[prev_frame]
        curr_tracks = frames[curr_frame]

        if not prev_tracks or not curr_tracks:
            merged_tracks.extend(curr_tracks)
            continue

        # Compute cost matrix based on distance
        cost_matrix = np.zeros((len(prev_tracks), len(curr_tracks)))
        for j, prev_track in enumerate(prev_tracks):
            for k, curr_track in enumerate(curr_tracks):
                if prev_track['team'] != curr_track['team']:
                    cost = float('inf')
                else:
                    distance = np.sqrt((prev_track['x'] - curr_track['x'])**2 + 
                                     (prev_track['y'] - curr_track['y'])**2)
                    cost = distance
                    if distance > distance_threshold:
                        cost = float('inf')
                cost_matrix[j, k] = cost
                # Debug print for first two frame pairs
                if debug_prints < 2:
                    # Estimate meters if field is 100m x 60m (example)
                    field_width_px = 1000  # adjust if your field is different
                    field_height_px = 600
                    field_width_m = 100
                    field_height_m = 60
                    meter_x = (prev_track['x'] - curr_track['x']) * (field_width_m / field_width_px)
                    meter_y = (prev_track['y'] - curr_track['y']) * (field_height_m / field_height_px)
                    meter_dist = np.sqrt(meter_x**2 + meter_y**2)
                    print(f"[DEBUG] Frame {prev_frame}->{curr_frame} | Prev ID {prev_track['id']} (team {prev_track['team']}) to Curr ID {curr_track['id']} (team {curr_track['team']}): Field dist = {distance:.2f} px, Est. {meter_dist:.2f} m")
        if debug_prints < 2:
            print(f"[DEBUG] Cost matrix for frames {prev_frame}->{curr_frame}:")
            print(cost_matrix)
            debug_prints += 1

        # Skip assignment if cost matrix is all inf
        if np.all(np.isinf(cost_matrix)):
            merged_tracks.extend(curr_tracks)
            continue

        # Use Hungarian algorithm to find optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Update IDs based on assignment
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < distance_threshold:
                prev_id = prev_tracks[row]['id']
                curr_id = curr_tracks[col]['id']
                if prev_id in current_ids:
                    curr_tracks[col]['id'] = current_ids[prev_id]
                else:
                    current_ids[curr_id] = curr_id

        merged_tracks.extend(curr_tracks)

    return merged_tracks

def merge_player_ids(player_tracks, iou_threshold=0.5, distance_threshold=100):
    """
    Merge player IDs across frames using the Hungarian algorithm.
    Args:
        player_tracks: List of player track dictionaries with 'frame', 'id', 'x', 'y', 'team'
        iou_threshold: IoU threshold for considering a match
        distance_threshold: Distance threshold for considering a match
    Returns:
        List of player track dictionaries with merged IDs
    """
    return merge_player_ids_hungarian(player_tracks, iou_threshold, distance_threshold) 