# DEPRECATED: This module and merge_player_ids are deprecated for player tracking. Use ByteTrack for player tracking instead.
# If needed for ball logic, move to a ball-specific module.
import numpy as np

def merge_player_ids(player_tracks, max_dist=30):
    """
    player_tracks: list of dicts with 'frame', 'id', 'x', 'y', 'team'
    max_dist: max distance to consider two IDs as the same player
    Returns: list of dicts with merged IDs
    """
    id_map = {}
    next_id = 1
    merged_tracks = []
    for track in player_tracks:
        found = False
        for old_id, info in id_map.items():
            if (track['team'] == info['team'] and
                np.linalg.norm([track['x']-info['x'], track['y']-info['y']]) < max_dist):
                merged_id = old_id
                found = True
                break
        if not found:
            merged_id = next_id
            id_map[merged_id] = {'x': track['x'], 'y': track['y'], 'team': track['team']}
            next_id += 1
        track['id'] = merged_id
        merged_tracks.append(track)
    return merged_tracks 