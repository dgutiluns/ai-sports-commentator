import numpy as np

def filter_and_smooth_ball_positions(ball_positions, max_jump=100, window=3):
    """
    ball_positions: list of dicts with 'frame', 'x', 'y' (may include None for missed detections)
    max_jump: max allowed distance between consecutive positions
    window: moving average window size
    Returns: list of filtered/smoothed positions (same length)
    """
    filtered = []
    last_valid = None
    for pos in ball_positions:
        if pos is None:
            filtered.append(None)
            continue
        if last_valid is not None:
            dist = np.linalg.norm([pos['x']-last_valid['x'], pos['y']-last_valid['y']])
            if dist > max_jump:
                filtered.append(None)
                continue
        filtered.append(pos)
        last_valid = pos
    # Moving average smoothing (ignoring None)
    smoothed = []
    for i in range(len(filtered)):
        window_vals = [p for p in filtered[max(0,i-window+1):i+1] if p is not None]
        if window_vals:
            avg_x = np.mean([p['x'] for p in window_vals])
            avg_y = np.mean([p['y'] for p in window_vals])
            smoothed.append({'frame': filtered[i]['frame'], 'x': avg_x, 'y': avg_y} if filtered[i] else None)
        else:
            smoothed.append(None)
    return smoothed 