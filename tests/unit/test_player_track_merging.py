import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.player_tracking import merge_player_ids

def test_player_track_merging():
    tracks = [
        {'id': 1, 'x': 100, 'y': 200, 'frame': 0},
        {'id': 2, 'x': 102, 'y': 202, 'frame': 1},
        {'id': 3, 'x': 300, 'y': 400, 'frame': 0},
        {'id': 4, 'x': 305, 'y': 405, 'frame': 1}
    ]
    print("Original tracks:")
    for t in tracks:
        print(f"  {t}")
    merged = merge_player_ids(tracks, distance_threshold=7)
    print("Merged tracks:")
    for t in merged:
        print(f"  {t}")
    assert any(t['id'] == 1 for t in merged) and any(t['id'] == 3 for t in merged), "Track merging failed"
    print("Player track merging test passed.")

if __name__ == "__main__":
    test_player_track_merging() 