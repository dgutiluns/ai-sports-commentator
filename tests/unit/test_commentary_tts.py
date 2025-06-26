import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.nlp.templates import event_to_commentary
from src.tts.gtts_speaker import save_tts_audio

def test_commentary_tts():
    event = {'event': 'goal', 'by_player': 7, 'frame': 10}
    text = event_to_commentary(event)
    print("Generated commentary:", text)
    audio_path = "test_goal_event.mp3"
    save_tts_audio(text, audio_path)
    assert os.path.exists(audio_path), "TTS audio file not created"
    file_size = os.path.getsize(audio_path)
    print(f"TTS audio file created at: {audio_path} (size: {file_size} bytes)")
    assert file_size > 0, "TTS audio file is empty"
    os.remove(audio_path)
    print("Commentary TTS test passed.")

if __name__ == "__main__":
    test_commentary_tts() 