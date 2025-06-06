from gtts import gTTS
import os
 
def save_tts_audio(text, audio_path):
    tts = gTTS(text)
    tts.save(audio_path)
    return audio_path 