from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip, AudioArrayClip
import numpy as np

def overlay_audio_on_video(video_path: str, audio_clips: list, output_path: str):
    """
    Overlay audio clips on a video at specified timestamps.
    
    Args:
        video_path: Path to the input video file
        audio_clips: List of tuples (audio_path, timestamp)
        output_path: Path to save the output video
    """
    # Load the video
    video = VideoFileClip(video_path)
    
    # If no audio clips, create a silent track of the same duration
    if not audio_clips:
        print("No audio clips to overlay. Creating silent track...")
        fps = 44100  # Standard audio sample rate
        duration = video.duration
        silent_audio = AudioArrayClip(np.zeros((int(duration * fps), 1)), fps=fps)
        audio_clips = [(silent_audio, 0)]
    
    # Load and set start times for audio clips
    audio_tracks = []
    for audio_path, start_time in audio_clips:
        if isinstance(audio_path, str):
            audio = AudioFileClip(audio_path)
        else:
            audio = audio_path  # Already an AudioClip
        audio = audio.with_start(start_time)
        audio_tracks.append(audio)
    
    # Combine audio tracks
    final_audio = CompositeAudioClip(audio_tracks)
    
    # Set the audio of the video
    final_video = video.set_audio(final_audio)
    
    # Write the result
    final_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    
    # Close clips to free resources
    video.close()
    final_video.close()
    for audio in audio_tracks:
        audio.close() 