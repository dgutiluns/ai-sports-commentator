from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip

def overlay_audio_on_video(video_path, audio_clips, output_path):
    """
    video_path: path to input video
    audio_clips: list of (audio_path, start_time) tuples
    output_path: path to save the output video
    """
    video = VideoFileClip(video_path)
    audio_tracks = [video.audio] if video.audio else []
    for audio_path, start_time in audio_clips:
        audio = AudioFileClip(audio_path).with_start(start_time)
        audio_tracks.append(audio)
    final_audio = CompositeAudioClip(audio_tracks)
    final_video = video.with_audio(final_audio)
    final_video.write_videofile(output_path, codec="libx264", audio_codec="aac") 