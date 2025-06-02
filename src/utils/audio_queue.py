def queue_audio_clips(audio_clips, min_gap=1.0):
    """
    audio_clips: list of (audio_path, start_time) tuples (start_time in seconds)
    min_gap: minimum gap (in seconds) between clips
    Returns: list of (audio_path, scheduled_time) tuples
    """
    scheduled = []
    current_time = 0.0
    for audio_path, _ in audio_clips:
        scheduled.append((audio_path, current_time))
        # Get duration of audio file
        try:
            from moviepy import AudioFileClip
            duration = AudioFileClip(audio_path).duration
        except Exception:
            duration = 2.0  # fallback if can't read duration
        current_time += duration + min_gap
    return scheduled 