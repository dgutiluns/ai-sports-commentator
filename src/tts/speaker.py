"""
Text-to-speech module for generating voice commentary.
"""

import logging
from typing import Optional
import tempfile
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from elevenlabs import generate, set_api_key
from config.config import TTS_CONFIG

logger = logging.getLogger(__name__)

class TTSSpeaker:
    """Handles text-to-speech conversion for commentary."""
    
    def __init__(self, sport: str, style: str = "professional"):
        """Initialize the TTS speaker.
        
        Args:
            sport: The sport type (e.g., 'soccer', 'basketball')
            style: Commentary style (e.g., 'professional', 'enthusiastic', 'analytical')
        """
        self.sport = sport
        self.style = style
        self.provider = TTS_CONFIG["provider"]
        self.voice_id = self._get_voice_id()
        self.stability = TTS_CONFIG["stability"]
        self.similarity_boost = TTS_CONFIG["similarity_boost"]
        
        # Set up ElevenLabs API key
        api_key = self._get_api_key()
        if api_key:
            set_api_key(api_key)
        else:
            logger.warning("No ElevenLabs API key found. TTS will not work.")
        
        logger.info(f"Initialized TTSSpeaker for {sport} with {style} style")
    
    def speak(self, text: str, save_path: Optional[Path] = None) -> bool:
        """Convert text to speech and play it.
        
        Args:
            text: Text to convert to speech
            save_path: Optional path to save the audio file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate audio
            audio = generate(
                text=text,
                voice=self.voice_id,
                model="eleven_monolingual_v1",
                stability=self.stability,
                similarity_boost=self.similarity_boost
            )
            
            # Save to temporary file if no save path provided
            if save_path is None:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = Path(temp_file.name)
                    temp_file.write(audio)
            else:
                temp_path = save_path
                temp_path.write_bytes(audio)
            
            # Play audio
            data, samplerate = sf.read(str(temp_path))
            sd.play(data, samplerate)
            sd.wait()
            
            # Clean up temporary file if we created one
            if save_path is None:
                temp_path.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Error in text-to-speech conversion: {str(e)}")
            return False
    
    def _get_voice_id(self) -> str:
        """Get the appropriate voice ID based on sport and style.
        
        Returns:
            Voice ID string
        """
        # TODO: Implement voice selection based on sport and style
        return TTS_CONFIG["voice_id"]
    
    def _get_api_key(self) -> Optional[str]:
        """Get the ElevenLabs API key from environment variables.
        
        Returns:
            API key string if found, None otherwise
        """
        import os
        return os.getenv("ELEVENLABS_API_KEY") 