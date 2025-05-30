"""
Configuration settings for the AI Commentator application.
"""

from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# Vision settings
VISION_CONFIG: Dict[str, Any] = {
    "model": "yolov8n.pt",  # Default YOLOv8 model
    "confidence_threshold": 0.5,
    "classes": {
        "soccer": [0, 32],  # 0 = person, 32 = sports ball in COCO
        "basketball": [0, 32]
    }
}

# NLP settings
NLP_CONFIG: Dict[str, Any] = {
    "model": "gpt-4",  # Default OpenAI model
    "temperature": 0.7,
    "max_tokens": 150,
    "commentary_styles": {
        "soccer": ["professional", "enthusiastic", "analytical"],
        "basketball": ["professional", "enthusiastic", "analytical"]
    }
}

# TTS settings
TTS_CONFIG: Dict[str, Any] = {
    "provider": "elevenlabs",
    "voice_id": "default",  # Will be configured per sport/style
    "stability": 0.5,
    "similarity_boost": 0.75
}

# Game engine settings
GAME_ENGINE_CONFIG: Dict[str, Any] = {
    "soccer": {
        "field_dimensions": (105, 68),  # meters
        "goal_dimensions": (7.32, 2.44),  # meters
        "penalty_area": (40.32, 16.5),  # meters
    },
    "basketball": {
        "court_dimensions": (28, 15),  # meters
        "hoop_height": 3.05,  # meters
        "three_point_line": 6.75,  # meters
    }
}

# Logging settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": BASE_DIR / "logs" / "ai_commentator.log",
        },
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
} 