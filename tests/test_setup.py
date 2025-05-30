#!/usr/bin/env python3
"""
Test script to verify the AI Commentator environment setup.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import openai
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_environment_variables():
    """Test if all required environment variables are set."""
    logger.info("Testing environment variables...")
    
    required_vars = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✓ All environment variables are set")
    return True

def test_openai_connection():
    """Test OpenAI API connection."""
    logger.info("Testing OpenAI API connection...")
    
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Try a simple API call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Say 'OpenAI connection successful'"}],
            max_tokens=10
        )
        logger.info("✓ OpenAI API connection successful")
        return True
    except Exception as e:
        logger.error(f"OpenAI API connection failed: {str(e)}")
        return False

def test_elevenlabs_connection():
    """Test ElevenLabs API connection."""
    logger.info("Testing ElevenLabs API connection...")
    
    try:
        # Initialize ElevenLabs client
        client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))
        
        # Get available voices
        available_voices = client.voices.get_all()
        if not available_voices.voices:
            raise Exception("No voices available")
            
        # Use the first available voice
        voice = available_voices.voices[0]
        
        # Generate audio
        audio = client.generate(
            text="ElevenLabs connection successful",
            voice=voice.voice_id,
            model="eleven_monolingual_v1"
        )
        logger.info("✓ ElevenLabs API connection successful")
        return True
    except Exception as e:
        logger.error(f"ElevenLabs API connection failed: {str(e)}")
        return False

def test_opencv():
    """Test OpenCV installation."""
    logger.info("Testing OpenCV installation...")
    
    try:
        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test_image.jpg", img)
        os.remove("test_image.jpg")
        logger.info("✓ OpenCV installation successful")
        return True
    except Exception as e:
        logger.error(f"OpenCV test failed: {str(e)}")
        return False

def main():
    """Run all tests."""
    # Load environment variables
    load_dotenv()
    
    # Run tests
    tests = [
        ("Environment Variables", test_environment_variables),
        ("OpenAI Connection", test_openai_connection),
        ("ElevenLabs Connection", test_elevenlabs_connection),
        ("OpenCV Installation", test_opencv)
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\nRunning {test_name} test...")
        result = test_func()
        results.append((test_name, result))
    
    # Print summary
    logger.info("\n=== Test Summary ===")
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n✓ All tests passed! Your environment is correctly set up.")
    else:
        logger.error("\n✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()