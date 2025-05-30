"""
NLP commentator module for generating sports commentary.
"""

import logging
from typing import Dict, Any, List

import openai
from config.config import NLP_CONFIG

logger = logging.getLogger(__name__)

class NLPCommentator:
    """Generates natural language commentary for sports events."""
    
    def __init__(self, sport: str, style: str = "professional"):
        """Initialize the NLP commentator.
        
        Args:
            sport: The sport type (e.g., 'soccer', 'basketball')
            style: Commentary style (e.g., 'professional', 'enthusiastic', 'analytical')
        """
        self.sport = sport
        self.style = style
        self.model = NLP_CONFIG["model"]
        self.temperature = NLP_CONFIG["temperature"]
        self.max_tokens = NLP_CONFIG["max_tokens"]
        
        # Validate style
        if style not in NLP_CONFIG["commentary_styles"][sport]:
            raise ValueError(f"Invalid style '{style}' for {sport}")
        
        logger.info(f"Initialized NLPCommentator for {sport} with {style} style")
    
    def generate_commentary(self, game_state: Dict[str, Any]) -> str:
        """Generate commentary based on the current game state.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            Generated commentary text
        """
        prompt = self._create_prompt(game_state)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            commentary = response.choices[0].message.content
            logger.debug(f"Generated commentary: {commentary}")
            return commentary
            
        except Exception as e:
            logger.error(f"Error generating commentary: {str(e)}")
            return "Unable to generate commentary at this moment."
    
    def _create_prompt(self, game_state: Dict[str, Any]) -> str:
        """Create a prompt for the language model based on game state.
        
        Args:
            game_state: Current state of the game
            
        Returns:
            Formatted prompt string
        """
        # TODO: Implement more sophisticated prompt engineering
        return f"""
        Sport: {self.sport}
        Style: {self.style}
        Game State:
        - Ball Position: {game_state.get('ball_position', 'Unknown')}
        - Number of Players: {len(game_state.get('player_positions', []))}
        - Recent Events: {', '.join(game_state.get('game_events', ['None']))}
        
        Generate a brief, engaging commentary about the current game situation.
        """
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines the commentator's personality.
        
        Returns:
            System prompt string
        """
        style_descriptions = {
            "professional": "You are a professional sports commentator with years of experience. "
                          "Your commentary is clear, precise, and focused on the technical aspects of the game.",
            "enthusiastic": "You are an enthusiastic sports commentator who brings energy and excitement to the game. "
                          "Your commentary is dynamic and engaging, emphasizing the emotional aspects of the sport.",
            "analytical": "You are an analytical sports commentator who provides deep insights into the game. "
                         "Your commentary focuses on strategy, tactics, and statistical analysis."
        }
        
        return f"""
        You are an AI sports commentator specializing in {self.sport}.
        {style_descriptions[self.style]}
        
        Guidelines:
        1. Keep commentary concise and relevant to the current game state
        2. Use appropriate sports terminology
        3. Maintain a natural, flowing commentary style
        4. Focus on significant events and player actions
        5. Avoid repetition and maintain context
        """ 