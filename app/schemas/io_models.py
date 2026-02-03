from pydantic import BaseModel, Field, HttpUrl
from typing import Optional

class VoiceRequest(BaseModel):
    # Mandatory for generic requests
    language: str = Field(..., description="Language of the audio")
    audio_format: str = Field(..., description="mp3 | wav")
    audio_base64: str = Field(..., description="Base64 encoded audio")
    message: Optional[str] = None
    
    # Keeping URL for compatibility if needed, but per instruction strictly Base64 is required now.
    # User's snippet shows ONLY Base64 fields in the updated class.
    # However, to avoid breaking any potential existing usage (though we just started),
    # I will strictly follow the user's snippet which makes these fields mandatory.
