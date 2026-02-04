from pydantic import BaseModel, HttpUrl, Field, validator
from typing import Optional

class VoiceRequest(BaseModel):
    # Support BOTH the URL method (original) and Base64 method (Tester)
    audio_url: Optional[HttpUrl] = Field(None, description="Publicly accessible URL to the audio file")
    
    # New fields for the Endpoint Tester
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio string")
    audio_format: Optional[str] = Field(None, description="File format e.g., 'wav', 'mp3'")
    language: Optional[str] = Field(None, description="Input language (optional context)")
    message: Optional[str] = None

    @validator('audio_base64')
    def check_input_method(cls, v, values):
        # Ensure either URL or Base64 is provided
        if not v and not values.get('audio_url'):
            raise ValueError('You must provide either audio_url OR audio_base64')
        return v

class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    risk_level: str
    language: str
    explanation: str

class ErrorResponse(BaseModel):
    detail: str
