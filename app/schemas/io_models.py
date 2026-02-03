from pydantic import BaseModel, Field
from typing import Optional


class VoiceRequest(BaseModel):
    language: str = Field(..., description="Language of the audio")
    audio_format: str = Field(..., description="mp3 | wav")
    audio_base64: str = Field(..., description="Base64 encoded audio")
    message: Optional[str] = None


class VoiceResponse(BaseModel):
    classification: str = Field(..., description="AI-generated | Human-generated")
    confidence: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., description="Low | Medium | High")
    language: str
    explanation: str


class ErrorResponse(BaseModel):
    detail: str
