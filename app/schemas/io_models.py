from pydantic import BaseModel, HttpUrl, Field
from typing import Optional

class VoiceRequest(BaseModel):
    # Tester fields
    language: Optional[str] = None
    audio_format: Optional[str] = None
    audio_base64: Optional[str] = None

    # Your original field (keep it)
    audio_url: Optional[HttpUrl] = None
    message: Optional[str] = None

class VoiceResponse(BaseModel):
    classification: str = Field(..., description="Final decision: AI-generated | Human-generated | unknown")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    risk_level: str = Field(..., description="Low | Medium | High")
    language: str = Field(..., description="Detected language or 'Detected (Agnostic)'")
    explanation: str = Field(..., description="Human-readable reason for the classification")

class ErrorResponse(BaseModel):
    detail: str
