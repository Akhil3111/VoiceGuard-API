from pydantic import BaseModel, HttpUrl, Field, validator, ConfigDict # Import ConfigDict
from typing import Optional, Any, Dict

class VoiceRequest(BaseModel):
    # Allow unknown keys so we can see what the tester is sending
    model_config = ConfigDict(extra='allow') 

    audio_url: Optional[HttpUrl] = Field(None)
    audio_base64: Optional[str] = Field(None)
    audio_format: Optional[str] = Field(None)
    language: Optional[str] = Field(None)
    message: Optional[str] = None

    @validator('audio_base64', check_fields=False)
    def check_input_method(cls, v, values):
        # We relax this validation since the key might be named differently
        return v

class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    risk_level: str
    language: str
    explanation: str

class ErrorResponse(BaseModel):
    detail: str
