from pydantic import BaseModel, HttpUrl, Field, validator, ConfigDict
from typing import Optional, Any

class VoiceRequest(BaseModel):
    # MODIFIED: Allow extra fields so we don't crash on unexpected keys like 'audioBase64Format'
    model_config = ConfigDict(extra='allow') 

    audio_url: Optional[HttpUrl] = Field(None)
    audio_base64: Optional[str] = Field(None)
    audio_format: Optional[str] = Field(None)
    language: Optional[str] = Field(None)
    message: Optional[str] = None

    @validator('audio_base64', check_fields=False)
    def check_input_method(cls, v, values):
        # MODIFIED: Relaxed validation to allow "hunting" for the key in endpoints.py
        return v

class VoiceResponse(BaseModel):
    classification: str
    confidence: float
    risk_level: str
    language: str
    explanation: str

class ErrorResponse(BaseModel):
    detail: str
