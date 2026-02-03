import os
from fastapi import Header, HTTPException, status

# In production, this would come from app.config or .env
API_KEY_SECRET = os.getenv("VOICE_GUARD_API_KEY", "hackathon-secret-key-123")

async def verify_api_key(x_api_key: str = Header(..., description="API Key for authentication")):
    """
    Validates the 'x-api-key' header.
    Rejects requests immediately if the key is missing or invalid.
    """
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return x_api_key
