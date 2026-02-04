import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.io_models import VoiceRequest, VoiceResponse, ErrorResponse
from app.api.dependencies import verify_api_key
from app.services.audio_loader import download_and_load_audio, load_audio_from_base64, AudioProcessingError
from app.services.feature_extract import extract_features, FeatureExtractionError
from app.services.scorer import IntelligenceEngine

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/analyze", 
    response_model=VoiceResponse, 
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Detect AI-Generated Voice"
)
async def analyze_audio_endpoint(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    try:
        # 1. Capture ALL data sent by the tester
        incoming_data = request.model_dump()
        logger.info(f"FULL PAYLOAD KEYS: {list(incoming_data.keys())}") # This will reveal the secret key name

        y, sr = None, None
        
        # 2. Smart Field Detection (Find the audio key automatically)
        # We look for 'audio_base64', OR 'audio_data', OR 'file', OR 'base64', etc.
        base64_candidate = (
            incoming_data.get("audio_base64") or 
            incoming_data.get("audioBase64") or   # Common camelCase variation
            incoming_data.get("audio_data") or    # Common variation
            incoming_data.get("file") or
            incoming_data.get("data")
        )

        # 3. Logic Branch
        if base64_candidate:
            logger.info("Found Base64 audio data!")
            # Use the format if provided, otherwise default to mp3/wav
            fmt = incoming_data.get("audio_format", "wav")
            y, sr = load_audio_from_base64(base64_candidate, fmt)
            
        elif request.audio_url:
            logger.info(f"Found URL: {request.audio_url}")
            y, sr = download_and_load_audio(str(request.audio_url))
            
        else:
            # If we still fail, the log above will tell us the exact keys available
            msg = f"No audio found. Available keys: {list(incoming_data.keys())}"
            logger.warning(msg)
            raise HTTPException(status_code=400, detail=msg)

        # 4. Pipeline
        features = extract_features(y, sr)
        result = IntelligenceEngine.analyze_voice(features)

        return VoiceResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            language=request.language if request.language else result["language"],
            explanation=result["explanation"]
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal analysis error")