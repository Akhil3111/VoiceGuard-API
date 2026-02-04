import logging
import asyncio # MODIFIED: Added asyncio
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
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Detect AI-Generated Voice"
)
async def analyze_audio_endpoint(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    # MODIFIED: Yield control briefly to stabilize event loop on single-worker instance
    await asyncio.sleep(0.1)

    try:
        # 1. Capture payload keys for debugging
        incoming_data = request.model_dump()
        logger.info(f"FULL PAYLOAD KEYS: {list(incoming_data.keys())}")

        y, sr = None, None
        
        # 2. Smart Field Detection
        base64_candidate = (
            incoming_data.get("audio_base64") or 
            incoming_data.get("audioBase64") or 
            incoming_data.get("audio_data") or
            incoming_data.get("file") or
            incoming_data.get("data")
        )

        # 3. Logic Branch
        if base64_candidate:
            logger.info("Found Base64 audio data!")
            fmt = incoming_data.get("audio_format", "wav")
            y, sr = load_audio_from_base64(base64_candidate, fmt)
            
        elif request.audio_url:
            logger.info(f"Found URL: {request.audio_url}")
            y, sr = download_and_load_audio(str(request.audio_url))
            
        else:
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

    # Error Handlers
    except AudioProcessingError as e:
        logger.warning(f"Client Bad Audio: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FeatureExtractionError as e:
        logger.error(f"Analysis Failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Could not extract features.")

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"System Crash: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal analysis error")