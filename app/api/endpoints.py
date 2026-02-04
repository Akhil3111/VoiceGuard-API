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
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Detect AI-Generated Voice"
)
async def analyze_audio_endpoint(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    try:
        y, sr = None, None

        # --- LOGIC BRANCH: URL vs BASE64 ---
        if request.audio_base64:
            # Handle the Tester's Base64 Format
            logger.info("Received Base64 input")
            y, sr = load_audio_from_base64(request.audio_base64, request.audio_format)
        elif request.audio_url:
            # Handle Standard URL Input
            logger.info(f"Received URL input: {request.audio_url}")
            y, sr = download_and_load_audio(str(request.audio_url))
        else:
            # Should be caught by Pydantic, but safety first
            raise HTTPException(status_code=400, detail="No audio provided.")

        # --- COMMON PIPELINE ---
        # 1. Feature Extraction
        features = extract_features(y, sr)

        # 2. Intelligence & Decision
        result = IntelligenceEngine.analyze_voice(features)

        return VoiceResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            language=request.language if request.language else result["language"], # Use provided language if available
            explanation=result["explanation"]
        )

    except AudioProcessingError as e:
        logger.warning(f"Bad Audio Input: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FeatureExtractionError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Audio content analysis failed.")

    except Exception as e:
        logger.error(f"Unhandled system error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal analysis failed.")
