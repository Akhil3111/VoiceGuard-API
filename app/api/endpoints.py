import logging
import os
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.io_models import VoiceRequest, VoiceResponse, ErrorResponse
from app.api.dependencies import verify_api_key
from app.services.audio_loader import download_and_load_audio, AudioProcessingError
from app.services.feature_extract import extract_features, FeatureExtractionError
from app.services.scorer import IntelligenceEngine
from app.services.base64_loader import decode_base64_audio, Base64AudioError

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post(
    "/analyze", 
    response_model=VoiceResponse, 
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Detect AI-Generated Voice"
)
async def analyze_audio_endpoint(request: VoiceRequest, api_key: str = Depends(verify_api_key)):
    """
    Main pipeline:
    1. Decode Base64 Audio
    2. Extract Acoustic Features
    3. Calculate Risk Score & Classification
    """
    temp_audio_path = None
    try:
        logger.info(f"Received analysis request. Language: {request.language}, Format: {request.audio_format}")

        # Step 1: Decode Base64 -> Temp File
        try:
            temp_audio_path = decode_base64_audio(
                request.audio_base64, 
                suffix=f".{request.audio_format}"
            )
        except Base64AudioError as e:
             raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        # Step 2: Load & Sanitize Audio
        # We use download_and_load_audio which supports local paths now
        y, sr = download_and_load_audio(temp_audio_path)

        # Step 3: Feature Extraction
        features = extract_features(y, sr)

        # Step 4: Intelligence & Decision
        result = IntelligenceEngine.analyze_voice(features)

        # Map internal dictionary to Pydantic Response
        return VoiceResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            language=result["language"],
            explanation=result["explanation"]
        )

    except AudioProcessingError as e:
        logger.warning(f"Client error (Bad Audio): {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FeatureExtractionError as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Audio could not be analyzed. content may be too noisy or short.")

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Unhandled system error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal analysis failed.")
    finally:
        # Cleanup decoded temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)
