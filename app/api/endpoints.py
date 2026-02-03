import logging
import base64
import tempfile
import os
import librosa
from fastapi import APIRouter, Depends, HTTPException, status
from app.schemas.io_models import VoiceRequest, VoiceResponse, ErrorResponse
from app.api.dependencies import verify_api_key
from app.services.audio_loader import download_and_load_audio, AudioProcessingError
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
    """
    Main pipeline:
    1. Download & Sanitize Audio
    2. Extract Acoustic Features
    3. Calculate Risk Score & Classification
    """
    try:
        logger.info(f"Received analysis request for URL: {request.audio_url}")

        # Step 1: Sanitization
        if request.audio_base64:
            try:
                audio_bytes = base64.b64decode(request.audio_base64)
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                
                try:
                    y, sr = librosa.load(tmp_path, sr=16000, mono=True)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid Base64 audio input"
                )

        elif request.audio_url:
            y, sr = download_and_load_audio(str(request.audio_url))

        else:
            raise HTTPException(
                status_code=422,
                detail="Either audio_url or audio_base64 must be provided"
            )

        # Step 2: Feature Extraction
        features = extract_features(y, sr)

        # Step 3: Intelligence & Decision
        result = IntelligenceEngine.analyze_voice(features)

        # Map internal dictionary to Pydantic Response (filters out _debug fields)
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

    except Exception as e:
        logger.error(f"Unhandled system error: {str(e)}", exc_info=True)
        # Never expose stack trace to user
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal analysis failed.")
