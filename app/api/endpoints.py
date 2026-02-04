import logging
from fastapi import APIRouter, Depends, HTTPException, status, Request
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
        # --- DEBUG LOGGING (Crucial for Hackathons) ---
        # This will show up in Render logs so you see EXACTLY what keys the tester sent
        incoming_data = request.model_dump(exclude_unset=True)
        logger.info(f"Incoming Request Keys: {list(incoming_data.keys())}")
        
        y, sr = None, None

        # --- LOGIC BRANCH ---
        if request.audio_base64:
            logger.info("Processing via Base64")
            y, sr = load_audio_from_base64(request.audio_base64, request.audio_format)
            
        elif request.audio_url:
            logger.info(f"Processing via URL: {request.audio_url}")
            y, sr = download_and_load_audio(str(request.audio_url))
            
        else:
            # If we get here, it means neither field was found.
            # The log above will tell us why (maybe they used a different key name?)
            error_msg = f"No audio found. Received keys: {list(incoming_data.keys())}. Expected 'audio_url' or 'audio_base64'."
            logger.warning(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # --- ML PIPELINE ---
        features = extract_features(y, sr)
        result = IntelligenceEngine.analyze_voice(features)

        return VoiceResponse(
            classification=result["classification"],
            confidence=result["confidence"],
            risk_level=result["risk_level"],
            language=request.language if request.language else result["language"],
            explanation=result["explanation"]
        )

    # --- ERROR HANDLERS ---
    except HTTPException as e:
        # PASSTHROUGH: Allows 400/401 errors to return cleanly to the user
        raise e

    except AudioProcessingError as e:
        logger.warning(f"Client Bad Audio: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    except FeatureExtractionError as e:
        logger.error(f"Analysis Failed: {str(e)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Could not extract features from audio.")

    except Exception as e:
        # Only catches unexpected crashes
        logger.error(f"System Crash: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")