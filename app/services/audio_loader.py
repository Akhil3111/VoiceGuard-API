import os
import logging
import requests
import tempfile
import librosa
import numpy as np
import base64
import binascii

logger = logging.getLogger(__name__)

# --- OPTIMIZED SETTINGS ---
# Reduced to 10s and 8kHz to prevent "Request Timed Out" on free cloud servers
MAX_AUDIO_DURATION_SEC = 10 
TARGET_SAMPLE_RATE = 8000   
DOWNLOAD_TIMEOUT_SEC = 5
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # Limit to 5MB

class AudioProcessingError(Exception):
    pass

def _process_file(filepath: str) -> tuple[np.ndarray, int]:
    """Internal helper to load and sanitize audio from a local file."""
    try:
        # Optimization: Check file size first
        file_size = os.path.getsize(filepath)
        if file_size > MAX_FILE_SIZE_BYTES:
            raise AudioProcessingError(f"File too large ({file_size/1024/1024:.1f}MB). Limit is 5MB.")

        # Load with fixed Sample Rate (8kHz) and Mono
        # res_type='kaiser_fast' speeds up loading significantly
        y, sr = librosa.load(
            filepath, 
            sr=TARGET_SAMPLE_RATE, 
            mono=True, 
            duration=MAX_AUDIO_DURATION_SEC, # HARD CAP: Only load first 10s
            res_type='kaiser_fast' 
        )
        
        if len(y) == 0:
            raise AudioProcessingError("Audio file is empty.")

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Safety check: Audio must be at least 0.1s long
        if len(y_trimmed) < (0.1 * sr):
             raise AudioProcessingError("Audio is too short (< 0.1s).")

        return y_trimmed, sr
    except Exception as e:
        logger.warning(f"Audio Load Error: {e}")
        raise AudioProcessingError(f"Corrupted or unsupported audio: {str(e)}")

def download_and_load_audio(url: str) -> tuple[np.ndarray, int]:
    """Downloads audio from a URL and loads it."""
    temp_path = None
    try:
        logger.info(f"Downloading from URL: {url}")
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            response.raise_for_status()
            
            # Streaming download to avoid memory spikes
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                temp_path = tmp_file.name
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    downloaded += len(chunk)
                    if downloaded > MAX_FILE_SIZE_BYTES:
                         raise AudioProcessingError("File too large (>5MB).")
                    tmp_file.write(chunk)
        
        y, sr = _process_file(temp_path)
        return y, sr

    except Exception as e:
        if isinstance(e, AudioProcessingError): raise
        raise AudioProcessingError(f"Download failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

def load_audio_from_base64(b64_string: str, file_format: str = "wav") -> tuple[np.ndarray, int]:
    """Decodes a Base64 string and loads it as audio."""
    temp_path = None
    try:
        logger.info("Processing Base64 audio input")
        
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        try:
            file_data = base64.b64decode(b64_string)
        except binascii.Error:
            raise AudioProcessingError("Invalid Base64 string provided.")

        # Safety: Don't write huge files to disk
        if len(file_data) > MAX_FILE_SIZE_BYTES:
             raise AudioProcessingError("Base64 audio too large (>5MB).")

        ext = f".{file_format}" if file_format else ".wav"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(file_data)
            
        y, sr = _process_file(temp_path)
        return y, sr

    except Exception as e:
        if isinstance(e, AudioProcessingError): raise
        raise AudioProcessingError(f"Base64 processing failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
