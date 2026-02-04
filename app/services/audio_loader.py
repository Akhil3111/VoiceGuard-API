import os
import logging
import requests
import tempfile
import librosa
import numpy as np
import base64
import binascii

logger = logging.getLogger(__name__)

# Constants
MAX_AUDIO_DURATION_SEC = 30
TARGET_SAMPLE_RATE = 16000
DOWNLOAD_TIMEOUT_SEC = 5
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

class AudioProcessingError(Exception):
    pass

def _process_file(filepath: str) -> tuple[np.ndarray, int]:
    """Internal helper to load and sanitize audio from a local file."""
    try:
        # Load with fixed Sample Rate (16kHz) and Mono
        y, sr = librosa.load(
            filepath, 
            sr=TARGET_SAMPLE_RATE, 
            mono=True, 
            duration=MAX_AUDIO_DURATION_SEC
        )
        
        if len(y) == 0:
            raise AudioProcessingError("Audio file is empty.")

        # Trim silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y_trimmed) < (0.5 * sr):
             raise AudioProcessingError("Audio is too short (< 0.5s).")

        return y_trimmed, sr
    except Exception as e:
        raise AudioProcessingError(f"Corrupted or unsupported audio: {str(e)}")

def download_and_load_audio(url: str) -> tuple[np.ndarray, int]:
    """Downloads audio from a URL and loads it."""
    temp_path = None
    try:
        logger.info(f"Downloading from URL: {url}")
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                temp_path = tmp_file.name
                for chunk in response.iter_content(chunk_size=8192):
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
        
        # 1. Clean the string (remove 'data:audio/mp3;base64,' header if present)
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]
            
        # 2. Decode Base64
        try:
            file_data = base64.b64decode(b64_string)
        except binascii.Error:
            raise AudioProcessingError("Invalid Base64 string provided.")

        # 3. Save to temp file
        # We use the provided format (e.g., .mp3, .wav) or default to .wav
        ext = f".{file_format}" if file_format else ".wav"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            temp_path = tmp_file.name
            tmp_file.write(file_data)
            
        # 4. Load using common helper
        y, sr = _process_file(temp_path)
        return y, sr

    except Exception as e:
        if isinstance(e, AudioProcessingError): raise
        raise AudioProcessingError(f"Base64 processing failed: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
