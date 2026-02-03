import os
import logging
import requests
import tempfile
import librosa
import numpy as np

# Configure module-level logger
logger = logging.getLogger(__name__)

# Constants for Safety & Performance
MAX_AUDIO_DURATION_SEC = 30
TARGET_SAMPLE_RATE = 16000
DOWNLOAD_TIMEOUT_SEC = 5
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB limit to prevent DoS

class AudioProcessingError(Exception):
    """Custom exception for expected audio processing failures."""
    pass

def download_and_load_audio(url: str) -> tuple[np.ndarray, int]:
    """
    Downloads audio from a URL, sanitizes it, and loads it into a NumPy array.
    
    Steps:
    1. Validate URL and download with timeout/size limits.
    2. Save to a secure temporary file.
    3. Load via Librosa (force 16kHz, Mono).
    4. Trim silence and cap duration at 30 seconds.
    
    Returns:
        tuple: (audio_time_series, sample_rate)
    
    Raises:
        AudioProcessingError: For download failures, corruption, or silent files.
    """
    temp_path = None
    try:
        # --- Step 1: Secure Download ---
        logger.info(f"Initiating download from: {url}")
        
        # Stream=True allows us to check headers/size before downloading the whole body
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
            response.raise_for_status()
            
            # content-length check (if provided by server)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
                raise AudioProcessingError(f"File too large. Limit is {MAX_FILE_SIZE_BYTES/1024/1024}MB.")

            # Create a temporary file
            # delete=False is required for compatibility with Windows, though we unlink manually
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp_file:
                temp_path = tmp_file.name
                downloaded_size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > MAX_FILE_SIZE_BYTES:
                            raise AudioProcessingError("File exceeded size limit during download.")
                        tmp_file.write(chunk)

        # --- Step 2: Load & Sanitize ---
        # Load with fixed Sample Rate (16kHz) and Mono to standardize input for ML
        try:
            y, sr = librosa.load(
                temp_path, 
                sr=TARGET_SAMPLE_RATE, 
                mono=True, 
                duration=MAX_AUDIO_DURATION_SEC # Cap duration immediately
            )
        except Exception as e:
            # Librosa/Soundfile failed to read the file (e.g., corrupted header, not audio)
            raise AudioProcessingError(f"Corrupted or unsupported audio format: {str(e)}")

        if len(y) == 0:
            raise AudioProcessingError("Audio file is empty.")

        # --- Step 3: Post-Processing ---
        # Trim leading/trailing silence (common in voicemail/recordings)
        # top_db=20 is a conservative threshold for silence
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        if len(y_trimmed) < (0.5 * sr):
             # Reject if audio is less than 0.5 seconds after trimming
            raise AudioProcessingError("Audio is too short for analysis (< 0.5s).")

        return y_trimmed, sr

    except requests.RequestException as e:
        raise AudioProcessingError(f"Network error during download: {str(e)}")
    except Exception as e:
        # Catch-all for unexpected errors to ensure function interface is stable
        if isinstance(e, AudioProcessingError):
            raise
        logger.error(f"Unexpected error in audio_loader: {e}", exc_info=True)
        raise AudioProcessingError("Internal processing error.")
    finally:
        # Always clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
