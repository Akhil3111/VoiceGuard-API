import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class FeatureExtractionError(Exception):
    pass

def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Extracts acoustic signals required for the 3-Signal Intelligence Engine.
    
    Signals Extracted:
    1. Timbre (MFCCs) -> For Acoustic Model (Signal A)
    2. Pitch Stability (F0) -> For Temporal Analysis (Signal B)
    3. Noise/Roughness (ZCR) -> For Imperfection Analysis (Signal C)
    
    Args:
        y (np.ndarray): Audio time series (mono).
        sr (int): Sample rate.
        
    Returns:
        dict: A dictionary of scalar features.
    """
    try:
        features = {}

        # --- 1. Spectral Features (Timbre) ---
        # MFCCs: Capture the "shape" of the vocal tract.
        # We take 13 coefficients (standard for speech).
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate statistics (Mean & Variance) to flatten the time series
        # axis=1 performs calculation across time
        features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist() # Vector of 13 floats
        features['mfcc_std'] = np.std(mfcc, axis=1).tolist()   # Vector of 13 floats
        
        # Spectral Flatness: AI voices often have higher flatness (whiter noise spectrum)
        flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness'] = float(np.mean(flatness))

        # --- 2. Temporal Features (Pitch/Prosody) ---
        # F0 (Fundamental Frequency): Detects pitch contour.
        # fmin=50, fmax=300 covers standard human speech range.
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C5'),
            sr=sr
        )
        
        # Clean pitch data: Remove NaNs (unvoiced segments like silence/breaths)
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            features['pitch_mean'] = float(np.mean(valid_f0))
            features['pitch_std'] = float(np.std(valid_f0))
            features['pitch_range'] = float(np.max(valid_f0) - np.min(valid_f0))
        else:
            # Fallback for unvoiced audio (whispering)
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0

        # --- 3. Environmental Features (Imperfection) ---
        # Zero Crossing Rate: Proxy for noisiness and fricatives
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # RMS Energy: Loudness dynamics
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = float(np.mean(rms))

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to analyze audio signals: {str(e)}")
