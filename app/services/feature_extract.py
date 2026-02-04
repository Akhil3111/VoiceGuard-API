import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class FeatureExtractionError(Exception):
    pass

def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Extracts acoustic signals. Optimized for 8kHz sample rate.
    """
    try:
        features = {}

        # 1. MFCCs (Timbre)
        # 13 coefficients is standard, works fine at 8kHz
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1).tolist()
        features['mfcc_std'] = np.std(mfcc, axis=1).tolist()
        
        # 2. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features['spectral_flatness'] = float(np.mean(flatness))

        # 3. Pitch (F0)
        # Important: fmax cannot exceed Nyquist frequency (sr/2 = 4000Hz)
        # We stick to human voice range (50-500Hz) which is safe.
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C5'), # ~523Hz
            sr=sr
        )
        
        valid_f0 = f0[~np.isnan(f0)]
        
        if len(valid_f0) > 0:
            features['pitch_mean'] = float(np.mean(valid_f0))
            features['pitch_std'] = float(np.std(valid_f0))
            features['pitch_range'] = float(np.max(valid_f0) - np.min(valid_f0))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0

        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        features['zcr_std'] = float(np.std(zcr))
        
        # 5. RMS Energy
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = float(np.mean(rms))

        return features

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}", exc_info=True)
        raise FeatureExtractionError(f"Failed to analyze audio signals: {str(e)}")
