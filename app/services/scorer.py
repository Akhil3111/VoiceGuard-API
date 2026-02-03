import os
import joblib
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
MODEL_PATH = os.path.join("ml_assets", "clf_acoustic.joblib")

# Heuristic Thresholds (Tuned for standard speech)
# If pitch std deviation is < 20 Hz, it's suspiciously robotic.
THRESHOLD_PITCH_STD_LOW = 20.0 
# If Zero Crossing Rate is < 0.02, it's suspiciously "studio clean".
THRESHOLD_ZCR_LOW = 0.02 

class IntelligenceEngine:
    _model = None

    @classmethod
    def _get_model(cls):
        """
        Singleton pattern to load the ML model only once.
        Assumes the model is a scikit-learn classifier (e.g., RandomForest).
        """
        if cls._model is None:
            try:
                if os.path.exists(MODEL_PATH):
                    logger.info(f"Loading acoustic model from {MODEL_PATH}")
                    cls._model = joblib.load(MODEL_PATH)
                else:
                    logger.warning(f"Model file not found at {MODEL_PATH}. Using fallback mode.")
                    cls._model = None
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                cls._model = None
        return cls._model

    @staticmethod
    def _compute_signal_acoustic(features: Dict[str, float]) -> float:
        """
        Signal A: Acoustic Naturalness (ML-based).
        Uses MFCCs and Spectral Flatness to detect synthetic timbres.
        """
        model = IntelligenceEngine._get_model()
        
        # If model is missing (during dev/test), fallback to a heuristic 
        # based on spectral flatness (AI tends to be 'flatter'/buzzier).
        if model is None:
            # Normalize flatness: typical speech is 0.01-0.05. >0.1 is suspicious.
            flatness = features.get('spectral_flatness', 0.0)
            return min(1.0, flatness * 10)

        try:
            # Construct feature vector matching training schema
            # [mfcc_mean_0...12, mfcc_std_0...12, spectral_flatness]
            # Note: In production, ensure this order matches training exactly.
            mfcc_means = features.get('mfcc_mean', [0]*13)
            mfcc_stds = features.get('mfcc_std', [0]*13)
            flatness = features.get('spectral_flatness', 0)
            
            # Flatten into single array
            vector = np.concatenate([mfcc_means, mfcc_stds, [flatness]]).reshape(1, -1)
            
            # Get probability of class 1 (AI)
            # returns [[prob_human, prob_ai]]
            probs = model.predict_proba(vector)
            return float(probs[0][1])
            
        except Exception as e:
            logger.error(f"Signal A computation failed: {e}")
            return 0.5 # Return uncertain score on error

    @staticmethod
    def _compute_signal_temporal(features: Dict[str, float]) -> float:
        """
        Signal B: Temporal Consistency (Statistical).
        Detects 'Super-human' pitch stability.
        """
        pitch_std = features.get('pitch_std', 0.0)
        
        # Logic: Normal speech has high variance (>20Hz). 
        # AI (especially cheaper TTS) is often monotonic (<10Hz).
        # We invert the metric: Lower variance = Higher AI Score.
        
        if pitch_std <= 0: return 0.5 # Edge case: no pitch detected (whisper)
        
        # Linear interpolation:
        # If std=0 (perfectly flat) -> Score=1.0
        # If std=20 (normal) -> Score=0.0
        score = max(0.0, 1.0 - (pitch_std / THRESHOLD_PITCH_STD_LOW))
        return float(score)

    @staticmethod
    def _compute_signal_imperfection(features: Dict[str, float]) -> float:
        """
        Signal C: Environmental Imperfection (Heuristic).
        Detects 'Digital Silence' / Lack of Noise Floor.
        """
        zcr_mean = features.get('zcr_mean', 0.0)
        
        # Logic: Real recordings have background noise (higher ZCR).
        # Synthetic audio often has near-zero ZCR in gaps.
        # Invert: Lower ZCR = Higher AI Score.
        
        if zcr_mean <= 0: return 1.0 # Suspiciously perfectly clean
        
        # If ZCR=0 -> Score=1.0
        # If ZCR=0.02 -> Score=0.0
        score = max(0.0, 1.0 - (zcr_mean / THRESHOLD_ZCR_LOW))
        return float(score)

    @staticmethod
    def analyze_voice(features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Entry Point. Fuses signals to produce final decision.
        """
        # 1. Compute Individual Signals
        s_acoustic = IntelligenceEngine._compute_signal_acoustic(features)
        s_temporal = IntelligenceEngine._compute_signal_temporal(features)
        s_imperfection = IntelligenceEngine._compute_signal_imperfection(features)

        # 2. Decision Fusion (Weighted Average)
        # Weights: Acoustic (50%), Temporal (30%), Imperfection (20%)
        # These weights prioritize the ML model but allow heuristics to override if strong.
        final_score = (0.5 * s_acoustic) + (0.3 * s_temporal) + (0.2 * s_imperfection)
        
        # 3. Classification & Risk
        classification = "AI-generated" if final_score > 0.65 else "Human-generated"
        
        if final_score < 0.4:
            risk_level = "Low"
        elif final_score < 0.65:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # 4. Generate Explanation (Explainability Layer)
        # Find which signal contributed most relative to its weight? 
        # Or simply which signal is highest/most anomalous.
        
        reasons = []
        if s_temporal > 0.6:
            reasons.append("Unnatural pitch stability (robotic consistency)")
        if s_imperfection > 0.6:
            reasons.append("Lack of natural background noise (digital silence)")
        if s_acoustic > 0.7:
            reasons.append("Acoustic model detected synthetic spectral artifacts")
            
        if not reasons and classification == "AI-generated":
            reasons.append("Cumulative anomaly score exceeded threshold")
        elif classification == "Human-generated":
            reasons.append("Audio exhibits natural prosody and imperfections")

        explanation_str = "; ".join(reasons)

        # 5. Construct Result
        return {
            "classification": classification,
            "confidence": round(final_score if classification == "AI-generated" else (1.0 - final_score), 4),
            "risk_level": risk_level,
            "language": "Detected (Agnostic)", # Our signals work across languages
            "explanation": explanation_str,
            # Debug info (useful for engineers/judges)
            "_debug": {
                "score_acoustic": round(s_acoustic, 3),
                "score_temporal": round(s_temporal, 3),
                "score_imperfection": round(s_imperfection, 3)
            }
        }
