import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.endpoints import router as api_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="VoiceGuard API",
    description="AI-Generated Voice Detection API",
    version="1.0.0"
)

# CORS (Allow all for hackathon)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "active", "message": "VoiceGuard API is running"}

# --- MODIFIED: WARMUP FUNCTION ---
@app.on_event("startup")
async def startup_event():
    """
    Pre-loads heavy libraries (Librosa/Numpy) into RAM during startup 
    so the first user request doesn't time out.
    """
    logger.info("🚀 Starting Warmup: Pre-loading audio libraries...")
    try:
        import numpy as np
        import librosa
        
        # Create 1 second of dummy silence at 8kHz
        # This forces the system to allocate memory for audio processing now
        dummy_signal = np.zeros(8000)
        
        # Run a dummy extraction to load the function caches
        librosa.feature.mfcc(y=dummy_signal, sr=8000, n_mfcc=13)
        
        logger.info("✅ Warmup Complete: Audio engine is ready.")
    except Exception as e:
        logger.warning(f"⚠️ Warmup non-critical error: {e}")
