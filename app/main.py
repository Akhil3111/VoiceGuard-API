import logging
from fastapi import FastAPI, Request
from app.api.endpoints import router as api_router

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="VoiceGuard AI API",
    description="Multi-Language AI Voice Detection System for India AI Impact Buildathon",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API Router
app.include_router(api_router, prefix="/api/v1", tags=["Analysis"])


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "VoiceGuard-API"}


# 🔥 ADD THIS: Root POST fallback (for GUVI tester)
@app.post("/")
async def root_post_fallback(request: Request):
    return {
        "message": "VoiceGuard API is live. Use POST /api/v1/analyze"
    }


# 🔥 ADD THIS: API prefix POST fallback
@app.post("/api/v1")
async def api_root_post_fallback(request: Request):
    return {
        "message": "Invalid endpoint. Use POST /api/v1/analyze"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
