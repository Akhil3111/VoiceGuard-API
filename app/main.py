import logging
from fastapi import FastAPI
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
    """Simple health check for load balancers."""
    return {"status": "healthy", "service": "VoiceGuard-API"}

if __name__ == "__main__":
    import uvicorn
    # In production, this is run via: uvicorn app.main:app --host 0.0.0.0 --port 80
    uvicorn.run(app, host="0.0.0.0", port=8000)
