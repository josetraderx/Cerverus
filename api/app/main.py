# api/app/main.py
import os
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .endpoints import anomaly_detection

app = FastAPI(
    title="Cerverus API",
    description="Financial Anomaly Detection API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# Configure CORS based on environment
def get_allowed_origins() -> list[str]:
    """Get allowed CORS origins based on environment."""
    environment = os.getenv("ENVIRONMENT", "development")

    if environment == "production":
        # In production, use specific origins from environment variable
        origins_str = os.getenv("ALLOWED_ORIGINS", "")
        if origins_str:
            return [origin.strip() for origin in origins_str.split(",")]
        return []  # No CORS in production by default

    # Development: allow common dev origins
    return [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*",  # Remove this in production
    ]


app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include anomaly detection router
app.include_router(anomaly_detection.router, prefix="/api/v1")


@app.get("/")
def read_root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "message": "Cerverus API is running",
        "version": "0.1.0",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "anomaly_detection": "/api/v1/anomaly",
        },
    }


@app.get("/health")
def health_check() -> Dict[str, Any]:
    """Health check endpoint for container orchestration."""
    return {
        "status": "ok",
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "0.1.0",
    }
