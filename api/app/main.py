# api/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoints import anomaly_detection

app = FastAPI(
    title="Cerverus API",
    description="Financial Anomaly Detection API",
    version="0.1.0"
)

# Configurar CORS para desarrollo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir el router de detección de anomalías
app.include_router(anomaly_detection.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {
        "message": "Cerverus API is running",
        "version": "0.1.0",
        "endpoints": {
            "docs": "/docs",
            "anomaly_detection": "/api/v1/anomaly"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}