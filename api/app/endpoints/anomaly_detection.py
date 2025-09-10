# api/app/endpoints/anomaly_detection.py
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cerverus.models.isolation_forest_eda import CerverusIsolationForest

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


class AnomalyRequest(BaseModel):
    data: List[Dict[str, Any]]
    contamination: float = 0.01


class AnomalyResponse(BaseModel):
    anomalies: List[Dict[str, Any]]
    total_anomalies: int
    total_points: int


@router.post("/detect", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyRequest):
    try:
        # Convertir a DataFrame
        df = pd.DataFrame(request.data)

        # Seleccionar solo columnas numéricas para el detector
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise HTTPException(
                status_code=400, detail="No numeric columns found for anomaly detection"
            )

        # Crear y entrenar detector con datos numéricos
        detector = CerverusIsolationForest(contamination=request.contamination)
        detector.fit(numeric_df)

        # Predecir sobre las mismas filas y mapear al dataframe original
        preds = detector.predict(numeric_df)
        mask = preds == -1
        anomalies = df[mask]

        return AnomalyResponse(
            anomalies=anomalies.to_dict("records"),
            total_anomalies=int(mask.sum()),
            total_points=len(df),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health_check():
    return {"status": "anomaly detection service is healthy"}
