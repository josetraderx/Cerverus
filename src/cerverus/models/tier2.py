# src/cerverus/models/tier2.py
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np

class IsolationForestDetector:
    """Production-ready Isolation Forest implementation"""
    
    def __init__(self, contamination=0.01):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.is_fitted = False
    
    def fit(self, data):
        """Train the model with data"""
        if len(data) < 10:
            raise ValueError("Insufficient data for training")
        
        self.model.fit(data)
        self.is_fitted = True
    
    def predict(self, data):
        """Predict anomalies (-1 for anomalies, 1 for normal)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        return self.model.predict(data)
    
    def get_anomalies(self, data):
        """Return only the anomalous data points"""
        predictions = self.predict(data)
        return data[predictions == -1]