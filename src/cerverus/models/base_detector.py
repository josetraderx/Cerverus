"""
Base Anomaly Detector Interface
Standard interface that ALL Cerverus detectors must implement.

Location: src/cerverus/models/base_detector.py
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd
import numpy as np


class BaseAnomalyDetector(ABC):
    """
    Standard interface for all Cerverus anomaly detection algorithms.
    ALL detectors must implement these methods with identical signatures.
    
    This ensures:
    - Polymorphism: Any detector can be used interchangeably
    - Consistency: All detectors work the same way
    - Maintainability: Adding new algorithms is standardized
    """
    
    @abstractmethod
    def load_data(self, since: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load data from database.
        
        Args:
            since: Start date filter (optional)
            
        Returns:
            DataFrame or None if error
        """
        pass
    
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standard feature engineering interface.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (full_dataframe_with_features, features_for_ml)
        """
        pass
    
    @abstractmethod
    def fit(self, X: pd.DataFrame) -> None:
        """
        Train the model on features.
        
        Args:
            X: Features DataFrame for training
        """
        pass
    
    @abstractmethod
    def predict_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies.
        
        Args:
            X: Features DataFrame for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
            - anomaly_labels: 1 = anomaly, 0 = normal
            - anomaly_scores: continuous anomaly scores
        """
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        pass

    @abstractmethod
    def get_core_features(self) -> list[str]:
        """
        Returns list of mandatory core features that all detectors must implement.
        These features will be used for ensemble compatibility.
        
        Returns:
            List of core feature names
        """
        pass

    def get_specific_features(self) -> Optional[list[str]]:
        """
        Optional method to return algorithm-specific features.
        
        Returns:
            List of specific feature names or None
        """
        return None

    def validate_features(self, df: pd.DataFrame) -> bool:
        """
        Validate that required features are present in DataFrame.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            bool: True if all features are present
        """
        core_features = self.get_core_features()
        missing = [f for f in core_features if f not in df.columns]
        if missing:
            self.logger.error(f"Missing core features: {missing}")
            return False
        return True