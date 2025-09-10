"""
CERVERUS SYSTEM - AUTOENCODER ANOMALY DETECTOR
Implements BaseAnomalyDetector interface for ensemble compatibility
Maintains feature engineering consistency with IF+LOF components
Target: ~10% anomaly rate for ensemble harmony

Location: src/cerverus/models/autoencoder_detector.py
"""

import logging
import os
from datetime import datetime
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
import joblib

from cerverus.models.base_detector import BaseAnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)

class CerverusAutoencoder(BaseAnomalyDetector):
    """
    Autoencoder anomaly detector for Cerverus financial data.
    Implements standard BaseAnomalyDetector interface.
    
    Features:
    - Standard interface compatibility
    - Reconstruction-based anomaly detection
    - Neural network architecture: 9→6→3→6→9
    - Compatible with ensemble systems
    """
    
    def __init__(self, encoding_dim: int = 3, contamination: float = 0.1,
                 epochs: int = 100, batch_size: int = 32, random_state: int = 42):
        """
        Initialize Autoencoder detector with standard interface.
        
        Args:
            encoding_dim: Bottleneck dimension (3 for compression)
            contamination: Expected anomaly rate (0.1 = 10%)
            epochs: Training epochs
            batch_size: Batch size for training
            random_state: Random seed for reproducibility
        """
        super().__init__()
        self.encoding_dim = encoding_dim
        self.contamination = contamination
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        
        # Model components
        self.scaler = StandardScaler()
        self.autoencoder = None
        self.encoder = None
        self.threshold = None
        self.is_fitted = False
        self.feature_names = []
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_state)
        np.random.seed(random_state)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        print(f"CerverusAutoencoder initialized:")
        print(f"   - Encoding dimension: {encoding_dim}")
        print(f"   - Contamination rate: {contamination}")
        print(f"   - Epochs: {epochs}")
        print(f"   - Batch size: {batch_size}")
    
    def get_core_features(self) -> list[str]:
        """
        Return list of mandatory core features for ensemble compatibility.
        
        Returns:
            List of core feature names that all detectors must implement
        """
        return [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume"
        ]

    def get_specific_features(self) -> list[str]:
        """
        Return Autoencoder-specific features.
        
        Returns:
            List of Autoencoder-specific feature names
        """
        return [
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "adjusted_close"
        ]

    def load_data(self, since: Optional[str] = "2025-04-01") -> Optional[pd.DataFrame]:
        """
        Load data from database.
        
        Args:
            since: Start date filter (YYYY-MM-DD format)
            
        Returns:
            DataFrame or None if error
        """
        # Import here to avoid circular imports
        from cerverus.models.isolation_forest_eda import CerverusIsolationForest
        
        detector = CerverusIsolationForest()
        df = detector.load_data(since)
        
        if df is not None:
            self.logger.info(f"Loaded {len(df)} rows from DB")
        else:
            self.logger.error("Failed to load data from database")
            
        return df

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create features following standard interface.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            Tuple: (full_dataframe_with_features, features_for_ml)
        """
        features_df = df.copy()
        
        # Ensure numeric types
        numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume']
        for col in numeric_columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        
        # Feature engineering - consistent with Tier 2
        features_df['price_range'] = features_df['high_price'] - features_df['low_price']
        features_df['daily_return'] = (features_df['close_price'] - features_df['open_price']) / features_df['open_price']
        features_df['volume_log'] = np.log1p(features_df['volume'])
        
        # Define feature columns for ML - use only available features
        base_features = [
            'open_price', 'high_price', 'low_price', 'close_price',
            'volume'
        ]
        
        # Filter to only features that exist in the data
        self.feature_names = [f for f in base_features if f in features_df.columns]
        
        # If we have the base price/volume features, engineer the derived ones
        if all(f in features_df.columns for f in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']):
            features_df['price_range'] = features_df['high_price'] - features_df['low_price']
            features_df['daily_return'] = (features_df['close_price'] - features_df['open_price']) / features_df['open_price']
            features_df['volume_log'] = np.log1p(features_df['volume'])
            self.feature_names.extend(['price_range', 'daily_return', 'volume_log'])
        
        # Validate we have at least the core price/volume features
        core_features = self.get_core_features()
        missing_core = [f for f in core_features if f not in features_df.columns]
        if missing_core:
            raise ValueError(f"Missing required core features: {missing_core}")
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        
        # Debug: Check for missing values before dropping
        self.logger.info(f"Before cleaning - Total rows: {len(features_df)}")
        for col in self.feature_names:
            if col in features_df.columns:
                missing_count = features_df[col].isna().sum()
                if missing_count > 0:
                    self.logger.info(f"Column '{col}' has {missing_count} missing values ({missing_count/len(features_df)*100:.2f}%)")
            else:
                self.logger.warning(f"Column '{col}' not found in data, will be excluded")
        
        # Drop rows with NaN values in feature columns (only for columns that exist)
        existing_features = [f for f in self.feature_names if f in features_df.columns]
        if not existing_features:
            self.logger.error("ERROR: No valid features found in the data!")
            return features_df, pd.DataFrame()
        
        features_df = features_df.dropna(subset=existing_features)
        
        # Create ML features DataFrame using only existing features
        ml_features = features_df[existing_features].copy()
        
        self.logger.info(f"Engineered {len(existing_features)} features; rows after clean: {len(features_df)}")
        
        # Additional debug info
        if len(features_df) == 0:
            self.logger.error("ERROR: All data was filtered out during cleaning!")
            # Let's examine what went wrong
            original_df = df.copy()
            self.logger.info(f"Original data shape: {original_df.shape}")
            self.logger.info(f"Original columns: {list(original_df.columns)}")
            for col in existing_features:
                if col in original_df.columns:
                    original_missing = original_df[col].isna().sum()
                    self.logger.info(f"Original '{col}': {original_missing}/{len(original_df)} missing ({original_missing/len(original_df)*100:.2f}%)")
                else:
                    self.logger.warning(f"Column '{col}' not found in original data")
        
        return features_df, ml_features
    
    def _build_autoencoder(self, input_dim: int) -> keras.Model:
        """
        Build autoencoder architecture: 9→6→3→6→9
        Symmetric encoder-decoder with bottleneck compression
        """
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder layers
        encoded = layers.Dense(6, activation='relu', name='encoder_hidden')(input_layer)
        encoded = layers.Dropout(0.1)(encoded)  # Light regularization
        encoded = layers.Dense(self.encoding_dim, activation='relu', name='bottleneck')(encoded)
        
        # Decoder layers  
        decoded = layers.Dense(6, activation='relu', name='decoder_hidden')(encoded)
        decoded = layers.Dropout(0.1)(decoded)  # Light regularization
        decoded = layers.Dense(input_dim, activation='linear', name='output')(decoded)
        
        # Create autoencoder model
        autoencoder = keras.Model(input_layer, decoded, name='CerverusAutoencoder')
        
        # Create encoder model (for feature extraction if needed)
        encoder = keras.Model(input_layer, encoded, name='Encoder')
        
        return autoencoder, encoder
    
    def fit(self, X: pd.DataFrame) -> None:
        """
        Train the autoencoder model on features.
        
        Args:
            X: Features DataFrame for training
        """
        try:
            self.logger.info("Training Autoencoder...")
            
            # Check if we have valid features
            if X.empty or len(X.columns) == 0:
                self.logger.error("ERROR: No valid features available for training!")
                raise ValueError("No valid features available for training")
            
            self.logger.info(f"Training with {len(X)} samples and {len(X.columns)} features: {list(X.columns)}")
            
            # Scale features for neural network training
            X_scaled = self.scaler.fit_transform(X)
            
            # Build autoencoder architecture
            self.autoencoder, self.encoder = self._build_autoencoder(X_scaled.shape[1])
            
            # Compile with appropriate loss and optimizer
            self.autoencoder.compile(
                optimizer='adam',
                loss='mse',  # Mean Squared Error for reconstruction
                metrics=['mae']  # Mean Absolute Error as additional metric
            )
            
            # Train autoencoder (early stopping to prevent overfitting)
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=10,
                restore_best_weights=True
            )
            
            self.logger.info(f"Training with {self.epochs} epochs, batch_size={self.batch_size}")
            history = self.autoencoder.fit(
                X_scaled, X_scaled,  # Autoencoder learns to reconstruct input
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=0  # Quiet training
            )
            
            # Calculate reconstruction errors for threshold setting
            predictions = self.autoencoder.predict(X_scaled, verbose=0)
            reconstruction_errors = np.mean(np.square(X_scaled - predictions), axis=1)
            
            # Set threshold for anomaly detection (based on contamination rate)
            self.threshold = np.percentile(reconstruction_errors, 100 * (1 - self.contamination))
            
            self.is_fitted = True
            
            # Training summary
            final_loss = history.history['loss'][-1]
            self.logger.info(f"[OK] Training completed!")
            self.logger.info(f"   Final loss: {final_loss:.6f}")
            self.logger.info(f"   Anomaly threshold: {self.threshold:.6f}")
            self.logger.info(f"   Expected anomaly rate: {self.contamination*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
    
    def predict_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using trained autoencoder.
        
        Args:
            X: Features DataFrame for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
            - anomaly_labels: 1 = anomaly, 0 = normal
            - anomaly_scores: continuous anomaly scores (reconstruction errors)
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before prediction")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get reconstructions
            reconstructions = self.autoencoder.predict(X_scaled, verbose=0)
            
            # Calculate reconstruction errors
            reconstruction_errors = np.mean(np.square(X_scaled - reconstructions), axis=1)
            
            # Binary anomaly labels (1 = anomaly, 0 = normal)
            anomaly_labels = (reconstruction_errors > self.threshold).astype(int)
            
            # Results summary
            n_anomalies = np.sum(anomaly_labels)
            anomaly_rate = (n_anomalies / len(anomaly_labels)) * 100
            
            self.logger.info(f"[RESULTS] Autoencoder Prediction Results:")
            self.logger.info(f"   Samples processed: {len(anomaly_labels):,}")
            self.logger.info(f"   Anomalies detected: {n_anomalies:,}")
            self.logger.info(f"   Anomaly rate: {anomaly_rate:.2f}%")
            self.logger.info(f"   Reconstruction error range: {reconstruction_errors.min():.6f} - {reconstruction_errors.max():.6f}")
            
            return anomaly_labels, reconstruction_errors
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise
    
    def save_model(self, filepath: str = "src/cerverus/models/autoencoder_model.joblib") -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'autoencoder': self.autoencoder,
            'encoder': self.encoder,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'encoding_dim': self.encoding_dim,
            'contamination': self.contamination,
            'timestamp': datetime.utcnow(),
            'version': '1.0_standard'
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to: {filepath}")

    def load_model(self, filepath: str = "src/cerverus/models/autoencoder_model.joblib") -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        try:
            model_data = joblib.load(filepath)
            
            self.autoencoder = model_data['autoencoder']
            self.encoder = model_data['encoder']
            self.scaler = model_data['scaler']
            self.threshold = model_data['threshold']
            self.feature_names = model_data.get('feature_names', [])
            self.is_fitted = True
            
            self.logger.info(f"Model loaded from: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")

    def get_model_info(self) -> dict:
        """
        Get model architecture and training information
        """
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "architecture": "9→6→3→6→9",
            "encoding_dim": self.encoding_dim,
            "contamination": self.contamination,
            "threshold": self.threshold,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "total_parameters": self.autoencoder.count_params() if self.autoencoder else 0
        }



# TESTING AND VALIDATION FUNCTIONS
def test_autoencoder_compatibility():
    """
    Test autoencoder compatibility with existing Tier 2 architecture
    """
    print("[TEST] Testing CerverusAutoencoder compatibility...")
    
    # Create sample data (matching the 9-feature structure)
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'open_price': np.random.normal(100, 10, 1000),
        'high_price': np.random.normal(105, 10, 1000),
        'low_price': np.random.normal(95, 10, 1000),
        'close_price': np.random.normal(102, 10, 1000),
        'volume': np.random.lognormal(10, 1, 1000),
        'adjusted_close': np.random.normal(102, 10, 1000)
    })
    
    # Test autoencoder
    autoencoder = CerverusAutoencoder(epochs=10)  # Quick test
    features_df, ml_features = autoencoder.engineer_features(sample_data)
    autoencoder.fit(ml_features)
    labels, scores = autoencoder.predict_anomalies(ml_features)
    
    print(f"[OK] Autoencoder test passed!")
    print(f"   Features engineered: {len(autoencoder.feature_names)}")
    print(f"   Anomalies detected: {np.sum(labels)} ({np.sum(labels)/len(labels)*100:.2f}%)")
    print(f"   Score range: {scores.min():.6f} - {scores.max():.6f}")
    
    return True

def main():
    """Main function to run complete Autoencoder pipeline."""
    print("Starting Cerverus Autoencoder - Standard Interface")
    print("=" * 60)
    
    # Initialize detector
    detector = CerverusAutoencoder(
        contamination=0.1,
        epochs=100,
        batch_size=32,
        random_state=42
    )
    
    try:
        # Load data
        print("[DATA] Loading data from database...")
        df = detector.load_data()
        if df is None or df.empty:
            print("[ERROR] No data available. Exiting.")
            return
        
        # Feature engineering
        print("[FEAT] Engineering features...")
        df_features, X = detector.engineer_features(df)
        
        # Train model
        print("[TRAIN] Training Autoencoder...")
        detector.fit(X)
        
        # Detect anomalies
        print("[DETECT] Detecting anomalies...")
        labels, scores = detector.predict_anomalies(X)
        
        # Save model
        detector.save_model()
        
        print("\n[OK] Autoencoder pipeline completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error in pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run compatibility test
    test_autoencoder_compatibility()
    print("\n[READY] CerverusAutoencoder ready for integration!")