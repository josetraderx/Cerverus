"""
ISOLATION FOREST - REFACTORED WITH STANDARD INTERFACE
====================================================

Isolation Forest implementation following BaseAnomalyDetector interface.
Compatible with all Cerverus ensemble systems.

Location: src/cerverus/models/isolation_forest_eda.py
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from cerverus.config.paths import RESULTS_DIR, ensure_dirs
from cerverus.models.base_detector import BaseAnomalyDetector

# Configure visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)


class CerverusIsolationForest(BaseAnomalyDetector):
    """
    Financial anomaly detector using Isolation Forest.
    Implements standard BaseAnomalyDetector interface.
    
    Features:
    - Standard interface compatibility
    - Financial-specific feature engineering
    - Robust preprocessing with RobustScaler
    - Comprehensive anomaly analysis
    """
    
    def __init__(self, contamination=0.1, n_estimators=200, random_state=42):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies (default 0.1 = 10%)
            n_estimators: Number of trees in ensemble (default 200)
            random_state: Seed for reproducibility
        """
        super().__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.scaler = RobustScaler()
        self.feature_names = []
        self.is_fitted = False
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        print(f"CerverusIsolationForest initialized:")
        print(f"   - Contamination rate: {contamination}")
        print(f"   - N estimators: {n_estimators}")
        print(f"   - Random state: {random_state}")
    
    def connect_db(self):
        """Connect to PostgreSQL database."""
        try:
            conn = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME", "cerverus_db"),
                user=os.getenv("DB_USER", "joseadmin"),
                password=os.getenv("DB_PASSWORD", "Jireh2023."),
                port=os.getenv("DB_PORT", "5433")
            )
            return conn
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return None
    
    def load_data(self, since: Optional[str] = "2025-04-01") -> Optional[pd.DataFrame]:
        """
        Load data from daily_prices_staging table.
        
        Args:
            since: Start date filter (YYYY-MM-DD format)
            
        Returns:
            DataFrame or None on failure
        """
        conn = self.connect_db()
        if conn is None:
            return None
        
        query = f"""
        SELECT 
            staging_id,
            symbol,
            trade_date,
            open_price,
            high_price, 
            low_price,
            close_price,
            volume,
            adjusted_close
        FROM daily_prices_staging 
        WHERE trade_date >= '{since}'
        ORDER BY trade_date DESC, symbol
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            
            # Convert numeric columns from Decimal to float
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adjusted_close']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            self.raw_data = df
            self.logger.info(f"Loaded {len(df)} rows from DB")
            self.logger.info(f"   - Unique symbols: {df['symbol'].nunique()}")
            self.logger.info(f"   - Date range: {df['trade_date'].min()} to {df['trade_date'].max()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None
        finally:
            conn.close()
    
    def get_core_features(self) -> list[str]:
        """Return list of mandatory core features for ensemble compatibility."""
        return [
            "price_volatility",
            "price_momentum",
            "volume_normalized",
            "day_of_week"
        ]

    def get_specific_features(self) -> list[str]:
        """Return Isolation Forest-specific features."""
        return [
            "price_gap",
            "volume_spike",
            "price_volume_interaction"
        ]

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
            features_df[col] = features_df[col].astype(float)
        
        # Basic price features
        features_df['price_range'] = features_df['high_price'] - features_df['low_price']
        features_df['price_change'] = features_df['close_price'] - features_df['open_price']
        features_df['price_change_pct'] = (features_df['price_change'] / features_df['open_price']) * 100.0
        
        # Financial anomaly features
        features_df['price_volatility'] = features_df['price_range'] / features_df['open_price']
        features_df['price_momentum'] = features_df['price_change_pct']
        
        # Volume features (normalized by symbol)
        features_df['volume_median_by_symbol'] = features_df.groupby('symbol')['volume'].transform('median')
        features_df['volume_normalized'] = features_df['volume'] / features_df['volume_median_by_symbol']
        
        # Sort for temporal features
        features_df = features_df.sort_values(['symbol', 'trade_date'])
        
        # Price gap analysis
        features_df['prev_close'] = features_df.groupby('symbol')['close_price'].shift(1)
        features_df['price_gap'] = ((features_df['open_price'] - features_df['prev_close']) / features_df['prev_close'] * 100.0).fillna(0.0)
        
        # Volume spike detection
        features_df['volume_ma5'] = features_df.groupby('symbol')['volume'].rolling(5, min_periods=1).mean().reset_index(0, drop=True)
        features_df['volume_spike'] = features_df['volume'] / features_df['volume_ma5']
        
        # Temporal features
        features_df['day_of_week'] = features_df['trade_date'].dt.dayofweek
        
        # Price-volume interaction
        features_df['price_volume_interaction'] = features_df['price_change_pct'] * np.log1p(features_df['volume_normalized'])
        
        # Select features for ML model
        self.feature_names = [
            'price_volatility',
            'price_momentum', 
            'volume_normalized',
            'price_gap',
            'volume_spike',
            'price_volume_interaction',
            'day_of_week'
        ]
        
        # Clean data
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna(subset=self.feature_names)
        
        # Create ML features DataFrame
        ml_features = features_df[self.feature_names].copy()
        
        self.processed_data = features_df
        
        self.logger.info(f"Engineered {len(self.feature_names)} features; rows after clean: {len(features_df)}")
        
        return features_df, ml_features
    
    def fit(self, X: pd.DataFrame) -> None:
        """
        Train Isolation Forest model.
        
        Args:
            X: Features DataFrame for training
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        self.logger.info(f"Isolation Forest trained on {len(X)} samples")
    
    def predict_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using trained model.
        
        Args:
            X: Features DataFrame for prediction
            
        Returns:
            Tuple: (anomaly_labels, anomaly_scores)
            - anomaly_labels: 1 = anomaly, 0 = normal
            - anomaly_scores: continuous anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions and scores
        predictions = self.model.predict(X_scaled)  # 1 = normal, -1 = anomaly
        scores = self.model.decision_function(X_scaled)  # More negative = more anomalous
        
        # Convert to standard format: 1 = anomaly, 0 = normal
        anomaly_labels = (predictions == -1).astype(int)
        
        return anomaly_labels, scores
    
    def save_model(self, filepath: str = "src/cerverus/models/isolation_forest_model.joblib") -> None:
        """Save trained model to disk."""
        if not self.is_fitted:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'contamination': self.contamination,
            'n_estimators': self.n_estimators,
            'timestamp': datetime.utcnow(),
            'version': '2.0_refactored'
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: str = "src/cerverus/models/isolation_forest_model.joblib") -> None:
        """Load trained model from disk."""
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data.get('feature_names', [])
            self.is_fitted = True
            
            self.logger.info(f"Model loaded from: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
    
    def analyze_results(self, df: pd.DataFrame, anomaly_labels: np.ndarray, anomaly_scores: np.ndarray) -> pd.DataFrame:
        """Generate comprehensive anomaly analysis."""
        results_df = df.copy() if self.processed_data is None else self.processed_data.copy()
        results_df = results_df.reset_index(drop=True)
        
        # Add results
        results_df['is_anomaly'] = anomaly_labels
        results_df['anomaly_score'] = anomaly_scores
        results_df['anomaly_probability'] = 1 / (1 + np.exp(anomaly_scores))
        
        # Analysis
        anomalies = results_df[results_df['is_anomaly'] == 1]
        
        self.logger.info("=" * 60)
        self.logger.info("ISOLATION FOREST ANALYSIS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total records analyzed: {len(results_df)}")
        self.logger.info(f"Anomalies detected: {len(anomalies)}")
        self.logger.info(f"Anomaly rate: {len(anomalies)/len(results_df)*100:.2f}%")
        
        if len(anomalies) > 0:
            self.logger.info(f"\nTop 10 most extreme anomalies:")
            top_anomalies = anomalies.nsmallest(10, 'anomaly_score')
            for _, row in top_anomalies.iterrows():
                symbol = row.get('symbol', '?')
                td = row.get('trade_date')
                td_str = td.strftime('%Y-%m-%d') if pd.notna(td) else 'N/A'
                self.logger.info(
                    f"  {symbol} - {td_str} - Score: {row['anomaly_score']:.4f} - "
                    f"Volatility: {row.get('price_volatility', 0):.4f}"
                )
        
        return results_df
    
    def create_visualizations(self, results_df: pd.DataFrame) -> None:
        """Generate anomaly visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Isolation Forest Anomaly Detection Results', fontsize=16)
        
        # Score distribution
        ax1 = axes[0, 0]
        normal_scores = results_df[results_df['is_anomaly'] == 0]['anomaly_score']
        anomaly_scores = results_df[results_df['is_anomaly'] == 1]['anomaly_score']
        
        ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal', density=True)
        if len(anomaly_scores) > 0:
            ax1.hist(anomaly_scores, bins=50, alpha=0.7, label='Anomalies', density=True)
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Top symbols with anomalies
        ax2 = axes[0, 1]
        if len(results_df[results_df['is_anomaly'] == 1]) > 0:
            symbol_counts = results_df[results_df['is_anomaly'] == 1]['symbol'].value_counts().head(10)
            symbol_counts.plot(kind='bar', ax=ax2)
            ax2.set_title('Top Symbols with Anomalies')
            ax2.set_xlabel('Symbol')
            ax2.set_ylabel('Anomaly Count')
            ax2.tick_params(axis='x', rotation=45)
        
        # Time series
        ax3 = axes[1, 0]
        if len(results_df[results_df['is_anomaly'] == 1]) > 0:
            daily_anomalies = results_df[results_df['is_anomaly'] == 1].groupby('trade_date').size()
            daily_anomalies.plot(ax=ax3)
            ax3.set_title('Anomalies Over Time')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Anomaly Count')
        
        # Feature scatter
        ax4 = axes[1, 1]
        normal_data = results_df[results_df['is_anomaly'] == 0]
        anomaly_data = results_df[results_df['is_anomaly'] == 1]
        
        ax4.scatter(normal_data['price_volatility'], normal_data['volume_normalized'], 
                   alpha=0.6, label='Normal', s=20)
        if len(anomaly_data) > 0:
            ax4.scatter(anomaly_data['price_volatility'], anomaly_data['volume_normalized'], 
                       alpha=0.8, label='Anomalies', s=40, color='red')
        ax4.set_xlabel('Price Volatility')
        ax4.set_ylabel('Volume Normalized')
        ax4.set_title('Volatility vs Volume')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        ensure_dirs()
        viz_path = Path(RESULTS_DIR) / "isolation_forest_analysis.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Visualization saved to: {viz_path}")
        plt.close()


def main():
    """Main function to run complete Isolation Forest pipeline."""
    print("Starting Isolation Forest - Standard Interface")
    print("=" * 60)
    
    # Initialize detector
    detector = CerverusIsolationForest(
        contamination=0.05,
        n_estimators=200,
        random_state=42
    )
    
    try:
        # Load data
        print("[DATA] Loading data from PostgreSQL...")
        df = detector.load_data()
        if df is None or df.empty:
            print("[ERROR] No data available. Exiting.")
            return
        
        # Feature engineering
        print("[FEAT] Engineering features...")
        df_features, X = detector.engineer_features(df)
        
        # Train model
        print("[TRAIN] Training Isolation Forest...")
        detector.fit(X)
        
        # Detect anomalies
        print("[DETECT] Detecting anomalies...")
        labels, scores = detector.predict_anomalies(X)
        
        # Analyze results
        print("[CHART] Analyzing results...")
        results = detector.analyze_results(df_features, labels, scores)
        
        # Create visualizations
        detector.create_visualizations(results)
        
        # Save results
        ensure_dirs()
        out_csv = Path(RESULTS_DIR) / "isolation_forest_results.csv"
        results.to_csv(out_csv, index=False)
        print(f"[FILE] Results exported: {out_csv}")
        
        # Save model
        detector.save_model()
        
        print("\n[OK] Isolation Forest pipeline completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()