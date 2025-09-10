"""
LOF Detector for Cerverus - Standard Interface Implementation

Implements Local Outlier Factor following BaseAnomalyDetector interface.
Compatible with all Cerverus ensemble systems.

Location: src/cerverus/models/lof_detector.py
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# use central paths
from cerverus.config.paths import RESULTS_DIR, ensure_dirs
from cerverus.models.base_detector import BaseAnomalyDetector

# Configure basic logging once at module import time
logging.basicConfig(level=logging.INFO)


class CerverusLOF(BaseAnomalyDetector):
    """
    Local Outlier Factor detector for Cerverus financial data.
    Implements standard BaseAnomalyDetector interface.

    Features:
    - Standard interface compatibility
    - Local density-based anomaly detection
    - Context-aware feature engineering
    - Robust preprocessing with RobustScaler
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float = 0.05,
        metric: str = "minkowski",
        p: int = 2,
        novelty: bool = False,
    ) -> None:
        super().__init__()
        """
        Initialize LOF detector with standard interface.
        
        Args:
            n_neighbors: Number of neighbors for LOF computation
            contamination: Expected proportion of anomalies
            metric: Distance metric for neighbor computation
            p: Power parameter for Minkowski distance
            novelty: Whether to use novelty detection mode
        """
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.p = p
        self.novelty = novelty

        # Model components
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            metric=self.metric,
            p=self.p,
            novelty=self.novelty,
            n_jobs=-1,
        )

        self.scaler = RobustScaler()
        self.feature_names: list[str] = []
        self.is_fitted = False

        # logger is created per-instance
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_core_features(self) -> list[str]:
       """
       Return list of mandatory core features for ensemble compatibility.
       
       Returns:
           List of core feature names that all detectors must implement
       """
       return [
           "price_change_pct",
           "local_volume_ratio",
           "day_of_week"
       ]

    def get_specific_features(self) -> list[str]:
       """
       Return LOF-specific features.
       
       Returns:
           List of LOF-specific feature names
       """
       return [
           "local_volatility",
           "local_price_deviation",
           "volume_price_interaction",
           "volatility_volume_ratio"
       ]

    # -------------------------- Database helpers --------------------------
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
        Load daily_prices_staging data from postgres since a given date.
        
        Args:
            since: Start date filter (YYYY-MM-DD format)

        Returns:
            DataFrame or None on failure
        """
        conn = self.connect_db()
        if conn is None:
            return None

        query = f"""
        SELECT symbol, trade_date, open_price, high_price, low_price,
               close_price, volume, adjusted_close
        FROM daily_prices_staging
        WHERE trade_date >= '{since}'
        ORDER BY symbol, trade_date;
        """

        try:
            df = pd.read_sql_query(query, conn)
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            self.logger.info(f"Loaded {len(df)} rows from DB")
            return df
        except Exception as exc:
            self.logger.error(f"Error loading data: {exc}")
            return None
        finally:
            conn.close()

    # -------------------------- Feature engineering --------------------------
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create LOF-targeted features following standard interface.
        
        Args:
            df: Raw data DataFrame

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (full_dataframe_with_features, features_for_ml)
        """
        features_df = df.copy()

        # Ensure numeric types (maintain all rows)
        numeric_cols = ["open_price", "high_price", "low_price", "close_price", "volume"]
        for col in numeric_cols:
            features_df[col] = features_df[col].astype(float)
        
        # Log any NaN values but keep all rows
        nan_count = features_df[numeric_cols].isna().sum().sum()
        if nan_count > 0:
            self.logger.warning(f"Found {nan_count} NaN values after numeric conversion")

        # Basic price features
        features_df["price_range"] = features_df["high_price"] - features_df["low_price"]
        features_df["price_change"] = features_df["close_price"] - features_df["open_price"]
        features_df["price_change_pct"] = (
            features_df["price_change"] / features_df["open_price"]
        ) * 100

        # Local density features (per symbol) - LOF specialization
        features_df["local_volume_ratio"] = (
            features_df.groupby("symbol")["volume"].transform(
                lambda x: x / x.rolling(window=5, min_periods=1).mean()
            )
        )

        features_df["local_volatility"] = (
            features_df.groupby("symbol")["price_change_pct"].transform(
                lambda x: x.rolling(window=5, min_periods=1).std()
            )
        )

        features_df["local_price_deviation"] = (
            features_df.groupby("symbol")["close_price"].transform(
                lambda x: (x - x.rolling(window=10, min_periods=1).mean())
                / x.rolling(window=10, min_periods=1).std()
            )
        )

        # Temporal/context features
        features_df["day_of_week"] = features_df["trade_date"].dt.dayofweek
        features_df["week_of_month"] = (features_df["trade_date"].dt.day - 1) // 7 + 1

        # Interaction features
        features_df["volume_price_interaction"] = (
            features_df["local_volume_ratio"] * features_df["price_change_pct"]
        )
        features_df["volatility_volume_ratio"] = (
            features_df["local_volatility"] / np.log1p(features_df["volume"])
        )

        # Define feature columns for ML
        feature_columns = [
            "price_change_pct",
            "local_volume_ratio",
            "local_volatility",
            "local_price_deviation",
            "volume_price_interaction",
            "volatility_volume_ratio",
            "day_of_week",
        ]

        # Clean infinities and NaNs
        features_df[feature_columns] = features_df[feature_columns].replace(
            [np.inf, -np.inf], np.nan
        )
        features_df = features_df.dropna(subset=feature_columns)

        self.feature_names = feature_columns
        self.logger.info(f"Engineered {len(feature_columns)} features; rows after clean: {len(features_df)}")

        # Return full dataframe and ML features
        return features_df, features_df[feature_columns]

    # -------------------------- Model training / prediction --------------------------
    def fit(self, X: pd.DataFrame) -> None:
        """
        Train LOF model on features.
        
        Args:
            X: Features DataFrame for training
        """
        X_scaled = self.scaler.fit_transform(X)
        n_samples = X_scaled.shape[0]

        # Adjust n_neighbors if necessary
        effective_n_neighbors = min(self.n_neighbors, max(1, n_samples - 1))

        if effective_n_neighbors != self.n_neighbors:
            self.logger.warning(f"Adjusting n_neighbors: {self.n_neighbors} -> {effective_n_neighbors}")
            # Create new model with adjusted n_neighbors
            self.model = LocalOutlierFactor(
                n_neighbors=effective_n_neighbors,
                contamination=self.contamination,
                metric=self.metric,
                p=self.p,
                novelty=self.novelty,
                n_jobs=-1
            )

        # For novelty=False, fit() stores data but doesn't train model
        # For novelty=True, fit() trains the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        self.logger.info(f"LOF model trained on {len(X)} samples")

    def predict_anomalies(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies using trained LOF model.
        
        Args:
            X: Features DataFrame for prediction
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (anomaly_labels, anomaly_scores)
            - anomaly_labels: 1 = anomaly, 0 = normal
            - anomaly_scores: continuous anomaly scores (more negative = more anomalous)
        """
        X_scaled = self.scaler.transform(X)
        
        if not self.is_fitted:
            # For novelty=False, use fit_predict
            predictions = self.model.fit_predict(X_scaled)
            lof_scores = self.model.negative_outlier_factor_
            self.is_fitted = True
        else:
            # If already trained, use existing scores
            lof_scores = self.model.negative_outlier_factor_
            threshold = np.percentile(lof_scores, 100 * self.contamination)
            predictions = np.where(lof_scores <= threshold, -1, 1)
        
        # Convert to standard format: 1 = anomaly, 0 = normal
        anomaly_labels = (predictions == -1).astype(int)
        
        return anomaly_labels, lof_scores

    # -------------------------- Analysis / persistence --------------------------
    def analyze_results(self, df: pd.DataFrame, anomaly_labels: np.ndarray, lof_scores: np.ndarray) -> pd.DataFrame:
        """Generate comprehensive LOF analysis."""
        results_df = df.copy()
        results_df = results_df.reset_index(drop=True)
        
        # Ensure sizes match
        if len(results_df) != len(anomaly_labels):
            raise ValueError("Length mismatch between df and anomaly labels")

        results_df["lof_anomaly"] = anomaly_labels
        results_df["lof_score"] = lof_scores

        anomalies = results_df[results_df["lof_anomaly"] == 1]

        self.logger.info("=" * 50)
        self.logger.info("LOF RESULTS ANALYSIS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total rows analyzed: {len(results_df)}")
        self.logger.info(f"Anomalies detected: {len(anomalies)}")
        if len(results_df) > 0:
            self.logger.info(f"Anomaly rate: {len(anomalies)/len(results_df)*100:.2f}%")

        if len(anomalies) > 0:
            self.logger.info("Top 10 extreme anomalies:")
            top = anomalies.nsmallest(10, "lof_score")
            for _, row in top.iterrows():
                symbol = row.get("symbol", "?")
                td = row.get("trade_date")
                td_str = td.strftime("%Y-%m-%d") if pd.notna(td) else "N/A"
                self.logger.info(
                    f"{symbol} - {td_str} - LOF: {row['lof_score']:.4f} - "
                    f"price_change_pct: {row.get('price_change_pct', float('nan')):.2f}%"
                )

        return results_df

    def save_model(self, filepath: str = "src/cerverus/models/lof_model.joblib") -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("No trained model to save")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "timestamp": datetime.utcnow(),
            "version": "2.0_refactored"
        }

        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str = "src/cerverus/models/lof_model.joblib") -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to load model from
        """
        try:
            model_data = joblib.load(filepath)
            self.model = model_data["model"]
            self.scaler = model_data["scaler"]
            self.feature_names = model_data.get("feature_names", [])
            self.is_fitted = True
            self.logger.info(f"Model loaded from {filepath}")
        except Exception as exc:
            self.logger.error(f"Error loading model: {exc}")


def main():
    """Main function to run LOF pipeline with standard interface."""
    print("Starting Cerverus LOF detector - Standard Interface")
    print("=" * 60)

    detector = CerverusLOF(n_neighbors=20, contamination=0.05)

    try:
        print("[DATA] Loading data from PostgreSQL...")
        df = detector.load_data()
        if df is None or df.empty:
            print("[ERROR] No data available or DB connection failed. Exiting.")
            return

        print("[FEAT] Applying feature engineering...")
        df_feat, X = detector.engineer_features(df)

        print("[TRAIN] Training LOF model...")
        detector.fit(X)

        print("[DETECT] Detecting anomalies...")
        labels, scores = detector.predict_anomalies(X)

        print("[ANALYZE] Analyzing results...")
        results = detector.analyze_results(df_feat, labels, scores)

        # Export to consolidated results folder
        ensure_dirs()
        out_csv = Path(RESULTS_DIR) / "lof_results.csv"
        results.to_csv(out_csv, index=False)
        print(f"[FILE] Results exported: {out_csv}")

        # Save model
        detector.save_model()
        print("[OK] LOF process completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error in LOF pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()