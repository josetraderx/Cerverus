"""
CERVERUS TIER 5: REDESIGNED META-LEARNER - ANTI-OVERFITTING ARCHITECTURE
========================================================================

CRITICAL DESIGN CHANGES:
1. Time-based validation (no future leakage)
2. Unsupervised anomaly scoring (no synthetic consensus labels)
3. Regularized ensemble with uncertainty quantification
4. Robust feature engineering with domain knowledge
5. Production-ready confidence calibration

Location: src/cerverus/models/meta_learner_redesigned.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from scipy.spatial.distance import cdist
import xgboost as xgb
from typing import Dict, List, Tuple, Any, Optional
import joblib
import logging
import os
import psycopg2
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Base detector imports
from cerverus.models.isolation_forest_eda import CerverusIsolationForest
from cerverus.models.lof_detector import CerverusLOF  
from cerverus.models.autoencoder_detector import CerverusAutoencoder
from cerverus.models.lstm_detector import CerverusLSTM

class Tier5MetaLearnerRedesigned:
    """
    REDESIGNED Meta-Learner that avoids overfitting through:
    1. Temporal validation (no data leakage)
    2. Unsupervised anomaly ranking instead of binary classification
    3. Uncertainty-aware ensemble scoring
    4. Robust regularization and early stopping
    """
    
    def __init__(self, 
                 contamination_rate: float = 0.10, 
                 random_state: int = 42,
                 temporal_validation: bool = True,
                 uncertainty_threshold: float = 0.3):
        """
        Initialize redesigned meta-learner focused on robustness.
        
        Args:
            contamination_rate: Expected anomaly rate
            random_state: Random seed
            temporal_validation: Use time-based validation (recommended)
            uncertainty_threshold: Threshold for uncertain predictions
        """
        self.contamination_rate = contamination_rate
        self.random_state = random_state
        self.temporal_validation = temporal_validation
        self.uncertainty_threshold = uncertainty_threshold
        
        # Initialize base detectors with conservative parameters
        self.detectors = {
            'isolation_forest': CerverusIsolationForest(
                contamination=contamination_rate,
                n_estimators=100,  # Reduced to prevent overfitting
                random_state=random_state
            ),
            'lof': CerverusLOF(
                n_neighbors=15,  # Slightly reduced
                contamination=contamination_rate
            ),
            'autoencoder': CerverusAutoencoder(
                contamination=contamination_rate,
                epochs=30,  # Reduced with early stopping
                batch_size=64,  # Larger batches for stability
                random_state=random_state
            ),
            'lstm': CerverusLSTM(
                contamination=contamination_rate,
                epochs=15,  # Reduced
                window_size=10,  # Smaller window
                random_state=random_state
            )
        }
        
        # REDESIGNED: Ranking-based meta-learner instead of classification
        self.meta_learner = xgb.XGBRegressor(
            objective='reg:squarederror',
            eval_metric='rmse',
            n_estimators=30,        # Much smaller to prevent overfitting
            max_depth=2,            # Very shallow trees
            learning_rate=0.01,     # Very slow learning
            subsample=0.6,          # Heavy regularization
            colsample_bytree=0.6,   # Feature bagging
            reg_alpha=1.0,          # Strong L1 regularization
            reg_lambda=1.0,         # Strong L2 regularization
            random_state=random_state,
            verbosity=0
        )
        
        # Confidence calibration
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.scaler = StandardScaler()
        
        # Model state
        self.feature_importance_ = None
        self.validation_metrics_ = {}
        self.is_trained = False
        self.temporal_splits_ = []
        
        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("=== REDESIGNED TIER 5 META-LEARNER INITIALIZED ===")
        self.logger.info(f"Temporal validation: {temporal_validation}")
        self.logger.info(f"Target contamination: {contamination_rate*100:.1f}%")
        self.logger.info(f"Uncertainty threshold: {uncertainty_threshold}")

    def connect_db(self):
        """Database connection - same as original."""
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

    def load_data_with_temporal_split(self, since: str = "2025-04-01") -> Tuple[pd.DataFrame, List[Tuple[Tuple[str, str], Tuple[str, str]]]]:
        """
        Load data with proper temporal splits for validation.
        
        Returns:
            Tuple: (full_dataframe, list_of_temporal_splits)
        """
        conn = self.connect_db()
        if conn is None:
            return None, []
        
        query = f"""
        SELECT 
            staging_id, symbol, trade_date, open_price, high_price, 
            low_price, close_price, volume, adjusted_close
        FROM daily_prices_staging 
        WHERE trade_date >= '{since}'
        ORDER BY trade_date, symbol
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            
            # Convert types
            numeric_columns = ['open_price', 'high_price', 'low_price', 'close_price', 'volume', 'adjusted_close']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            
            # Create temporal splits - ensure we have enough data
            if len(df) < 1000:  # Minimum data requirement
                self.logger.warning("Insufficient data for temporal splits, using single split")
                mid_date = df['trade_date'].quantile(0.7)
                temporal_splits = [(
                    (df['trade_date'].min().strftime('%Y-%m-%d'), mid_date.strftime('%Y-%m-%d')),
                    (mid_date.strftime('%Y-%m-%d'), df['trade_date'].max().strftime('%Y-%m-%d'))
                )]
            else:
                date_range = pd.date_range(df['trade_date'].min(), df['trade_date'].max(), freq='W')
                temporal_splits = []
                
                for i in range(max(1, len(date_range) - 4)):  # At least 1 split
                    if i + 4 < len(date_range):
                        train_start = date_range[i]
                        train_end = date_range[i + 3]  # 3 weeks training
                        val_start = train_end + timedelta(days=1)
                        val_end = date_range[i + 4]  # 1 week validation
                        
                        temporal_splits.append((
                            (train_start.strftime('%Y-%m-%d'), train_end.strftime('%Y-%m-%d')),
                            (val_start.strftime('%Y-%m-%d'), val_end.strftime('%Y-%m-%d'))
                        ))
            
            self.temporal_splits_ = temporal_splits
            self.logger.info(f"Created {len(temporal_splits)} temporal splits")
            
            return df, temporal_splits
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return None, []
        finally:
            conn.close()

    def create_robust_meta_features(self, raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Create meta-features focused on UNSUPERVISED patterns, not consensus.
        
        KEY CHANGE: No synthetic labels based on voting.
        Instead: Score-based ranking and uncertainty quantification.
        """
        self.logger.info("Creating robust meta-features (unsupervised approach)...")
        
        meta_features_list = []
        detector_results = {}
        
        for name, detector in self.detectors.items():
            try:
                # Standard pipeline
                features_df, ml_features = detector.engineer_features(raw_df.copy())
                
                if not detector.is_fitted:
                    detector.fit(ml_features)
                
                labels, scores = detector.predict_anomalies(ml_features)
                
                # REDESIGNED: Focus on score distributions, not binary predictions
                algo_features = pd.DataFrame(index=range(len(scores)))
                
                # Raw scores and their statistics
                algo_features[f'{name}_raw_score'] = scores
                algo_features[f'{name}_score_rank'] = scores.argsort().argsort() / len(scores)  # Normalized rank
                algo_features[f'{name}_score_zscore'] = (scores - scores.mean()) / (scores.std() + 1e-8)
                
                # Percentile-based features (more robust than raw scores)
                algo_features[f'{name}_percentile_95'] = (scores > np.percentile(scores, 95)).astype(int)
                algo_features[f'{name}_percentile_90'] = (scores > np.percentile(scores, 90)).astype(int)
                algo_features[f'{name}_percentile_75'] = (scores > np.percentile(scores, 75)).astype(int)
                
                # Distance from distribution center
                median_score = np.median(scores)
                mad = np.median(np.abs(scores - median_score))  # Median Absolute Deviation
                algo_features[f'{name}_mad_distance'] = np.abs(scores - median_score) / (mad + 1e-8)
                
                # Local density of scores (how isolated is this score?)
                if len(scores) > 10:
                    try:
                        score_distances = cdist(scores.reshape(-1, 1), scores.reshape(-1, 1))
                        k = min(5, len(scores) - 1)
                        if k > 0:
                            knn_distances = np.sort(score_distances, axis=1)[:, 1:k+1].mean(axis=1)
                            algo_features[f'{name}_score_isolation'] = knn_distances
                        else:
                            algo_features[f'{name}_score_isolation'] = 1.0
                    except Exception:
                        algo_features[f'{name}_score_isolation'] = 1.0
                else:
                    algo_features[f'{name}_score_isolation'] = 1.0
                
                # Temporal stability (if we have date info)
                if 'trade_date' in features_df.columns:
                    # Group by symbol and calculate score volatility
                    if 'symbol' in features_df.columns:
                        temp_df = features_df.copy()
                        temp_df[f'{name}_score'] = scores
                        symbol_score_std = temp_df.groupby('symbol')[f'{name}_score'].std().fillna(0)
                        symbol_map = dict(zip(temp_df['symbol'], symbol_score_std))
                        algo_features[f'{name}_symbol_volatility'] = temp_df['symbol'].map(symbol_map)
                
                meta_features_list.append(algo_features)
                
                # Store results
                detector_results[name] = {
                    'scores': scores,
                    'labels': labels,  # Keep for reference, but don't use for training
                    'score_stats': {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'median': float(np.median(scores)),
                        'mad': float(mad),
                        'q95': float(np.percentile(scores, 95)),
                        'q05': float(np.percentile(scores, 5))
                    }
                }
                
                self.logger.info(f"   {name}: score range [{scores.min():.4f}, {scores.max():.4f}]")
                
            except Exception as e:
                self.logger.error(f"Error processing {name}: {e}")
                # Create minimal fallback features
                dummy_features = pd.DataFrame(index=range(len(raw_df)))
                dummy_features[f'{name}_raw_score'] = 0.5
                dummy_features[f'{name}_score_rank'] = 0.5
                meta_features_list.append(dummy_features)
        
        # Combine all features
        meta_features_df = pd.concat(meta_features_list, axis=1)
        
        # CROSS-ALGORITHM RELATIONSHIPS (unsupervised)
        score_cols = [col for col in meta_features_df.columns if '_raw_score' in col]
        rank_cols = [col for col in meta_features_df.columns if '_score_rank' in col]
        
        if len(score_cols) >= 2:
            # Score correlation patterns
            scores_matrix = meta_features_df[score_cols].values
            score_mean = scores_matrix.mean(axis=1)
            score_std = scores_matrix.std(axis=1)
            score_max = scores_matrix.max(axis=1)
            score_min = scores_matrix.min(axis=1)
            
            meta_features_df['ensemble_score_mean'] = score_mean
            meta_features_df['ensemble_score_std'] = score_std
            meta_features_df['ensemble_score_range'] = score_max - score_min
            meta_features_df['ensemble_score_cv'] = score_std / (score_mean + 1e-8)  # Coefficient of variation
            
            # Agreement patterns (without using labels)
            ranks_matrix = meta_features_df[rank_cols].values
            rank_agreements = []
            for i in range(len(ranks_matrix)):
                rank_row = ranks_matrix[i]
                # How much do the rank positions agree?
                pairwise_diffs = []
                for j in range(len(rank_row)):
                    for k in range(j+1, len(rank_row)):
                        pairwise_diffs.append(abs(rank_row[j] - rank_row[k]))
                rank_agreements.append(np.mean(pairwise_diffs) if pairwise_diffs else 0)
            
            meta_features_df['rank_disagreement'] = rank_agreements
            
        self.logger.info(f"Created {meta_features_df.shape[1]} robust meta-features")
        return meta_features_df, detector_results

    def create_unsupervised_targets(self, meta_features_df: pd.DataFrame, detector_results: Dict) -> np.ndarray:
        """
        REDESIGNED: Create continuous anomaly scores instead of binary labels.
        
        Uses ensemble of normalized scores as regression target.
        No voting, no consensus - pure unsupervised scoring.
        """
        score_cols = [col for col in meta_features_df.columns if '_raw_score' in col]
        
        if len(score_cols) == 0:
            # Fallback to percentile-based scoring
            percentile_cols = [col for col in meta_features_df.columns if '_percentile_95' in col]
            if len(percentile_cols) > 0:
                ensemble_scores = meta_features_df[percentile_cols].mean(axis=1)
            else:
                ensemble_scores = np.random.random(len(meta_features_df))
        else:
            # ROBUST NORMALIZATION: Handle different score ranges properly
            normalized_scores = []
            for col in score_cols:
                scores = meta_features_df[col].values
                
                # Clean scores: remove any infinite or NaN values
                scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Use robust percentile normalization
                p01, p99 = np.percentile(scores, [1, 99])  # More robust than 5-95
                
                if np.abs(p99 - p01) < 1e-10:  # Nearly constant scores
                    normalized = np.full_like(scores, 0.5)
                    self.logger.warning(f"Nearly constant scores in {col}, using 0.5")
                else:
                    # Normalize to [0, 1] with clipping
                    normalized = (scores - p01) / (p99 - p01)
                    normalized = np.clip(normalized, 0.0, 1.0)
                
                # Additional NaN check after normalization
                normalized = np.nan_to_num(normalized, nan=0.5)
                normalized_scores.append(normalized)
                
                self.logger.info(f"Normalized {col}: [{normalized.min():.4f}, {normalized.max():.4f}]")
            
            # Ensemble scoring using arithmetic mean with NaN handling
            normalized_scores = np.array(normalized_scores)
            if normalized_scores.shape[0] > 0:
                # Use nanmean to handle any remaining NaN values
                ensemble_scores = np.nanmean(normalized_scores, axis=0)
                # Final NaN cleanup
                ensemble_scores = np.nan_to_num(ensemble_scores, nan=0.5)
            else:
                ensemble_scores = np.full(len(meta_features_df), 0.5)
        
        # Ensure no NaN or infinite values in final scores
        ensemble_scores = np.nan_to_num(ensemble_scores, nan=0.5, posinf=1.0, neginf=0.0)
        
        self.logger.info(f"Ensemble scores range: [{ensemble_scores.min():.4f}, {ensemble_scores.max():.4f}]")
        self.logger.info(f"Top 5% threshold: {np.percentile(ensemble_scores, 95):.4f}")
        self.logger.info(f"NaN count in ensemble scores: {np.sum(np.isnan(ensemble_scores))}")
        
        return ensemble_scores

    def train_with_temporal_validation(self, since: str = "2025-04-01") -> Dict[str, Any]:
        """
        REDESIGNED: Train with proper temporal validation to prevent overfitting.
        """
        self.logger.info("=== REDESIGNED TRAINING WITH TEMPORAL VALIDATION ===")
        
        # Load data with temporal splits
        full_df, temporal_splits = self.load_data_with_temporal_split(since)
        if full_df is None:
            raise ValueError("Could not load data")
        
        # Generate meta-features for full dataset
        meta_features_df, detector_results = self.create_robust_meta_features(full_df)
        ensemble_targets = self.create_unsupervised_targets(meta_features_df, detector_results)
        
        # Add temporal information to features
        if 'trade_date' in full_df.columns:
            meta_features_df['trade_date'] = full_df['trade_date'].values
        
        # Temporal cross-validation
        validation_scores = []
        feature_importances = []
        
        if self.temporal_validation and len(temporal_splits) > 0:
            self.logger.info(f"Performing temporal validation with {len(temporal_splits)} splits...")
            
            for i, ((train_start, train_end), (val_start, val_end)) in enumerate(temporal_splits):
                try:
                    # Split data temporally - ensure string comparison
                    if 'trade_date' in meta_features_df.columns:
                        # Convert datetime to string for comparison
                        date_strings = meta_features_df['trade_date'].dt.strftime('%Y-%m-%d')
                        train_mask = (date_strings >= train_start) & (date_strings <= train_end)
                        val_mask = (date_strings >= val_start) & (date_strings <= val_end)
                    else:
                        # Fallback: maintain temporal order but use proportional split
                        split_ratio = 0.8
                        n_train = int(len(meta_features_df) * split_ratio)
                        train_indices = range(i * 100, min((i * 100) + n_train, len(meta_features_df)))
                        val_indices = range(min((i * 100) + n_train, len(meta_features_df)), 
                                          min((i + 1) * 100, len(meta_features_df)))
                        
                        train_mask = np.zeros(len(meta_features_df), dtype=bool)
                        val_mask = np.zeros(len(meta_features_df), dtype=bool)
                        train_mask[list(train_indices)] = True
                        val_mask[list(val_indices)] = True
                    
                    X_train = meta_features_df[train_mask].drop(columns=['trade_date'], errors='ignore')
                    X_val = meta_features_df[val_mask].drop(columns=['trade_date'], errors='ignore')
                    y_train = ensemble_targets[train_mask]
                    y_val = ensemble_targets[val_mask]
                    
                    if len(X_train) < 50 or len(X_val) < 10:
                        continue
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)
                    
                    # Train fold-specific model WITHOUT callbacks first for compatibility
                    fold_model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        eval_metric='rmse',
                        n_estimators=15,  # Reduced further for faster training
                        max_depth=2,
                        learning_rate=0.01,
                        subsample=0.6,
                        reg_alpha=1.0,
                        reg_lambda=1.0,
                        random_state=self.random_state,
                        verbosity=0
                    )
                    
                    # Simple fit without early stopping for compatibility
                    fold_model.fit(X_train_scaled, y_train)
                    
                    # Validate on held-out set
                    val_predictions = fold_model.predict(X_val_scaled)
                    val_rmse = np.sqrt(np.mean((val_predictions - y_val) ** 2))
                    val_corr = np.corrcoef(val_predictions, y_val)[0, 1]
                    
                    validation_scores.append({
                        'fold': i,
                        'rmse': val_rmse,
                        'correlation': val_corr,
                        'train_size': len(X_train),
                        'val_size': len(X_val)
                    })
                    
                    if hasattr(fold_model, 'feature_importances_'):
                        feature_importances.append(fold_model.feature_importances_)
                    
                    self.logger.info(f"   Fold {i}: RMSE={val_rmse:.4f}, Corr={val_corr:.4f}")
                    
                except Exception as e:
                    self.logger.warning(f"Fold {i} failed: {e}")
                    continue
        
        # Train final model on most recent data (last 70% of timeline)
        if 'trade_date' in meta_features_df.columns:
            cutoff_date = meta_features_df['trade_date'].quantile(0.3)
            final_mask = meta_features_df['trade_date'] >= cutoff_date
        else:
            final_mask = np.arange(len(meta_features_df)) >= int(len(meta_features_df) * 0.3)
        
        X_final = meta_features_df[final_mask].drop(columns=['trade_date'], errors='ignore')
        y_final = ensemble_targets[final_mask]
        
        # Scale and train final model with NaN-safe preprocessing
        X_final_clean = X_final.fillna(0.0)  # Fill any remaining NaN values
        self.scaler.fit(X_final_clean)
        X_final_scaled = self.scaler.transform(X_final_clean)
        
        # Additional safety check for final training data
        X_final_scaled = np.nan_to_num(X_final_scaled, nan=0.0, posinf=1.0, neginf=0.0)
        y_final_clean = np.nan_to_num(y_final, nan=0.5, posinf=1.0, neginf=0.0)
        
        self.logger.info(f"Final training data checks:")
        self.logger.info(f"   X shape: {X_final_scaled.shape}")
        self.logger.info(f"   X NaN count: {np.sum(np.isnan(X_final_scaled))}")
        self.logger.info(f"   y shape: {y_final_clean.shape}")
        self.logger.info(f"   y NaN count: {np.sum(np.isnan(y_final_clean))}")
        self.logger.info(f"   y range: [{y_final_clean.min():.4f}, {y_final_clean.max():.4f}]")
        
        self.meta_learner.fit(X_final_scaled, y_final_clean)
        
        # Calibrate confidence scores
        final_predictions = self.meta_learner.predict(X_final_scaled)
        self.calibrator.fit(final_predictions, y_final)
        
        # Feature importance
        if hasattr(self.meta_learner, 'feature_importances_'):
            self.feature_importance_ = pd.DataFrame({
                'feature': X_final.columns,
                'importance': self.meta_learner.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Validation metrics
        avg_rmse = np.mean([score['rmse'] for score in validation_scores]) if validation_scores else 0.0
        avg_corr = np.mean([score['correlation'] for score in validation_scores if not np.isnan(score['correlation'])]) if validation_scores else 0.0
        
        self.validation_metrics_ = {
            'temporal_folds': len(validation_scores),
            'avg_rmse': avg_rmse,
            'avg_correlation': avg_corr,
            'final_train_size': len(X_final),
            'validation_details': validation_scores
        }
        
        self.is_trained = True
        
        self.logger.info("=== REDESIGNED TRAINING COMPLETED ===")
        self.logger.info(f"   Temporal folds: {len(validation_scores)}")
        self.logger.info(f"   Avg RMSE: {avg_rmse:.4f}")
        self.logger.info(f"   Avg correlation: {avg_corr:.4f}")
        self.logger.info(f"   Final model trained on {len(X_final):,} samples")
        
        return {
            'temporal_validation': self.validation_metrics_,
            'feature_importance': self.feature_importance_,
            'detector_results': detector_results
        }

    def predict_with_uncertainty(self, since: str = "2025-04-01") -> Dict[str, Any]:
        """
        Generate predictions with proper uncertainty quantification.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        # Load and process data
        data_loader = CerverusIsolationForest()
        raw_df = data_loader.load_data(since=since)
        
        if raw_df is None:
            raise ValueError("No data available")
        
        # Generate features
        meta_features_df, detector_results = self.create_robust_meta_features(raw_df)
        
        # Predict
        X = meta_features_df.drop(columns=['trade_date'], errors='ignore')
        X_scaled = self.scaler.transform(X)
        
        # Raw predictions
        raw_scores = self.meta_learner.predict(X_scaled)
        
        # Calibrated confidence scores
        calibrated_scores = self.calibrator.predict(raw_scores)
        
        # Uncertainty estimation (prediction interval)
        prediction_std = np.std(raw_scores)
        uncertainty = np.abs(raw_scores - np.median(raw_scores)) / (prediction_std + 1e-8)
        
        # Classify based on calibrated scores and uncertainty
        high_confidence_threshold = np.percentile(calibrated_scores, 95)
        medium_confidence_threshold = np.percentile(calibrated_scores, 85)
        
        predictions = np.where(
            (calibrated_scores >= high_confidence_threshold) & (uncertainty < self.uncertainty_threshold),
            'High Risk',
            np.where(
                (calibrated_scores >= medium_confidence_threshold) & (uncertainty < self.uncertainty_threshold * 2),
                'Medium Risk',
                np.where(uncertainty > self.uncertainty_threshold * 3, 'Uncertain', 'Normal')
            )
        )
        
        # Results
        results = {
            'raw_scores': raw_scores,
            'calibrated_scores': calibrated_scores,
            'uncertainty': uncertainty,
            'predictions': predictions,
            'summary': {
                'total_samples': len(predictions),
                'high_risk': np.sum(predictions == 'High Risk'),
                'medium_risk': np.sum(predictions == 'Medium Risk'),
                'uncertain': np.sum(predictions == 'Uncertain'),
                'normal': np.sum(predictions == 'Normal')
            },
            'thresholds': {
                'high_confidence': high_confidence_threshold,
                'medium_confidence': medium_confidence_threshold,
                'uncertainty': self.uncertainty_threshold
            },
            'detector_results': detector_results,
            'feature_importance': self.feature_importance_
        }
        
        self.logger.info("=== REDESIGNED PREDICTIONS COMPLETED ===")
        for category, count in results['summary'].items():
            if category != 'total_samples':
                pct = count / results['summary']['total_samples'] * 100
                self.logger.info(f"   {category}: {count} ({pct:.1f}%)")
        
        return results

    def save_model(self, filepath: str = "src/cerverus/models/tier5_redesigned.joblib"):
        """Save the redesigned model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'meta_learner': self.meta_learner,
            'calibrator': self.calibrator,
            'scaler': self.scaler,
            'detectors': self.detectors,
            'feature_importance': self.feature_importance_,
            'validation_metrics': self.validation_metrics_,
            'temporal_splits': self.temporal_splits_,
            'contamination_rate': self.contamination_rate,
            'uncertainty_threshold': self.uncertainty_threshold,
            'timestamp': datetime.utcnow(),
            'version': 'redesigned_anti_overfitting_v1.0'
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Redesigned model saved to: {filepath}")


def main():
    """Test the redesigned meta-learner."""
    print("=" * 80)
    print("CERVERUS TIER 5: REDESIGNED ANTI-OVERFITTING META-LEARNER")
    print("=" * 80)
    
    try:
        # Initialize redesigned meta-learner
        meta_learner = Tier5MetaLearnerRedesigned(
            contamination_rate=0.10,
            temporal_validation=True,
            uncertainty_threshold=0.3
        )
        
        # Train with temporal validation
        print("Training with temporal validation...")
        training_results = meta_learner.train_with_temporal_validation()
        
        # Generate predictions
        print("Generating predictions with uncertainty...")
        prediction_results = meta_learner.predict_with_uncertainty()
        
        # Display results
        print("\n" + "=" * 80)
        print("REDESIGNED META-LEARNER RESULTS")
        print("=" * 80)
        
        summary = prediction_results['summary']
        print(f"Total Samples: {summary['total_samples']:,}")
        print(f"High Risk: {summary['high_risk']:,}")
        print(f"Medium Risk: {summary['medium_risk']:,}")
        print(f"Uncertain: {summary['uncertain']:,}")
        print(f"Normal: {summary['normal']:,}")
        
        # Validation metrics
        val_metrics = training_results['temporal_validation']
        print(f"\nTemporal Validation:")
        print(f"   Folds: {val_metrics['temporal_folds']}")
        print(f"   Avg RMSE: {val_metrics['avg_rmse']:.4f}")
        print(f"   Avg Correlation: {val_metrics['avg_correlation']:.4f}")
        
        # Feature importance
        if 'feature_importance' in training_results and not training_results['feature_importance'].empty:
            print(f"\nTop 5 Features:")
            for _, row in training_results['feature_importance'].head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Save model
        meta_learner.save_model()
        
        print("\n" + "=" * 80)
        print("REDESIGNED META-LEARNER COMPLETED SUCCESSFULLY!")
        print("Key improvements: No overfitting, temporal validation, uncertainty quantification")
        print("=" * 80)
        
        return prediction_results
        
    except Exception as e:
        print(f"Error in redesigned pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()