
---
title: "01 - Workflow ML"
version: "1.0"
owner: "Equipo de ML/Data Science"
contact: "#team-ml-platform"
last_updated: "2025-09-09"
---

# 01 - Workflow ML

Ciclo ML: ingestión → features → training → evaluación → deploy.
Requisitos de datasets y validación de modelos.

## Checklist de Calidad para Workflow ML
- [ ] Ingestión y preparación de datos implementada
- [ ] Feature engineering avanzado configurado
- [ ] Entrenamiento y validación de modelos completado
- [ ] Evaluación y métricas de performance verificadas
- [ ] Deployment y monitoring de modelos activo
- [ ] Model registry y versionado funcionando

## Ciclo de Vida de Machine Learning para Detección de Fraude

### Fase 1: Ingestión y Preparación de Datos

#### Estrategias de Ingestión de Datos para Detección de Fraude
```python
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class FraudDataIngestion:
    """
    Clase para ingestión y preparación de datos para detección de fraude.
    Maneja múltiples fuentes de datos y asegura calidad para entrenamiento de modelos.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_sources = config.get('data_sources', {})
        self.quality_thresholds = config.get('quality_thresholds', {})
    
    def ingest_transaction_data(
        self,
        start_date: datetime,
        end_date: datetime,
        data_sources: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ingesta datos de transacciones desde múltiples fuentes.
        
        Args:
            start_date: Fecha de inicio del período de datos
            end_date: Fecha de fin del período de datos
            data_sources: Lista de fuentes de datos a utilizar (opcional)
        
        Returns:
            DataFrame con datos de transacciones consolidados
        """
        if data_sources is None:
            data_sources = list(self.data_sources.keys())
        
        all_data = []
        
        for source_name in data_sources:
            try:
                source_config = self.data_sources[source_name]
                data = self._ingest_from_source(source_name, source_config, start_date, end_date)
                
                if data is not None and not data.empty:
                    data['data_source'] = source_name
                    all_data.append(data)
                    self.logger.info(f"Ingested {len(data)} records from {source_name}")
                else:
                    self.logger.warning(f"No data ingested from {source_name}")
                    
            except Exception as e:
                self.logger.error(f"Error ingesting data from {source_name}: {str(e)}")
                continue
        
        if not all_data:
            raise ValueError("No data could be ingested from any source")
        
        # Consolidar datos
        consolidated_data = pd.concat(all_data, ignore_index=True)
        
        # Validar calidad de datos
        quality_report = self._validate_data_quality(consolidated_data)
        
        if not quality_report['passes_quality_check']:
            self.logger.warning(f"Data quality issues detected: {quality_report['issues']}")
        
        return consolidated_data
    
    def _ingest_from_source(
        self,
        source_name: str,
        source_config: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Ingesta datos desde una fuente específica.
        
        Args:
            source_name: Nombre de la fuente de datos
            source_config: Configuración de la fuente
            start_date: Fecha de inicio
            end_date: Fecha de fin
        
        Returns:
            DataFrame con datos de la fuente o None si falla
        """
        source_type = source_config.get('type')
        
        if source_type == 'database':
            return self._ingest_from_database(source_config, start_date, end_date)
        elif source_type == 'api':
            return self._ingest_from_api(source_config, start_date, end_date)
        elif source_type == 'file':
            return self._ingest_from_file(source_config, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source type: {source_type}")
    
    def _ingest_from_database(
        self,
        config: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Ingesta datos desde base de datos"""
        import sqlalchemy
        
        connection_string = config['connection_string']
        query = config['query'].format(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        engine = sqlalchemy.create_engine(connection_string)
        
        try:
            df = pd.read_sql(query, engine)
            return df
        finally:
            engine.dispose()
    
    def _ingest_from_api(
        self,
        config: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Ingesta datos desde API"""
        import requests
        
        url = config['url']
        headers = config.get('headers', {})
        params = config.get('params', {})
        
        # Añadir parámetros de fecha
        params.update({
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        })
        
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        data = response.json()
        return pd.DataFrame(data)
    
    def _ingest_from_file(
        self,
        config: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Ingesta datos desde archivo"""
        file_path = config['file_path']
        file_format = config.get('format', 'csv')
        
        if file_format == 'csv':
            df = pd.read_csv(file_path)
        elif file_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_format == 'json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        
        # Filtrar por rango de fechas si hay columna de timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            df = df[mask]
        
        return df
    
    def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida la calidad de los datos ingestados.
        
        Args:
            data: DataFrame con datos a validar
        
        Returns:
            Diccionario con resultados de validación
        """
        quality_report = {
            'total_records': len(data),
            'missing_values': {},
            'duplicate_records': 0,
            'data_types': {},
            'value_ranges': {},
            'passes_quality_check': True,
            'issues': []
        }
        
        # Verificar valores faltantes
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                quality_report['missing_values'][column] = {
                    'count': missing_count,
                    'percentage': missing_count / len(data) * 100
                }
                
                # Verificar si excede umbral
                threshold = self.quality_thresholds.get('max_missing_percentage', 5.0)
                if missing_count / len(data) * 100 > threshold:
                    quality_report['issues'].append(
                        f"Column {column} has {missing_count / len(data) * 100:.1f}% missing values (threshold: {threshold}%)"
                    )
                    quality_report['passes_quality_check'] = False
        
        # Verificar registros duplicados
        quality_report['duplicate_records'] = data.duplicated().sum()
        if quality_report['duplicate_records'] > 0:
            quality_report['issues'].append(
                f"Found {quality_report['duplicate_records']} duplicate records"
            )
        
        # Verificar tipos de datos
        for column in data.columns:
            quality_report['data_types'][column] = str(data[column].dtype)
        
        # Verificar rangos de valores para columnas numéricas
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            quality_report['value_ranges'][column] = {
                'min': data[column].min(),
                'max': data[column].max(),
                'mean': data[column].mean(),
                'std': data[column].std()
            }
        
        return quality_report

# Uso de la clase de ingestión
ingestion_config = {
    'data_sources': {
        'transactions_db': {
            'type': 'database',
            'connection_string': 'postgresql://user:password@localhost:5432/fraud_db',
            'query': "SELECT * FROM transactions WHERE timestamp >= '{start_date}' AND timestamp <= '{end_date}'"
        },
        'external_api': {
            'type': 'api',
            'url': 'https://api.example.com/transactions',
            'headers': {'Authorization': 'Bearer token'},
            'params': {'limit': 10000}
        },
        'historical_data': {
            'type': 'file',
            'file_path': '/data/historical_transactions.parquet',
            'format': 'parquet'
        }
    },
    'quality_thresholds': {
        'max_missing_percentage': 5.0,
        'max_duplicate_percentage': 1.0
    }
}

data_ingestion = FraudDataIngestion(ingestion_config)
transaction_data = data_ingestion.ingest_transaction_data(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)
```

#### Preprocesamiento y Limpieza de Datos para Detección de Fraude
```python
class FraudDataPreprocessor:
    """
    Clase para preprocesamiento y limpieza de datos para detección de fraude.
    Maneja valores faltantes, outliers, codificación de variables y normalización.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        fit_transformers: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocesa datos para detección de fraude.
        
        Args:
            data: DataFrame con datos crudos
            fit_transformers: Si es True, ajusta transformadores; si False, usa existentes
        
        Returns:
            Tupla con DataFrame preprocesado y metadata del preprocesamiento
        """
        processed_data = data.copy()
        preprocessing_metadata = {
            'original_shape': data.shape,
            'processing_steps': [],
            'feature_engineering': {}
        }
        
        # 1. Manejar valores faltantes
        processed_data, missing_metadata = self._handle_missing_values(
            processed_data, fit_transformers
        )
        preprocessing_metadata['missing_values_handling'] = missing_metadata
        preprocessing_metadata['processing_steps'].append('missing_values_handling')
        
        # 2. Manejar outliers
        processed_data, outliers_metadata = self._handle_outliers(
            processed_data, fit_transformers
        )
        preprocessing_metadata['outliers_handling'] = outliers_metadata
        preprocessing_metadata['processing_steps'].append('outliers_handling')
        
        # 3. Feature engineering
        processed_data, feature_metadata = self._engineer_features(
            processed_data, fit_transformers
        )
        preprocessing_metadata['feature_engineering'] = feature_metadata
        preprocessing_metadata['processing_steps'].append('feature_engineering')
        
        # 4. Codificar variables categóricas
        processed_data, encoding_metadata = self._encode_categorical_features(
            processed_data, fit_transformers
        )
        preprocessing_metadata['categorical_encoding'] = encoding_metadata
        preprocessing_metadata['processing_steps'].append('categorical_encoding')
        
        # 5. Normalizar/escalar características
        processed_data, scaling_metadata = self._scale_features(
            processed_data, fit_transformers
        )
        preprocessing_metadata['feature_scaling'] = scaling_metadata
        preprocessing_metadata['processing_steps'].append('feature_scaling')
        
        preprocessing_metadata['final_shape'] = processed_data.shape
        
        return processed_data, preprocessing_metadata
    
    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        fit_transformers: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Maneja valores faltantes en el dataset"""
        metadata = {
            'missing_before': {},
            'missing_after': {},
            'methods_used': {}
        }
        
        # Registrar valores faltantes antes del tratamiento
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                metadata['missing_before'][column] = missing_count
        
        # Estrategias para diferentes tipos de columnas
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        # Para columnas numéricas: imputación con mediana
        for column in numeric_columns:
            if data[column].isnull().sum() > 0:
                if fit_transformers:
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='median')
                    data[column] = imputer.fit_transform(data[[column]]).flatten()
                    self.imputers[f'{column}_numeric'] = imputer
                else:
                    imputer = self.imputers.get(f'{column}_numeric')
                    if imputer:
                        data[column] = imputer.transform(data[[column]]).flatten()
                
                metadata['methods_used'][column] = 'median_imputation'
        
        # Para columnas categóricas: imputación con moda
        for column in categorical_columns:
            if data[column].isnull().sum() > 0:
                if fit_transformers:
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='most_frequent')
                    data[column] = imputer.fit_transform(data[[column]]).flatten()
                    self.imputers[f'{column}_categorical'] = imputer
                else:
                    imputer = self.imputers.get(f'{column}_categorical')
                    if imputer:
                        data[column] = imputer.transform(data[[column]]).flatten()
                
                metadata['methods_used'][column] = 'mode_imputation'
        
        # Registrar valores faltantes después del tratamiento
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                metadata['missing_after'][column] = missing_count
        
        return data, metadata
    
    def _handle_outliers(
        self,
        data: pd.DataFrame,
        fit_transformers: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Maneja outliers en variables numéricas"""
        metadata = {
            'outliers_detected': {},
            'outliers_treated': {},
            'methods_used': {}
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            # Método IQR para detección de outliers
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identificar outliers
            outliers_mask = (data[column] < lower_bound) | (data[column] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count > 0:
                metadata['outliers_detected'][column] = outliers_count
                
                # Tratamiento de outliers: winsorización
                if fit_transformers:
                    from sklearn.preprocessing import FunctionTransformer
                    winsorizer = FunctionTransformer(
                        lambda x: np.clip(x, lower_bound, upper_bound)
                    )
                    data[column] = winsorizer.fit_transform(data[[column]]).flatten()
                    self.scalers[f'{column}_winsorizer'] = winsorizer
                else:
                    winsorizer = self.scalers.get(f'{column}_winsorizer')
                    if winsorizer:
                        data[column] = winsorizer.transform(data[[column]]).flatten()
                
                metadata['outliers_treated'][column] = outliers_count
                metadata['methods_used'][column] = 'winsorization'
        
        return data, metadata
    
    def _engineer_features(
        self,
        data: pd.DataFrame,
        fit_transformers: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Ingeniería de características para detección de fraude"""
        metadata = {
            'features_created': [],
            'feature_importance': {}
        }
        
        # Asegurar que tenemos columna de timestamp
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Extraer características temporales
            data['hour'] = data['timestamp'].dt.hour
            data['day_of_week'] = data['timestamp'].dt.dayofweek
            data['day_of_month'] = data['timestamp'].dt.day
            data['month'] = data['timestamp'].dt.month
            data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
            
            metadata['features_created'].extend([
                'hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend'
            ])
        
        # Características de monto
        if 'amount' in data.columns:
            # Log-transform del monto para manejar sesgo
            data['log_amount'] = np.log1p(data['amount'])
            metadata['features_created'].append('log_amount')
            
            # Categorías de monto
            if fit_transformers:
                from sklearn.preprocessing import KBinsDiscretizer
                amount_binner = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
                data['amount_category'] = amount_binner.fit_transform(data[['amount']]).flatten()
                self.scalers['amount_binner'] = amount_binner
            else:
                amount_binner = self.scalers.get('amount_binner')
                if amount_binner:
                    data['amount_category'] = amount_binner.transform(data[['amount']]).flatten()
            
            metadata['features_created'].append('amount_category')
        
        # Características de usuario (si hay historial)
        if 'user_id' in data.columns:
            # Estadísticas por usuario
            user_stats = data.groupby('user_id')['amount'].agg(['mean', 'std', 'count'])
            user_stats.columns = ['user_mean_amount', 'user_std_amount', 'user_transaction_count']
            
            data = data.merge(user_stats, left_on='user_id', right_index=True, how='left')
            
            # Ratio del monto actual vs promedio del usuario
            data['amount_vs_user_mean'] = data['amount'] / data['user_mean_amount']
            
            metadata['features_created'].extend([
                'user_mean_amount', 'user_std_amount', 'user_transaction_count', 'amount_vs_user_mean'
            ])
        
        # Características de comerciante
        if 'merchant_id' in data.columns:
            # Frecuencia de transacciones por comerciante
            merchant_freq = data.groupby('merchant_id').size()
            data['merchant_frequency'] = data['merchant_id'].map(merchant_freq)
            
            metadata['features_created'].append('merchant_frequency')
        
        # Características de ubicación
        if 'location' in data.columns:
            # Frecuencia por ubicación
            location_freq = data.groupby('location').size()
            data['location_frequency'] = data['location'].map(location_freq)
            
            metadata['features_created'].append('location_frequency')
        
        return data, metadata
    
    def _encode_categorical_features(
        self,
        data: pd.DataFrame,
        fit_transformers: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Codifica variables categóricas"""
        metadata = {
            'categorical_features': [],
            'encoding_methods': {},
            'feature_names': {}
        }
        
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        for column in categorical_columns:
            # Saltar columnas que no son características (IDs, timestamps)
            if column in ['transaction_id', 'user_id', 'merchant_id', 'timestamp']:
                continue
            
            unique_values = data[column].nunique()
            
            if unique_values <= 10:
                # One-hot encoding para pocas categorías
                if fit_transformers:
                    from sklearn.preprocessing import OneHotEncoder
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded_features = encoder.fit_transform(data[[column]])
                    
                    # Crear nombres de características
                    feature_names = [f"{column}_{val}" for val in encoder.categories_[0]]
                    
                    # Añadir características codificadas
                    for i, feature_name in enumerate(feature_names):
                        data[feature_name] = encoded_features[:, i]
                    
                    self.encoders[f'{column}_onehot'] = (encoder, feature_names)
                else:
                    encoder, feature_names = self.encoders.get(f'{column}_onehot', (None, None))
                    if encoder:
                        encoded_features = encoder.transform(data[[column]])
                        for i, feature_name in enumerate(feature_names):
                            data[feature_name] = encoded_features[:, i]
                
                metadata['encoding_methods'][column] = 'one_hot'
                metadata['feature_names'][column] = feature_names
                metadata['categorical_features'].append(column)
                
                # Eliminar columna original
                data = data.drop(column, axis=1)
            
            elif unique_values <= 50:
                # Target encoding para categorías moderadas
                if 'is_fraud' in data.columns and fit_transformers:
                    from category_encoders import TargetEncoder
                    encoder = TargetEncoder()
                    data[f'{column}_target_encoded'] = encoder.fit_transform(
                        data[column], data['is_fraud']
                    )
                    self.encoders[f'{column}_target'] = encoder
                    
                    metadata['encoding_methods'][column] = 'target'
                    metadata['categorical_features'].append(column)
                    
                    # Eliminar columna original
                    data = data.drop(column, axis=1)
                else:
                    # Si no hay target o no es entrenamiento, usar frequency encoding
                    freq_encoding = data[column].value_counts().to_dict()
                    data[f'{column}_freq_encoded'] = data[column].map(freq_encoding)
                    
                    metadata['encoding_methods'][column] = 'frequency'
                    metadata['categorical_features'].append(column)
                    
                    # Eliminar columna original
                    data = data.drop(column, axis=1)
            else:
                # Frequency encoding para muchas categorías
                freq_encoding = data[column].value_counts().to_dict()
                data[f'{column}_freq_encoded'] = data[column].map(freq_encoding)
                
                metadata['encoding_methods'][column] = 'frequency'
                metadata['categorical_features'].append(column)
                
                # Eliminar columna original
                data = data.drop(column, axis=1)
        
        return data, metadata
    
    def _scale_features(
        self,
        data: pd.DataFrame,
        fit_transformers: bool
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Escala características numéricas"""
        metadata = {
            'scaled_features': [],
            'scaling_method': 'standard',
            'scaler_params': {}
        }
        
        # Seleccionar columnas numéricas (excluyendo target)
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        if 'is_fraud' in numeric_columns:
            numeric_columns = numeric_columns.drop('is_fraud')
        
        if len(numeric_columns) > 0:
            if fit_transformers:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                self.scalers['standard_scaler'] = scaler
                
                metadata['scaler_params'] = {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                }
            else:
                scaler = self.scalers.get('standard_scaler')
                if scaler:
                    data[numeric_columns] = scaler.transform(data[numeric_columns])
            
            metadata['scaled_features'] = numeric_columns.tolist()
        
        return data, metadata
    
    def transform_new_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica transformaciones pre-entrenadas a nuevos datos.
        
        Args:
            data: DataFrame con nuevos datos a transformar
        
        Returns:
            DataFrame con datos transformados
        """
        processed_data, _ = self.preprocess_data(data, fit_transformers=False)
        return processed_data

# Uso del preprocesador
preprocessor_config = {
    'missing_value_strategy': 'median_mode',
    'outlier_treatment': 'winsorization',
    'feature_engineering': True,
    'categorical_encoding': 'auto',
    'scaling_method': 'standard'
}

preprocessor = FraudDataPreprocessor(preprocessor_config)
processed_data, preprocessing_metadata = preprocessor.preprocess_data(
    transaction_data,
    fit_transformers=True
)
```

### Fase 2: Feature Engineering para Detección de Fraude

#### Ingeniería de Características Avanzada
```python
class FraudFeatureEngineer:
    """
    Clase para ingeniería de características avanzada para detección de fraude.
    Crea características basadas en comportamiento temporal, patrones de uso y anomalías.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_importance = {}
    
    def create_behavioral_features(
        self,
        data: pd.DataFrame,
        historical_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Crea características basadas en comportamiento del usuario.
        
        Args:
            data: DataFrame con transacciones actuales
            historical_data: DataFrame con datos históricos (opcional)
        
        Returns:
            DataFrame con características de comportamiento añadidas
        """
        feature_data = data.copy()
        
        # Combinar con datos históricos si están disponibles
        if historical_data is not None:
            combined_data = pd.concat([historical_data, data], ignore_index=True)
        else:
            combined_data = data.copy()
        
        # Ordenar por usuario y timestamp
        combined_data = combined_data.sort_values(['user_id', 'timestamp'])
        
        # Características de tiempo entre transacciones
        combined_data['prev_timestamp'] = combined_data.groupby('user_id')['timestamp'].shift(1)
        combined_data['time_since_last_transaction'] = (
            combined_data['timestamp'] - combined_data['prev_timestamp']
        ).dt.total_seconds()
        
        # Características de frecuencia
        user_transaction_counts = combined_data.groupby('user_id').size()
        feature_data['user_transaction_frequency'] = feature_data['user_id'].map(user_transaction_counts)
        
        # Características de monto promedio por usuario
        user_avg_amount = combined_data.groupby('user_id')['amount'].mean()
        feature_data['user_avg_amount'] = feature_data['user_id'].map(user_avg_amount)
        
        # Ratio del monto actual vs promedio histórico
        feature_data['amount_vs_user_avg'] = feature_data['amount'] / feature_data['user_avg_amount']
        
        # Desviación estándar de montos por usuario
        user_std_amount = combined_data.groupby('user_id')['amount'].std()
        feature_data['user_std_amount'] = feature_data['user_id'].map(user_std_amount)
        
        # Z-score del monto actual vs histórico del usuario
        feature_data['amount_z_score'] = (
            feature_data['amount'] - feature_data['user_avg_amount']
        ) / feature_data['user_std_amount']
        
        # Características de patrón temporal
        user_hour_patterns = combined_data.groupby(['user_id', 'hour']).size().unstack(fill_value=0)
        user_hour_patterns = user_hour_patterns.div(user_hour_patterns.sum(axis=1), axis=0)
        
        # Entropía de patrones horarios (medida de aleatoriedad)
        from scipy.stats import entropy
        user_hour_entropy = user_hour_patterns.apply(entropy, axis=1)
        feature_data['user_hour_entropy'] = feature_data['user_id'].map(user_hour_entropy)
        
        # Características de comerciante preferido
        user_merchant_counts = combined_data.groupby(['user_id', 'merchant_id']).size().unstack(fill_value=0)
        user_preferred_merchant = user_merchant_counts.idxmax(axis=1)
        feature_data['user_preferred_merchant'] = feature_data['user_id'].map(user_preferred_merchant)
        
        # Ratio de transacciones con comerciante preferido
        user_total_transactions = user_merchant_counts.sum(axis=1)
        user_preferred_merchant_count = user_merchant_counts.max(axis=1)
        user_preferred_ratio = user_preferred_merchant_count / user_total_transactions
        feature_data['user_preferred_merchant_ratio'] = feature_data['user_id'].map(user_preferred_ratio)
        
        # Características de ubicación
        if 'location' in combined_data.columns:
            user_location_counts = combined_data.groupby(['user_id', 'location']).size().unstack(fill_value=0)
            user_location_diversity = (user_location_counts > 0).sum(axis=1)
            feature_data['user_location_diversity'] = feature_data['user_id'].map(user_location_diversity)
            
            # Cambio de ubicación respecto a transacción anterior
            combined_data['prev_location'] = combined_data.groupby('user_id')['location'].shift(1)
            feature_data['location_changed'] = (
                feature_data.index.map(
                    lambda x: combined_data.loc[x, 'location'] != combined_data.loc[x, 'prev_location']
                    if x in combined_data.index and pd.notna(combined_data.loc[x, 'prev_location'])
                    else False
                )
            ).astype(int)
        
        # Características de dispositivo (si está disponible)
        if 'device_id' in combined_data.columns:
            user_device_counts = combined_data.groupby(['user_id', 'device_id']).size().unstack(fill_value=0)
            user_device_diversity = (user_device_counts > 0).sum(axis=1)
            feature_data['user_device_diversity'] = feature_data['user_id'].map(user_device_diversity)
            
            # Cambio de dispositivo respecto a transacción anterior
            combined_data['prev_device'] = combined_data.groupby('user_id')['device_id'].shift(1)
            feature_data['device_changed'] = (
                feature_data.index.map(
                    lambda x: combined_data.loc[x, 'device_id'] != combined_data.loc[x, 'prev_device']
                    if x in combined_data.index and pd.notna(combined_data.loc[x, 'prev_device'])
                    else False
                )
            ).astype(int)
        
        return feature_data
    
    def create_network_features(
        self,
        data: pd.DataFrame,
        user_network_data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Crea características basadas en análisis de redes (usuario-comerciante).
        
        Args:
            data: DataFrame con transacciones
            user_network_data: Datos de red de usuarios (opcional)
        
        Returns:
            DataFrame con características de red añadidas
        """
        feature_data = data.copy()
        
        # Construir grafo usuario-comerciante
        import networkx as nx
        
        # Crear grafo bipartito usuario-comerciante
        G = nx.Graph()
        
        # Añadir nodos y aristas
        for _, row in data.iterrows():
            user_id = row['user_id']
            merchant_id = row['merchant_id']
            amount = row['amount']
            
            G.add_node(user_id, type='user')
            G.add_node(merchant_id, type='merchant')
            G.add_edge(user_id, merchant_id, weight=amount)
        
        # Características de centralidad para usuarios
        user_degree_centrality = nx.degree_centrality(G)
        user_betweenness_centrality = nx.betweenness_centrality(G)
        
        # Filtrar solo usuarios
        user_centrality = {
            node: user_degree_centrality[node] 
            for node in user_degree_centrality 
            if G.nodes[node]['type'] == 'user'
        }
        
        user_betweenness = {
            node: user_betweenness_centrality[node] 
            for node in user_betweenness_centrality 
            if G.nodes[node]['type'] == 'user'
        }
        
        feature_data['user_degree_centrality'] = feature_data['user_id'].map(user_centrality).fillna(0)
        feature_data['user_betweenness_centrality'] = feature_data['user_id'].map(user_betweenness).fillna(0)
        
        # Características de comunidad (detección de comunidades)
        from networkx.algorithms import community
        
        # Detectar comunidades usando algoritmo de Louvain
        communities = community.louvain_communities(G)
        
        # Asignar comunidad a cada nodo
        node_community = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_community[node] = i
        
        feature_data['user_community'] = feature_data['user_id'].map(node_community).fillna(-1)
        feature_data['merchant_community'] = feature_data['merchant_id'].map(node_community).fillna(-1)
        
        # Características de similitud con otros usuarios
        user_neighbors = {}
        for user in [n for n in G.nodes() if G.nodes[n]['type'] == 'user']:
            user_neighbors[user] = set(G.neighbors(user))
        
        def jaccard_similarity(set1, set2):
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0
        
        # Calcular similitud con usuarios similares (para rendimiento, usar muestra)
        user_similarity = {}
        users = list(user_neighbors.keys())
        
        for i, user1 in enumerate(users):
            similarities = []
            for j, user2 in enumerate(users):
                if i != j:
                    sim = jaccard_similarity(user_neighbors[user1], user_neighbors[user2])
                    similarities.append(sim)
            
            user_similarity[user1] = max(similarities) if similarities else 0
        
        feature_data['user_max_similarity'] = feature_data['user_id'].map(user_similarity).fillna(0)
        
        return feature_data
    
    def create_temporal_features(
        self,
        data: pd.DataFrame,
        window_sizes: List[int] = [1, 7, 30]
    ) -> pd.DataFrame:
        """
        Crea características basadas en patrones temporales con diferentes ventanas.
        
        Args:
            data: DataFrame con transacciones
            window_sizes: Lista de tamaños de ventana en días
        
        Returns:
            DataFrame con características temporales añadidas
        """
        feature_data = data.copy()
        
        # Asegurar que tenemos timestamp
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        
        for window_size in window_sizes:
            window_suffix = f"_{window_size}d"
            
            # Características de usuario por ventana
            user_window_stats = data.set_index('timestamp').groupby('user_id').rolling(
                f'{window_size}D'
            ).agg({
                'amount': ['count', 'mean', 'std', 'sum', 'max'],
                'is_fraud': 'sum' if 'is_fraud' in data.columns else 'count'
            })
            
            # Aplanar multiíndice
            user_window_stats.columns = [
                f'user_{col[0]}{window_suffix}' 
                for col in user_window_stats.columns
            ]
            
            # Unir con datos originales
            feature_data = feature_data.merge(
                user_window_stats,
                left_on=['user_id', 'timestamp'],
                right_index=True,
                how='left'
            )
            
            # Características de comerciante por ventana
            merchant_window_stats = data.set_index('timestamp').groupby('merchant_id').rolling(
                f'{window_size}D'
            ).agg({
                'amount': ['count', 'mean', 'std'],
                'is_fraud': 'sum' if 'is_fraud' in data.columns else 'count'
            })
            
            merchant_window_stats.columns = [
                f'merchant_{col[0]}{window_suffix}' 
                for col in merchant_window_stats.columns
            ]
            
            feature_data = feature_data.merge(
                merchant_window_stats,
                left_on=['merchant_id', 'timestamp'],
                right_index=True,
                how='left'
            )
            
            # Ratio de actividad actual vs ventana
            if f'user_amountcount{window_suffix}' in feature_data.columns:
                feature_data[f'user_activity_ratio{window_suffix}'] = (
                    1 / (feature_data[f'user_amountcount{window_suffix}'] + 1)
                )
            
            # Tendencia de montos (pendiente de regresión lineal)
            def calculate_trend(group):
                if len(group) < 2:
                    return 0
                
                x = np.arange(len(group))
                y = group['amount'].values
                
                # Calcular pendiente
                slope = np.polyfit(x, y, 1)[0]
                return slope
            
            user_trends = data.groupby('user_id').rolling(
                f'{window_size}D'
            ).apply(calculate_trend)
            
            feature_data[f'user_amount_trend{window_suffix}'] = user_trends.values
        
        # Características de estacionalidad
        if 'hour' in feature_data.columns:
            # Patrones por hora del día
            hour_patterns = data.groupby(['user_id', 'hour']).size().unstack(fill_value=0)
            hour_patterns_norm = hour_patterns.div(hour_patterns.sum(axis=1), axis=0)
            
            for hour in range(24):
                if hour in hour_patterns_norm.columns:
                    feature_data[f'user_hour_{hour}_ratio'] = feature_data['user_id'].map(
                        hour_patterns_norm[hour]
                    ).fillna(0)
        
        if 'day_of_week' in feature_data.columns:
            # Patrones por día de la semana
            dow_patterns = data.groupby(['user_id', 'day_of_week']).size().unstack(fill_value=0)
            dow_patterns_norm = dow_patterns.div(dow_patterns.sum(axis=1), axis=0)
            
            for dow in range(7):
                if dow in dow_patterns_norm.columns:
                    feature_data[f'user_dow_{dow}_ratio'] = feature_data['user_id'].map(
                        dow_patterns_norm[dow]
                    ).fillna(0)
        
        return feature_data
    
    def create_anomaly_features(
        self,
        data: pd.DataFrame,
        anomaly_detector_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Crea características basadas en detección de anomalías.
        
        Args:
            data: DataFrame con transacciones
            anomaly_detector_config: Configuración para detectores de anomalías
        
        Returns:
            DataFrame con características de anomalías añadidas
        """
        feature_data = data.copy()
        
        # Seleccionar características numéricas para detección de anomalías
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Excluir características que no son útiles para detección de anomalías
        exclude_features = ['is_fraud', 'transaction_id']
        numeric_features = [f for f in numeric_features if f not in exclude_features]
        
        if not numeric_features:
            return feature_data
        
        # Isolation Forest para detección de anomalías
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=anomaly_detector_config.get('contamination', 0.1),
            random_state=42
        )
        
        # Ajustar y predecir anomalías
        anomaly_scores = iso_forest.fit_predict(data[numeric_features])
        feature_data['isolation_forest_anomaly'] = anomaly_scores
        feature_data['isolation_forest_score'] = iso_forest.decision_function(data[numeric_features])
        
        # Local Outlier Factor (LOF)
        from sklearn.neighbors import LocalOutlierFactor
        
        lof = LocalOutlierFactor(
            n_neighbors=anomaly_detector_config.get('n_neighbors', 20),
            contamination=anomaly_detector_config.get('contamination', 0.1)
        )
        
        lof_anomaly_scores = lof.fit_predict(data[numeric_features])
        feature_data['lof_anomaly'] = lof_anomaly_scores
        
        # One-Class SVM
        from sklearn.svm import OneClassSVM
        
        oc_svm = OneClassSVM(
            nu=anomaly_detector_config.get('nu', 0.1),
            kernel='rbf',
            gamma='scale'
        )
        
        svm_anomaly_scores = oc_svm.fit_predict(data[numeric_features])
        feature_data['svm_anomaly'] = svm_anomaly_scores
        
        # Características de ensemble de anomalías
        feature_data['anomaly_ensemble_score'] = (
            (feature_data['isolation_forest_anomaly'] == -1).astype(int) +
            (feature_data['lof_anomaly'] == -1).astype(int) +
            (feature_data['svm_anomaly'] == -1).astype(int)
        ) / 3
        
        feature_data['is_anomaly_ensemble'] = (feature_data['anomaly_ensemble_score'] > 0.5).astype(int)
        
        # Características de distancia a centroides
        from sklearn.cluster import KMeans
        
        n_clusters = anomaly_detector_config.get('n_clusters', 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        clusters = kmeans.fit_predict(data[numeric_features])
        feature_data['cluster'] = clusters
        
        # Calcular distancia a centroide del cluster
        distances = kmeans.transform(data[numeric_features])
        feature_data['distance_to_centroid'] = distances.min(axis=1)
        
        # Z-score de distancia a centroide
        mean_distance = feature_data['distance_to_centroid'].mean()
        std_distance = feature_data['distance_to_centroid'].std()
        feature_data['distance_z_score'] = (
            feature_data['distance_to_centroid'] - mean_distance
        ) / std_distance
        
        feature_data['is_distance_outlier'] = (
            feature_data['distance_z_score'].abs() > 3
        ).astype(int)
        
        return feature_data
    
    def select_features(
        self,
        data: pd.DataFrame,
        target_column: str,
        selection_method: str = 'importance',
        n_features: int = 50
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Selecciona características más relevantes para detección de fraude.
        
        Args:
            data: DataFrame con características
            target_column: Nombre de columna objetivo
            selection_method: Método de selección ('importance', 'correlation', 'mutual_info')
            n_features: Número de características a seleccionar
        
        Returns:
            Tupla con DataFrame filtrado y lista de características seleccionadas
        """
        # Separar características y target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Eliminar columnas no numéricas
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        if selection_method == 'importance':
            # Usar Random Forest para importancia de características
            from sklearn.ensemble import RandomForestClassifier
            
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            
            # Obtener importancia de características
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Seleccionar top N características
            selected_features = feature_importance.head(n_features)['feature'].tolist()
            self.feature_importance = feature_importance.set_index('feature')['importance'].to_dict()
            
        elif selection_method == 'correlation':
            # Usar correlación con target
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected_features = correlations.head(n_features).index.tolist()
            self.feature_importance = correlations.to_dict()
            
        elif selection_method == 'mutual_info':
            # Usar información mutua
            from sklearn.feature_selection import mutual_info_classif
            
            mi_scores = mutual_info_classif(X, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
            selected_features = mi_scores.head(n_features).index.tolist()
            self.feature_importance = mi_scores.to_dict()
        
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        # Filtrar dataset
        filtered_data = data[selected_features + [target_column]]
        
        return filtered_data, selected_features

# Uso del ingeniero de características
feature_engineer_config = {
    'behavioral_features': True,
    'network_features': True,
    'temporal_features': True,
    'anomaly_features': True,
    'feature_selection': {
        'method': 'importance',
        'n_features': 50
    },
    'anomaly_detection': {
        'contamination': 0.1,
        'n_neighbors': 20,
        'nu': 0.1,
        'n_clusters': 5
    }
}

feature_engineer = FraudFeatureEngineer(feature_engineer_config)

# Crear características de comportamiento
behavioral_features = feature_engineer.create_behavioral_features(
    processed_data,
    historical_data=None
)

# Crear características de red
network_features = feature_engineer.create_network_features(behavioral_features)

# Crear características temporales
temporal_features = feature_engineer.create_temporal_features(
    network_features,
    window_sizes=[1, 7, 30]
)

# Crear características de anomalías
anomaly_features = feature_engineer.create_anomaly_features(
    temporal_features,
    feature_engineer_config['anomaly_detection']
)

# Seleccionar características más importantes
if 'is_fraud' in anomaly_features.columns:
    final_features, selected_feature_list = feature_engineer.select_features(
        anomaly_features,
        'is_fraud',
        selection_method='importance',
        n_features=50
    )
else:
    final_features = anomaly_features
    selected_feature_list = anomaly_features.select_dtypes(include=[np.number]).columns.tolist()
```

### Fase 3: Entrenamiento y Validación de Modelos

#### Entrenamiento de Modelos para Detección de Fraude
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import joblib
import logging
from datetime import datetime

class FraudModelTrainer:
    """
    Clase para entrenamiento y validación de modelos de detección de fraude.
    Soporta múltiples algoritmos y técnicas de ensemble.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.model_performance = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = {}
    
    def prepare_training_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        validation_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepara datos para entrenamiento, validación y prueba.
        
        Args:
            data: DataFrame con datos completos
            target_column: Nombre de columna objetivo
            test_size: Proporción para conjunto de prueba
            validation_size: Proporción para conjunto de validación
            random_state: Semilla para reproducibilidad
        
        Returns:
            Tupla con (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Separar características y target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Dividir entrenamiento en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=validation_size, random_state=random_state, stratify=y_train
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.info(f"Fraud rate - Train: {y_train.mean():.4f}, Val: {y_val.mean():.4f}, Test: {y_test.mean():.4f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_isolation_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrena modelo Isolation Forest para detección de anomalías.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            model_config: Configuración del modelo
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        from sklearn.ensemble import IsolationForest
        
        # Configurar modelo
        iso_forest = IsolationForest(
            n_estimators=model_config.get('n_estimators', 100),
            max_samples=model_config.get('max_samples', 'auto'),
            contamination=model_config.get('contamination', 0.1),
            max_features=model_config.get('max_features', 1.0),
            bootstrap=model_config.get('bootstrap', False),
            n_jobs=model_config.get('n_jobs', -1),
            random_state=42
        )
        
        # Entrenar modelo
        self.logger.info("Training Isolation Forest model...")
        iso_forest.fit(X_train)
        
        # Predecir anomalías
        train_predictions = iso_forest.predict(X_train)
        val_predictions = iso_forest.predict(X_val)
        
        # Convertir predicciones a etiquetas binarias (1: normal, -1: anomalía)
        train_binary = (train_predictions == 1).astype(int)
        val_binary = (val_predictions == 1).astype(int)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, train_binary)
        val_metrics = self._calculate_metrics(y_val, val_binary)
        
        # Obtener scores de anomalía
        train_scores = iso_forest.decision_function(X_train)
        val_scores = iso_forest.decision_function(X_val)
        
        # Calcular AUC-ROC usando scores
        train_auc = roc_auc_score(y_train, -train_scores)  # Negativo porque menor score = más anómalo
        val_auc = roc_auc_score(y_val, -val_scores)
        
        train_metrics['auc_roc'] = train_auc
        val_metrics['auc_roc'] = val_auc
        
        # Guardar modelo
        model_name = "isolation_forest"
        self.models[model_name] = iso_forest
        
        # Guardar importancia de características
        if hasattr(iso_forest, 'feature_importances_'):
            self.feature_importance[model_name] = dict(
                zip(X_train.columns, iso_forest.feature_importances_)
            )
        
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model_config,
            'feature_importance': self.feature_importance.get(model_name, {})
        }
        
        self.model_performance[model_name] = results
        
        self.logger.info(f"Isolation Forest - Val AUC: {val_auc:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return results
    
    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrena modelo Random Forest para detección de fraude.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            model_config: Configuración del modelo
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Configurar modelo
        rf = RandomForestClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', None),
            min_samples_split=model_config.get('min_samples_split', 2),
            min_samples_leaf=model_config.get('min_samples_leaf', 1),
            max_features=model_config.get('max_features', 'sqrt'),
            bootstrap=model_config.get('bootstrap', True),
            class_weight=model_config.get('class_weight', 'balanced'),
            random_state=42,
            n_jobs=model_config.get('n_jobs', -1)
        )
        
        # Entrenar modelo
        self.logger.info("Training Random Forest model...")
        rf.fit(X_train, y_train)
        
        # Predecir probabilidades
        train_probs = rf.predict_proba(X_train)[:, 1]
        val_probs = rf.predict_proba(X_val)[:, 1]
        
        # Predecir etiquetas
        train_preds = rf.predict(X_train)
        val_preds = rf.predict(X_val)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, train_preds, train_probs)
        val_metrics = self._calculate_metrics(y_val, val_preds, val_probs)
        
        # Guardar modelo
        model_name = "random_forest"
        self.models[model_name] = rf
        
        # Guardar importancia de características
        self.feature_importance[model_name] = dict(
            zip(X_train.columns, rf.feature_importances_)
        )
        
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model_config,
            'feature_importance': self.feature_importance[model_name]
        }
        
        self.model_performance[model_name] = results
        
        self.logger.info(f"Random Forest - Val AUC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return results
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrena modelo XGBoost para detección de fraude.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            model_config: Configuración del modelo
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        try:
            import xgboost as xgb
        except ImportError:
            self.logger.error("XGBoost not installed. Skipping XGBoost training.")
            return {}
        
        # Configurar modelo
        xgb_model = xgb.XGBClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', 6),
            learning_rate=model_config.get('learning_rate', 0.1),
            subsample=model_config.get('subsample', 1.0),
            colsample_bytree=model_config.get('colsample_bytree', 1.0),
            scale_pos_weight=model_config.get('scale_pos_weight', 1.0),
            random_state=42,
            n_jobs=model_config.get('n_jobs', -1),
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        # Entrenar modelo
        self.logger.info("Training XGBoost model...")
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=model_config.get('early_stopping_rounds', 10),
            verbose=False
        )
        
        # Predecir probabilidades
        train_probs = xgb_model.predict_proba(X_train)[:, 1]
        val_probs = xgb_model.predict_proba(X_val)[:, 1]
        
        # Predecir etiquetas
        train_preds = xgb_model.predict(X_train)
        val_preds = xgb_model.predict(X_val)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, train_preds, train_probs)
        val_metrics = self._calculate_metrics(y_val, val_preds, val_probs)
        
        # Guardar modelo
        model_name = "xgboost"
        self.models[model_name] = xgb_model
        
        # Guardar importancia de características
        self.feature_importance[model_name] = dict(
            zip(X_train.columns, xgb_model.feature_importances_)
        )
        
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model_config,
            'feature_importance': self.feature_importance[model_name],
            'best_iteration': xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else None
        }
        
        self.model_performance[model_name] = results
        
        self.logger.info(f"XGBoost - Val AUC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return results
    
    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrena modelo LightGBM para detección de fraude.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            model_config: Configuración del modelo
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        try:
            import lightgbm as lgb
        except ImportError:
            self.logger.error("LightGBM not installed. Skipping LightGBM training.")
            return {}
        
        # Configurar modelo
        lgb_model = lgb.LGBMClassifier(
            n_estimators=model_config.get('n_estimators', 100),
            max_depth=model_config.get('max_depth', -1),
            learning_rate=model_config.get('learning_rate', 0.1),
            num_leaves=model_config.get('num_leaves', 31),
            subsample=model_config.get('subsample', 1.0),
            colsample_bytree=model_config.get('colsample_bytree', 1.0),
            class_weight=model_config.get('class_weight', 'balanced'),
            random_state=42,
            n_jobs=model_config.get('n_jobs', -1),
            verbose=-1
        )
        
        # Entrenar modelo
        self.logger.info("Training LightGBM model...")
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=model_config.get('early_stopping_rounds', 10),
                    verbose=False
                )
            ]
        )
        
        # Predecir probabilidades
        train_probs = lgb_model.predict_proba(X_train)[:, 1]
        val_probs = lgb_model.predict_proba(X_val)[:, 1]
        
        # Predecir etiquetas
        train_preds = lgb_model.predict(X_train)
        val_preds = lgb_model.predict(X_val)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, train_preds, train_probs)
        val_metrics = self._calculate_metrics(y_val, val_preds, val_probs)
        
        # Guardar modelo
        model_name = "lightgbm"
        self.models[model_name] = lgb_model
        
        # Guardar importancia de características
        self.feature_importance[model_name] = dict(
            zip(X_train.columns, lgb_model.feature_importances_)
        )
        
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model_config,
            'feature_importance': self.feature_importance[model_name],
            'best_iteration': lgb_model.best_iteration_ if hasattr(lgb_model, 'best_iteration_') else None
        }
        
        self.model_performance[model_name] = results
        
        self.logger.info(f"LightGBM - Val AUC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return results
    
    def train_neural_network(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrena modelo de red neuronal para detección de fraude.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            model_config: Configuración del modelo
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        except ImportError:
            self.logger.error("TensorFlow not installed. Skipping Neural Network training.")
            return {}
        
        # Configurar modelo
        model = Sequential()
        
        # Capa de entrada
        model.add(Dense(
            units=model_config.get('input_units', 128),
            activation='relu',
            input_shape=(X_train.shape[1],)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(model_config.get('dropout_rate', 0.2)))
        
        # Capas ocultas
        for units in model_config.get('hidden_units', [64, 32]):
            model.add(Dense(units=units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(model_config.get('dropout_rate', 0.2)))
        
        # Capa de salida
        model.add(Dense(units=1, activation='sigmoid'))
        
        # Compilar modelo
        model.compile(
            optimizer=Adam(learning_rate=model_config.get('learning_rate', 0.001)),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        # Configurar callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=model_config.get('early_stopping_patience', 10),
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=model_config.get('lr_patience', 5),
                min_lr=1e-6
            )
        ]
        
        # Calcular class weights para manejar desbalance
        class_weight = {
            0: 1.0,
            1: model_config.get('fraud_weight', 10.0)
        }
        
        # Entrenar modelo
        self.logger.info("Training Neural Network model...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=model_config.get('epochs', 100),
            batch_size=model_config.get('batch_size', 32),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=0
        )
        
        # Predecir probabilidades
        train_probs = model.predict(X_train).flatten()
        val_probs = model.predict(X_val).flatten()
        
        # Predecir etiquetas
        train_preds = (train_probs > 0.5).astype(int)
        val_preds = (val_probs > 0.5).astype(int)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, train_preds, train_probs)
        val_metrics = self._calculate_metrics(y_val, val_preds, val_probs)
        
        # Guardar modelo
        model_name = "neural_network"
        self.models[model_name] = model
        
        # Guardar historial de entrenamiento
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_config': model_config,
            'training_history': history.history,
            'best_epoch': np.argmin(history.history['val_loss']) + 1
        }
        
        self.model_performance[model_name] = results
        
        self.logger.info(f"Neural Network - Val AUC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return results
    
    def train_ensemble_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        ensemble_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Entrena modelo ensemble combinando múltiples modelos.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Target de entrenamiento
            X_val: Características de validación
            y_val: Target de validación
            ensemble_config: Configuración del ensemble
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        from sklearn.ensemble import VotingClassifier, StackingClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Obtener modelos base
        base_models = []
        
        for model_name, model in self.models.items():
            if model_name in ensemble_config.get('base_models', []):
                base_models.append((model_name, model))
        
        if not base_models:
            self.logger.error("No base models found for ensemble training.")
            return {}
        
        # Configurar ensemble
        ensemble_method = ensemble_config.get('method', 'voting')
        
        if ensemble_method == 'voting':
            ensemble = VotingClassifier(
                estimators=base_models,
                voting=ensemble_config.get('voting', 'soft'),
                n_jobs=ensemble_config.get('n_jobs', -1)
            )
        elif ensemble_method == 'stacking':
            meta_estimator = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
            
            ensemble = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_estimator,
                cv=ensemble_config.get('cv', 5),
                n_jobs=ensemble_config.get('n_jobs', -1)
            )
        else:
            self.logger.error(f"Unknown ensemble method: {ensemble_method}")
            return {}
        
        # Entrenar ensemble
        self.logger.info(f"Training {ensemble_method} ensemble model...")
        ensemble.fit(X_train, y_train)
        
        # Predecir probabilidades
        train_probs = ensemble.predict_proba(X_train)[:, 1]
        val_probs = ensemble.predict_proba(X_val)[:, 1]
        
        # Predecir etiquetas
        train_preds = ensemble.predict(X_train)
        val_preds = ensemble.predict(X_val)
        
        # Calcular métricas
        train_metrics = self._calculate_metrics(y_train, train_preds, train_probs)
        val_metrics = self._calculate_metrics(y_val, val_preds, val_probs)
        
        # Guardar modelo
        model_name = f"{ensemble_method}_ensemble"
        self.models[model_name] = ensemble
        
        results = {
            'model_name': model_name,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'ensemble_config': ensemble_config,
            'base_models': [name for name, _ in base_models]
        }
        
        self.model_performance[model_name] = results
        
        self.logger.info(f"{ensemble_method.title()} Ensemble - Val AUC: {val_metrics['auc_roc']:.4f}, Val F1: {val_metrics['f1']:.4f}")
        
        return results
    
    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calcula métricas de evaluación para el modelo.
        
        Args:
            y_true: Valores verdaderos
            y_pred: Predicciones del modelo
            y_prob: Probabilidades predichas (opcional)
        
        Returns:
            Diccionario con métricas calculadas
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # Calcular AUC-ROC si se proporcionan probabilidades
        if y_prob is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
                metrics['average_precision'] = average_precision_score(y_true, y_prob)
            except Exception as e:
                self.logger.warning(f"Error calculating AUC metrics: {str(e)}")
                metrics['auc_roc'] = 0.0
                metrics['average_precision'] = 0.0
        
        return metrics
    
    def select_best_model(
        self,
        metric: str = 'auc_roc',
        higher_is_better: bool = True
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Selecciona el mejor modelo basado en métrica especificada
