---
title: "01 - Directrices de Implementación"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# 01 - Directrices de Implementación

Estándares de código, manejo de errores, logging estructurado y trazabilidad.
Requisitos de pruebas y CI.

## Checklist de Calidad para Implementación
- [ ] Estándares de código limpio aplicados
- [ ] Manejo de errores con jerarquía de excepciones personalizada
- [ ] Logging estructurado configurado
- [ ] Trazabilidad y auditoría implementada
- [ ] Tests unitarios, integración y performance ejecutados
- [ ] Pipeline CI/CD configurado y funcionando

## Estándares de Código para Detección de Fraude y Anomalías

### Principios de Código Limpio en Contexto de Detección de Fraude

#### Legibilidad sobre Complejidad
```python
# MAL: Código complejo difícil de entender
def calc_fraud(data, model, thresh=0.8):
    import numpy as np
    pred = model.predict(np.array([list(data.values())]))
    return pred[0] > thresh

# BIEN: Código claro y autodocumentado
def calculate_fraud_probability(
    transaction_features: Dict[str, float],
    fraud_detection_model: Any,
    probability_threshold: float = 0.8
) -> Tuple[bool, float]:
    """
    Calcula la probabilidad de fraude para una transacción.
    
    Args:
        transaction_features: Diccionario con características de la transacción
        fraud_detection_model: Modelo entrenado para detección de fraude
        probability_threshold: Umbral para clasificar como fraude
    
    Returns:
        Tupla con (es_fraude, probabilidad_fraude)
    """
    # Convertir características a formato esperado por el modelo
    feature_vector = prepare_feature_vector(transaction_features)
    
    # Obtener probabilidad de fraude del modelo
    fraud_probability = fraud_detection_model.predict_proba(feature_vector)[0][1]
    
    # Determinar si es fraude basado en el umbral
    is_fraud = fraud_probability > probability_threshold
    
    return is_fraud, fraud_probability
```

#### Nombres Significativos en Detección de Fraude
```python
# MAL: Nombres genéricos
def process_data(d1, d2, t=0.05):
    return d1 * (1 + t) - d2

# BIEN: Nombres específicos del dominio
def detect_anomalous_transaction_pattern(
    transaction_history: List[Dict[str, Any]],
    current_transaction: Dict[str, Any],
    anomaly_threshold: float = 2.5
) -> Dict[str, Any]:
    """
    Detecta patrones anómalos en transacciones comparando con historial.
    
    Args:
        transaction_history: Lista de transacciones anteriores del usuario
        current_transaction: Transacción actual a evaluar
        anomaly_threshold: Umbral de puntuación Z para considerar anomalía
    
    Returns:
        Diccionario con resultados de detección de anomalías
    """
    # Extraer características relevantes del historial
    historical_amounts = [tx.get('amount', 0) for tx in transaction_history]
    historical_times = [tx.get('timestamp', 0) for tx in transaction_history]
    
    # Calcular estadísticas del historial
    mean_amount = np.mean(historical_amounts)
    std_amount = np.std(historical_amounts)
    
    # Calcular puntuación Z para el monto actual
    current_amount = current_transaction.get('amount', 0)
    z_score = abs((current_amount - mean_amount) / std_amount) if std_amount > 0 else 0
    
    # Detectar anomalía basada en puntuación Z
    is_anomalous = z_score > anomaly_threshold
    
    return {
        'is_anomalous': is_anomalous,
        'z_score': z_score,
        'mean_historical_amount': mean_amount,
        'current_amount': current_amount,
        'anomaly_threshold': anomaly_threshold
    }
```

### Manejo de Errores en Sistemas de Detección de Fraude

#### Jerarquía de Excepciones Personalizadas
```python
class CerverusException(Exception):
    """Base exception for Cerverus system"""
    pass

class DataQualityError(CerverusException):
    """Error when data doesn't meet quality standards"""
    pass

class ModelPredictionError(CerverusException):
    """Error when model prediction fails"""
    pass

class FraudDetectionError(CerverusException):
    """Error in fraud detection process"""
    pass

class AnomalyDetectionError(CerverusException):
    """Error in anomaly detection process"""
    pass

# Uso en funciones críticas
def detect_fraud_with_ensemble(
    transaction_data: Dict[str, Any],
    models: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Detecta fraude usando ensemble de múltiples modelos.
    
    Args:
        transaction_data: Datos de la transacción a analizar
        models: Diccionario con modelos de detección de fraude
        config: Configuración de parámetros de detección
    
    Returns:
        Diccionario con resultados de detección de fraude
    
    Raises:
        DataQualityError: Si los datos de entrada son inválidos
        ModelPredictionError: Si algún modelo falla en predicción
        FraudDetectionError: Si el proceso de detección falla
    """
    try:
        # Validar datos de entrada
        if not transaction_data:
            raise DataQualityError("Transaction data cannot be empty")
        
        required_fields = ['amount', 'timestamp', 'user_id', 'merchant_id']
        missing_fields = [field for field in required_fields if field not in transaction_data]
        
        if missing_fields:
            raise DataQualityError(f"Missing required fields: {missing_fields}")
        
        # Validar que los modelos estén cargados
        if not models:
            raise ModelPredictionError("No fraud detection models available")
        
        # Ejecutar predicciones con cada modelo
        predictions = {}
        for model_name, model in models.items():
            try:
                prediction = model.predict(transaction_data)
                predictions[model_name] = prediction
            except Exception as e:
                raise ModelPredictionError(f"Model {model_name} prediction failed: {str(e)}")
        
        # Combinar predicciones usando método configurado
        ensemble_method = config.get('ensemble_method', 'weighted_average')
        final_prediction = combine_predictions(predictions, ensemble_method)
        
        # Calcular confianza de la predicción
        confidence = calculate_prediction_confidence(predictions)
        
        return {
            'is_fraud': final_prediction > config.get('fraud_threshold', 0.8),
            'fraud_probability': final_prediction,
            'confidence': confidence,
            'individual_predictions': predictions,
            'ensemble_method': ensemble_method
        }
        
    except Exception as e:
        if isinstance(e, (DataQualityError, ModelPredictionError)):
            raise
        raise FraudDetectionError(f"Unexpected error in fraud detection: {str(e)}")
```

#### Manejo de Errores en APIs de Detección de Fraude
```python
from fastapi import HTTPException, status
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FraudAPIError(HTTPException):
    """Base error for fraud detection API endpoints"""
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        details: Optional[dict] = None
    ):
        super().__init__(status_code=status_code, detail={
            "error_code": error_code,
            "message": message,
            "details": details or {}
        })
        self.error_code = error_code

# Errores específicos del dominio de detección de fraude
class InvalidTransactionDataError(FraudAPIError):
    def __init__(self, field_name: str, reason: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_TRANSACTION_DATA",
            message=f"Invalid transaction data for field: {field_name}",
            details={"field": field_name, "reason": reason}
        )

class ModelNotAvailableError(FraudAPIError):
    def __init__(self, model_name: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code="MODEL_NOT_AVAILABLE",
            message=f"Fraud detection model not available: {model_name}",
            details={"model_name": model_name}
        )

class HighRiskTransactionError(FraudAPIError):
    def __init__(self, fraud_score: float, threshold: float):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="HIGH_RISK_TRANSACTION",
            message=f"Transaction exceeds fraud risk threshold",
            details={
                "fraud_score": fraud_score,
                "threshold": threshold,
                "action_required": "manual_review"
            }
        )

# Uso en endpoints
@app.post("/api/v1/fraud/detect")
async def detect_fraud_endpoint(transaction_data: Dict[str, Any]):
    """
    Detecta potencial fraude en una transacción.
    
    Args:
        transaction_data: Datos de la transacción a analizar
    
    Returns:
        Resultados de detección de fraude
    """
    try:
        # Validar datos de entrada
        validate_transaction_data(transaction_data)
        
        # Obtener modelos de detección de fraude
        fraud_models = model_registry.get_fraud_models()
        if not fraud_models:
            raise ModelNotAvailableError("default_fraud_model")
        
        # Obtener configuración
        fraud_config = config_service.get_fraud_detection_config()
        
        # Detectar fraude
        fraud_result = fraud_detection_service.detect_fraud_with_ensemble(
            transaction_data=transaction_data,
            models=fraud_models,
            config=fraud_config
        )
        
        # Si es fraude de alto riesgo, generar alerta
        if fraud_result['is_fraud'] and fraud_result['fraud_probability'] > 0.95:
            alert_service.create_high_risk_alert(
                transaction_id=transaction_data.get('transaction_id'),
                fraud_score=fraud_result['fraud_probability'],
                reason="High fraud probability detected"
            )
            raise HighRiskTransactionError(
                fraud_result['fraud_probability'],
                fraud_config.get('fraud_threshold', 0.8)
            )
        
        return {
            "transaction_id": transaction_data.get('transaction_id'),
            "fraud_detection_result": fraud_result,
            "processed_at": datetime.utcnow().isoformat()
        }
        
    except InvalidTransactionDataError as e:
        logger.warning(f"Invalid transaction data: {transaction_data.get('transaction_id')}")
        raise e
    except ModelNotAvailableError as e:
        logger.error(f"Fraud detection model not available")
        raise e
    except HighRiskTransactionError as e:
        logger.warning(f"High risk transaction detected: {transaction_data.get('transaction_id')}")
        raise e
    except Exception as e:
        logger.error(f"Unexpected error in fraud detection: {str(e)}")
        raise FraudAPIError(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="INTERNAL_ERROR",
            message="Internal server error in fraud detection"
        )
```

## Logging Estructurado para Sistemas de Detección de Fraude

### Configuración de Logging con Structured Logging
```python
import structlog
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class FraudDetectionLogger:
    """
    Logger estructurado para sistemas de detección de fraude con soporte para
    auditoría y cumplimiento regulatorio.
    """
    
    def __init__(self, service_name: str, environment: str = "production"):
        self.service_name = service_name
        self.environment = environment
        self.logger = structlog.get_logger(service_name)
        
        # Configurar formato de logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                self._add_fraud_detection_metadata,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _add_fraud_detection_metadata(self, logger, method_name, event_dict):
        """Añade metadata específica para sistemas de detección de fraude"""
        event_dict.update({
            "service": self.service_name,
            "environment": self.environment,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        })
        return event_dict
    
    def log_fraud_detection(
        self,
        transaction_id: str,
        user_id: str,
        fraud_score: float,
        confidence: float,
        model_used: str,
        features_used: Dict[str, Any],
        is_fraud: bool,
        action_taken: str
    ):
        """
        Registra detección de fraude con metadata completa para auditoría.
        
        Args:
            transaction_id: Identificador único de la transacción
            user_id: Identificador del usuario
            fraud_score: Puntuación de fraude calculada
            confidence: Confianza de la predicción
            model_used: Modelo utilizado para la detección
            features_used: Características utilizadas en la predicción
            is_fraud: Si se clasificó como fraude
            action_taken: Acción tomada (approve, reject, manual_review)
        """
        # Hash de features para privacidad
        features_hash = self._hash_features(features_used)
        
        self.logger.info(
            "fraud_detection",
            transaction_id=transaction_id,
            user_id=user_id,
            fraud_score=fraud_score,
            confidence=confidence,
            model_used=model_used,
            features_hash=features_hash,
            feature_count=len(features_used),
            is_fraud=is_fraud,
            action_taken=action_taken,
            audit_required=is_fraud,
            compliance_check=True
        )
    
    def log_model_prediction(
        self,
        model_name: str,
        model_version: str,
        input_features: Dict[str, Any],
        prediction: float,
        confidence: Optional[float] = None,
        prediction_time_ms: Optional[float] = None
    ):
        """
        Registra predicción de modelo ML con metadata para trazabilidad.
        
        Args:
            model_name: Nombre del modelo
            model_version: Versión del modelo
            input_features: Features utilizadas para la predicción
            prediction: Valor predicho
            confidence: Confianza de la predicción (si aplica)
            prediction_time_ms: Tiempo de inferencia en milisegundos
        """
        # Hash de features para privacidad
        features_hash = self._hash_features(input_features)
        
        self.logger.info(
            "model_prediction",
            model_name=model_name,
            model_version=model_version,
            features_hash=features_hash,
            feature_count=len(input_features),
            prediction=prediction,
            confidence=confidence,
            prediction_time_ms=prediction_time_ms,
            monitoring_required=True
        )
    
    def log_anomaly_detection(
        self,
        transaction_id: str,
        anomaly_score: float,
        anomaly_type: str,
        baseline_stats: Dict[str, float],
        detection_method: str,
        threshold: float
    ):
        """
        Registra detección de anomalías con contexto estadístico.
        
        Args:
            transaction_id: Identificador de la transacción
            anomaly_score: Puntuación de anomalía
            anomaly_type: Tipo de anomalía detectada
            baseline_stats: Estadísticas de referencia
            detection_method: Método de detección utilizado
            threshold: Umbral utilizado para detección
        """
        anomaly_status = "detected" if anomaly_score > threshold else "normal"
        
        self.logger.info(
            "anomaly_detection",
            transaction_id=transaction_id,
            anomaly_score=anomaly_score,
            anomaly_type=anomaly_type,
            baseline_stats=baseline_stats,
            detection_method=detection_method,
            threshold=threshold,
            anomaly_status=anomaly_status,
            alert_required=anomaly_status == "detected"
        )
    
    def log_data_quality_issue(
        self,
        data_source: str,
        issue_type: str,
        affected_records: int,
        severity: str,
        description: str
    ):
        """
        Registra problemas de calidad de datos.
        
        Args:
            data_source: Fuente de datos afectada
            issue_type: Tipo de problema (missing_values, outliers, etc.)
            affected_records: Número de registros afectados
            severity: Severidad (low, medium, high, critical)
            description: Descripción detallada del problema
        """
        self.logger.warning(
            "data_quality_issue",
            data_source=data_source,
            issue_type=issue_type,
            affected_records=affected_records,
            severity=severity,
            description=description,
            investigation_required=severity in ["high", "critical"]
        )
    
    def _hash_features(self, features: Dict[str, Any]) -> str:
        """Genera hash de features para privacidad y trazabilidad"""
        import hashlib
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.sha256(features_str.encode()).hexdigest()[:16]

# Instancia global del logger
fraud_detection_logger = FraudDetectionLogger("cerverus_fraud_detection", "production")
```

## Trazabilidad y Auditoría en Sistemas de Detección de Fraude

### Sistema de Trazabilidad para Operaciones Críticas
```python
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
import uuid
import json

@dataclass
class FraudDetectionAuditTrail:
    """
    Registro de auditoría para operaciones de detección de fraude.
    Cumple con requisitos regulatorios de trazabilidad.
    """
    operation_id: str
    operation_type: str
    transaction_id: str
    user_id: str
    timestamp: datetime
    input_data: Dict[str, Any]
    detection_result: Dict[str, Any]
    execution_time_ms: float
    model_used: str
    model_version: str
    fraud_score: float
    confidence: float
    is_fraud: bool
    action_taken: str
    reviewer_id: Optional[str] = None
    review_notes: Optional[str] = None
    compliance_flags: Optional[Dict[str, bool]] = None
    metadata: Optional[Dict[str, Any]] = None

class FraudAuditTrailManager:
    """
    Gestor de trails de auditoría para sistemas de detección de fraude.
    Proporciona trazabilidad completa de operaciones críticas.
    """
    
    def __init__(self, storage_backend: str = "database"):
        self.storage_backend = storage_backend
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Inicializa backend de almacenamiento para auditoría"""
        if self.storage_backend == "database":
            # Conexión a base de datos para auditoría
            pass
        elif self.storage_backend == "file":
            # Sistema de archivos con rotación
            pass
    
    def create_fraud_detection_audit_trail(
        self,
        transaction_id: str,
        user_id: str,
        input_data: Dict[str, Any],
        detection_result: Dict[str, Any],
        execution_time_ms: float,
        model_used: str,
        model_version: str,
        fraud_score: float,
        confidence: float,
        is_fraud: bool,
        action_taken: str,
        reviewer_id: Optional[str] = None,
        review_notes: Optional[str] = None,
        compliance_flags: Optional[Dict[str, bool]] = None
    ) -> str:
        """
        Crea un nuevo registro de auditoría para detección de fraude.
        
        Args:
            transaction_id: Identificador único de la transacción
            user_id: Identificador del usuario
            input_data: Datos de entrada de la transacción
            detection_result: Resultados de la detección
            execution_time_ms: Tiempo de ejecución
            model_used: Modelo utilizado
            model_version: Versión del modelo
            fraud_score: Puntuación de fraude
            confidence: Confianza de la predicción
            is_fraud: Si se clasificó como fraude
            action_taken: Acción tomada
            reviewer_id: ID del revisor (si aplica)
            review_notes: Notas de revisión (si aplica)
            compliance_flags: Flags de cumplimiento regulatorio
        
        Returns:
            ID único del registro de auditoría
        """
        operation_id = str(uuid.uuid4())
        
        audit_trail = FraudDetectionAuditTrail(
            operation_id=operation_id,
            operation_type="fraud_detection",
            transaction_id=transaction_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            input_data=self._sanitize_data(input_data),
            detection_result=self._sanitize_data(detection_result),
            execution_time_ms=execution_time_ms,
            model_used=model_used,
            model_version=model_version,
            fraud_score=fraud_score,
            confidence=confidence,
            is_fraud=is_fraud,
            action_taken=action_taken,
            reviewer_id=reviewer_id,
            review_notes=review_notes,
            compliance_flags=compliance_flags or {},
            metadata={
                "environment": "production",
                "version": "1.0.0",
                "compliance_check": True
            }
        )
        
        self._store_audit_trail(audit_trail)
        
        # Log para monitoreo en tiempo real
        fraud_detection_logger.logger.info(
            "fraud_audit_trail_created",
            operation_id=operation_id,
            transaction_id=transaction_id,
            user_id=user_id,
            is_fraud=is_fraud,
            fraud_score=fraud_score,
            action_taken=action_taken
        )
        
        return operation_id
    
    def get_fraud_detection_audit_trail(self, operation_id: str) -> Optional[FraudDetectionAuditTrail]:
        """
        Recupera un registro de auditoría por ID.
        
        Args:
            operation_id: ID del registro de auditoría
        
        Returns:
            Registro de auditoría si existe, None en caso contrario
        """
        # Implementar recuperación según backend
        pass
    
    def get_audit_trails_by_transaction(
        self,
        transaction_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[FraudDetectionAuditTrail]:
        """
        Recupera registros de auditoría por transacción.
        
        Args:
            transaction_id: ID de la transacción
            start_date: Fecha de inicio del filtro
            end_date: Fecha de fin del filtro
        
        Returns:
            Lista de registros de auditoría que cumplen los filtros
        """
        # Implementar consulta con filtros
        pass
    
    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza datos para auditoría, removiendo información sensible.
        
        Args:
            data: Datos originales
        
        Returns:
            Datos sanitizados para almacenamiento seguro
        """
        sanitized = data.copy()
        
        # Lista de campos sensibles a sanitizar
        sensitive_fields = [
            "card_number", "cvv", "expiry_date", "pin", "password",
            "ssn", "account_number", "routing_number"
        ]
        
        for field in sensitive_fields:
            if field in sanitized:
                sanitized[field] = "***REDACTED***"
        
        return sanitized
    
    def _store_audit_trail(self, audit_trail: FraudDetectionAuditTrail):
        """
        Almacena registro de auditoría en el backend configurado.
        
        Args:
            audit_trail: Registro de auditoría a almacenar
        """
        if self.storage_backend == "database":
            self._store_in_database(audit_trail)
        elif self.storage_backend == "file":
            self._store_in_file(audit_trail)
    
    def _store_in_database(self, audit_trail: FraudDetectionAuditTrail):
        """Almacena registro en base de datos"""
        # Implementar almacenamiento en base de datos
        pass
    
    def _store_in_file(self, audit_trail: FraudDetectionAuditTrail):
        """Almacena registro en sistema de archivos"""
        # Implementar almacenamiento en archivos con rotación
        pass

# Instancia global para auditoría
fraud_audit_manager = FraudAuditTrailManager(storage_backend="database")

# Decorador para auditoría automática
def audit_fraud_detection(operation_type: str):
    """
    Decorador para añadir auditoría automática a funciones de detección de fraude.
    
    Args:
        operation_type: Tipo de operación para registro de auditoría
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            transaction_id = kwargs.get("transaction_id") or "unknown"
            user_id = kwargs.get("user_id") or "unknown"
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds() * 1000
                
                # Crear registro de auditoría
                fraud_audit_manager.create_fraud_detection_audit_trail(
                    transaction_id=transaction_id,
                    user_id=user_id,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    detection_result={"result": str(result)},
                    execution_time_ms=execution_time,
                    model_used=result.get("model_used", "unknown"),
                    model_version=result.get("model_version", "unknown"),
                    fraud_score=result.get("fraud_score", 0.0),
                    confidence=result.get("confidence", 0.0),
                    is_fraud=result.get("is_fraud", False),
                    action_taken=result.get("action_taken", "none")
                )
                
                return result
                
            except Exception as e:
                end_time = datetime.utcnow()
                execution_time = (end_time - start_time).total_seconds() * 1000
                
                # Crear registro de auditoría para error
                fraud_audit_manager.create_fraud_detection_audit_trail(
                    transaction_id=transaction_id,
                    user_id=user_id,
                    input_data={"args": str(args), "kwargs": str(kwargs)},
                    detection_result={},
                    execution_time_ms=execution_time,
                    model_used="unknown",
                    model_version="unknown",
                    fraud_score=0.0,
                    confidence=0.0,
                    is_fraud=False,
                    action_taken="error"
                )
                
                raise
        
        return wrapper
    return decorator

# Uso del decorador
@audit_fraud_detection("fraud_detection")
def detect_fraud_in_transaction(
    transaction_data: Dict[str, Any],
    user_id: str,
    model_name: str = "isolation_forest"
) -> Dict[str, Any]:
    """
    Detecta fraude en una transacción con auditoría automática.
    
    Args:
        transaction_data: Datos de la transacción
        user_id: ID del usuario
        model_name: Nombre del modelo a utilizar
    
    Returns:
        Resultados de la detección de fraude
    """
    # Lógica de detección de fraude
    fraud_result = {
        "transaction_id": transaction_data.get("transaction_id"),
        "fraud_score": 0.85,
        "confidence": 0.92,
        "is_fraud": True,
        "action_taken": "manual_review",
        "model_used": model_name,
        "model_version": "1.2.0"
    }
    
    return fraud_result
```

## Requisitos de Pruebas para Sistemas de Detección de Fraude

### Estrategia de Testing Integral

#### Testing Unitario para Componentes de Detección de Fraude
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, Any

class TestFraudDetectionCalculations:
    """Tests unitarios para cálculos de detección de fraude"""
    
    def test_fraud_probability_calculation(self):
        """Test de cálculo de probabilidad de fraude"""
        # Datos de prueba
        transaction_features = {
            "amount": 1000.0,
            "time_since_last_transaction": 3600,
            "merchant_category": "electronics",
            "user_history_score": 0.8
        }
        
        # Mock del modelo
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.15, 0.85]])
        
        # Ejecutar función
        is_fraud, fraud_probability = calculate_fraud_probability(
            transaction_features=transaction_features,
            fraud_detection_model=mock_model,
            probability_threshold=0.8
        )
        
        # Validaciones
        assert isinstance(is_fraud, bool)
        assert isinstance(fraud_probability, float)
        assert 0 <= fraud_probability <= 1
        assert is_fraud == True  # 0.85 > 0.8
        assert fraud_probability == 0.85
    
    def test_anomaly_detection(self):
        """Test de detección de anomalías"""
        # Datos de prueba
        transaction_history = [
            {"amount": 50.0, "timestamp": "2023-01-01T10:00:00Z"},
            {"amount": 75.0, "timestamp": "2023-01-01T11:00:00Z"},
            {"amount": 60.0, "timestamp": "2023-01-01T12:00:00Z"}
        ]
        
        current_transaction = {
            "amount": 500.0,  # Monto anómalamente alto
            "timestamp": "2023-01-01T13:00:00Z"
        }
        
        # Ejecutar función
        anomaly_result = detect_anomalous_transaction_pattern(
            transaction_history=transaction_history,
            current_transaction=current_transaction,
            anomaly_threshold=2.0
        )
        
        # Validaciones
        assert "is_anomalous" in anomaly_result
        assert "z_score" in anomaly_result
        assert "mean_historical_amount" in anomaly_result
        assert "current_amount" in anomaly_result
        assert anomaly_result["is_anomalous"] == True
        assert anomaly_result["z_score"] > 2.0
    
    def test_invalid_transaction_data(self):
        """Test de validación de datos de transacción inválidos"""
        # Datos inválidos (campos faltantes)
        invalid_transaction = {
            "amount": 1000.0,
            # Faltan campos requeridos como timestamp, user_id, etc.
        }
        
        # Mock del modelo
        mock_model = Mock()
        
        # Validar que lanza excepción
        with pytest.raises(DataQualityError):
            detect_fraud_with_ensemble(
                transaction_data=invalid_transaction,
                models={"isolation_forest": mock_model},
                config={"fraud_threshold": 0.8}
            )

class TestFraudDetectionAPI:
    """Tests para endpoints de API de detección de fraude"""
    
    @pytest.fixture
    def client(self):
        """Fixture para cliente de test"""
        from fastapi.testclient import TestClient
        from api.app.main import app
        
        return TestClient(app)
    
    def test_fraud_detection_endpoint_success(self, client):
        """Test de endpoint de detección de fraude exitoso"""
        # Mock de servicios
        with patch('api.app.endpoints.fraud.model_registry.get_fraud_models') as mock_get_models, \
             patch('api.app.endpoints.fraud.fraud_detection_service.detect_fraud_with_ensemble') as mock_detect_fraud, \
             patch('api.app.endpoints.fraud.config_service.get_fraud_detection_config') as mock_get_config:
            
            # Configurar mocks
            mock_get_models.return_value = {
                "isolation_forest": Mock(),
                "autoencoder": Mock()
            }
            
            mock_detect_fraud.return_value = {
                "is_fraud": False,
                "fraud_probability": 0.25,
                "confidence": 0.88,
                "individual_predictions": {
                    "isolation_forest": 0.2,
                    "autoencoder": 0.3
                },
                "ensemble_method": "weighted_average"
            }
            
            mock_get_config.return_value = {
                "fraud_threshold": 0.8,
                "ensemble_method": "weighted_average"
            }
            
            # Datos de prueba
            transaction_data = {
                "transaction_id": "test_tx_123",
                "amount": 100.0,
                "timestamp": "2023-01-01T10:00:00Z",
                "user_id": "user_123",
                "merchant_id": "merchant_456"
            }
            
            # Ejecutar request
            response = client.post("/api/v1/fraud/detect", json=transaction_data)
            
            # Validaciones
            assert response.status_code == 200
            data = response.json()
            assert "fraud_detection_result" in data
            assert data["transaction_id"] == "test_tx_123"
            assert data["fraud_detection_result"]["is_fraud"] == False
    
    def test_fraud_detection_endpoint_invalid_data(self, client):
        """Test de endpoint con datos inválidos"""
        # Datos inválidos (campos faltantes)
        invalid_transaction = {
            "amount": 100.0,
            # Faltan campos requeridos
        }
        
        response = client.post("/api/v1/fraud/detect", json=invalid_transaction)
        
        assert response.status_code == 422  # Validation error

class TestFraudDetectionModels:
    """Tests para modelos de detección de fraude"""
    
    def test_isolation_forest_model(self):
        """Test de modelo Isolation Forest"""
        # Datos de prueba
        transaction_features = {
            "amount": 1000.0,
            "time_since_last_transaction": 3600,
            "merchant_category_risk": 0.8,
            "user_history_score": 0.2,
            "transaction_frequency": 5
        }
        
        # Mock del modelo
        with patch('src.cerverus.algorithms.isolation_forest.IsolationForestDetector.predict') as mock_predict:
            mock_predict.return_value = {
                "anomaly_score": -0.15,
                "is_anomaly": True,
                "confidence": 0.91
            }
            
            # Ejecutar predicción
            detector = IsolationForestDetector()
            result = detector.predict(transaction_features)
            
            # Validaciones
            assert "anomaly_score" in result
            assert "is_anomaly" in result
            assert "confidence" in result
            assert result["anomaly_score"] < 0  # Anomalía en Isolation Forest
            assert isinstance(result["is_anomaly"], bool)
            assert 0 <= result["confidence"] <= 1
    
    def test_autoencoder_model(self):
        """Test de modelo Autoencoder"""
        # Datos de prueba
        transaction_features = {
            "amount": 5000.0,
            "time_since_last_transaction": 300,
            "merchant_category_risk": 0.9,
            "user_history_score": 0.1,
            "transaction_frequency": 15
        }
        
        # Mock del modelo
        with patch('src.cerverus.algorithms.autoencoder.AutoencoderDetector.predict') as mock_predict:
            mock_predict.return_value = {
                "reconstruction_error": 2.5,
                "threshold": 1.0,
                "is_anomaly": True,
                "anomaly_score": 0.85
            }
            
            # Ejecutar predicción
            detector = AutoencoderDetector()
            result = detector.predict(transaction_features)
            
            # Validaciones
            assert "reconstruction_error" in result
            assert "threshold" in result
            assert "is_anomaly" in result
            assert "anomaly_score" in result
            assert result["reconstruction_error"] > result["threshold"]
            assert result["is_anomaly"] == True
            assert 0 <= result["anomaly_score"] <= 1

class TestDataQuality:
    """Tests para validación de calidad de datos"""
    
    def test_transaction_data_validation(self):
        """Test de validación de datos de transacción"""
        # Datos válidos
        valid_data = {
            "transaction_id": "tx_123",
            "amount": 100.0,
            "timestamp": "2023-01-01T10:00:00Z",
            "user_id": "user_123",
            "merchant_id": "merchant_456"
        }
        
        assert validate_transaction_data(valid_data) is True
        
        # Datos inválidos
        invalid_data = {
            "transaction_id": "tx_123",
            "amount": -100.0,  # Monto negativo inválido
            "timestamp": "2023-01-01T10:00:00Z",
            "user_id": "user_123",
            "merchant_id": "merchant_456"
        }
        
        assert validate_transaction_data(invalid_data) is False
    
    def test_missing_data_handling(self):
        """Test de manejo de datos faltantes"""
        # Datos con valores faltantes
        incomplete_data = {
            "transaction_id": "tx_123",
            "amount": None,  # Valor faltante
            "timestamp": "2023-01-01T10:00:00Z",
            "user_id": "user_123",
            "merchant_id": "merchant_456"
        }
        
        # Validar que se maneja correctamente
        result = handle_missing_transaction_data(incomplete_data)
        assert result["amount_imputed"] is True
        assert result["imputation_method"] == "median"
```

#### Testing de Integración para Sistemas de Detección de Fraude
```python
import pytest
import requests
import json
from datetime import datetime, timedelta

class TestFraudDetectionSystemIntegration:
    """Tests de integración para el sistema de detección de fraude completo"""
    
    @pytest.fixture(scope="class")
    def test_environment(self):
        """Fixture para configurar entorno de test"""
        # Configurar base de datos de test
        # Iniciar servicios mock
        # Cargar datos de prueba
        yield
        # Limpiar entorno
    
    def test_complete_fraud_detection_flow(self, test_environment):
        """Test de flujo completo de detección de fraude"""
        # 1. Ingestar datos de transacción
        transaction_data = {
            "transaction_id": "test_tx_123",
            "amount": 1500.0,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user_123",
            "merchant_id": "merchant_456",
            "merchant_category": "electronics",
            "location": "New York"
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/transactions/ingest",
            json=transaction_data
        )
        assert response.status_code == 201
        
        # 2. Ejecutar detección de fraude
        response = requests.post(
            "http://localhost:8000/api/v1/fraud/detect",
            json=transaction_data
        )
        assert response.status_code == 200
        fraud_result = response.json()["fraud_detection_result"]
        
        # 3. Verificar que se generó alerta si es fraude
        if fraud_result["is_fraud"]:
            response = requests.get(
                f"http://localhost:8000/api/v1/alerts/transaction/{transaction_data['transaction_id']}"
            )
            assert response.status_code == 200
            alerts = response.json()["alerts"]
            assert len(alerts) > 0
            assert alerts[0]["severity"] in ["medium", "high"]
        
        # 4. Verificar registro de auditoría
        response = requests.get(
            f"http://localhost:8000/api/v1/audit/transaction/{transaction_data['transaction_id']}"
        )
        assert response.status_code == 200
        audit_records = response.json()["audit_records"]
        assert len(audit_records) > 0
        assert audit_records[0]["operation_type"] == "fraud_detection"
    
    def test_model_ensemble_integration(self, test_environment):
        """Test de integración de ensemble de modelos"""
        # 1. Configurar ensemble de modelos
        ensemble_config = {
            "models": ["isolation_forest", "autoencoder", "lof"],
            "weights": [0.4, 0.4, 0.2],
            "combination_method": "weighted_average",
            "fraud_threshold": 0.75
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/models/ensemble/config",
            json=ensemble_config
        )
        assert response.status_code == 200
        
        # 2. Probar con transacción de prueba
        test_transaction = {
            "transaction_id": "test_tx_ensemble",
            "amount": 2500.0,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user_456",
            "merchant_id": "merchant_789",
            "merchant_category": "jewelry",
            "location": "Los Angeles"
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/fraud/detect",
            json=test_transaction
        )
        assert response.status_code == 200
        
        result = response.json()["fraud_detection_result"]
        assert "individual_predictions" in result
        assert "ensemble_method" in result
        assert len(result["individual_predictions"]) == 3
        assert result["ensemble_method"] == "weighted_average"
    
    def test_real_time_monitoring_integration(self, test_environment):
        """Test de integración de monitoreo en tiempo real"""
        # 1. Configurar umbrales de monitoreo
        monitoring_config = {
            "fraud_score_threshold": 0.8,
            "anomaly_detection_threshold": 2.0,
            "alert_cooldown_minutes": 5,
            "notification_channels": ["email", "slack"]
        }
        
        response = requests.post(
            "http://localhost:8000/api/v1/monitoring/config",
            json=monitoring_config
        )
        assert response.status_code == 200
        
        # 2. Simular transacciones de alto riesgo
        high_risk_transactions = [
            {
                "transaction_id": f"high_risk_tx_{i}",
                "amount": 5000.0 + i * 1000,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "user_risk",
                "merchant_id": "merchant_risk",
                "merchant_category": "electronics",
                "location": "Unknown"
            }
            for i in range(3)
        ]
        
        for tx in high_risk_transactions:
            response = requests.post(
                "http://localhost:8000/api/v1/fraud/detect",
                json=tx
            )
            assert response.status_code == 200
        
        # 3. Verificar métricas de monitoreo
        response = requests.get(
            "http://localhost:8000/api/v1/monitoring/metrics"
        )
        assert response.status_code == 200
        metrics = response.json()
        
        assert "fraud_detection_rate" in metrics
        assert "average_fraud_score" in metrics
        assert "alert_count" in metrics
        assert metrics["alert_count"] > 0

class TestPerformanceBenchmarks:
    """Tests de rendimiento para componentes críticos"""
    
    def test_fraud_detection_performance(self):
        """Test de rendimiento para detección de fraude"""
        import time
        
        # Generar datos de prueba
        test_transactions = [
            {
                "transaction_id": f"perf_tx_{i}",
                "amount": 100.0 + i * 10,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": f"user_{i % 100}",
                "merchant_id": f"merchant_{i % 50}",
                "merchant_category": "retail",
                "location": "Test Location"
            }
            for i in range(100)
        ]
        
        # Medir tiempo de ejecución
        start_time = time.time()
        
        for tx in test_transactions:
            response = requests.post(
                "http://localhost:8000/api/v1/fraud/detect",
                json=tx
            )
            assert response.status_code == 200
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_transaction = total_time / len(test_transactions)
        
        # Validar rendimiento (debe ser < 100ms por transacción)
        assert avg_time_per_transaction < 0.1, f"Fraud detection too slow: {avg_time_per_transaction:.3f}s"
    
    def test_model_prediction_performance(self):
        """Test de rendimiento de predicción de modelos"""
        import time
        
        # Datos de prueba para predicción
        test_features = {
            "amount": 1000.0,
            "time_since_last_transaction": 3600,
            "merchant_category_risk": 0.7,
            "user_history_score": 0.3,
            "transaction_frequency": 8,
            "location_risk": 0.5,
            "device_trust_score": 0.8,
            "ip_risk_score": 0.2
        }
        
        # Medir tiempo de predicción para cada modelo
        models = ["isolation_forest", "autoencoder", "lof"]
        
        for model_name in models:
            start_time = time.time()
            
            response = requests.post(
                f"http://localhost:8000/api/v1/models/{model_name}/predict",
                json=test_features
            )
            
            end_time = time.time()
            prediction_time = end_time - start_time
            
            # Validar tiempo de respuesta (debe ser < 50ms)
            assert prediction_time < 0.05, f"Model {model_name} prediction too slow: {prediction_time:.3f}s"
            assert response.status_code == 200
```

#### Testing de Carga y Estrés
```python
import pytest
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time

class TestFraudDetectionLoadStress:
    """Tests de carga y estrés para el sistema de detección de fraude"""
    
    @pytest.mark.slow
    def test_concurrent_fraud_detection(self):
        """Test de detección de fraude concurrente"""
        num_concurrent_transactions = 50
        
        def detect_fraud(transaction_id):
            """Función para detectar fraude en una transacción individual"""
            transaction_data = {
                "transaction_id": f"concurrent_tx_{transaction_id}",
                "amount": 100.0 + transaction_id * 10,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": f"user_{transaction_id % 20}",
                "merchant_id": f"merchant_{transaction_id % 10}",
                "merchant_category": "retail",
                "location": "Test Location"
            }
            
            response = requests.post(
                "http://localhost:8000/api/v1/fraud/detect",
                json=transaction_data
            )
            
            return response.status_code == 200
        
        # Ejecutar detecciones concurrentemente
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(detect_fraud, i) for i in range(num_concurrent_transactions)]
            results = [future.result() for future in futures]
        
        # Validar que todas las detecciones fueron exitosas
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.95, f"Fraud detection success rate too low: {success_rate:.2%}"
    
    @pytest.mark.slow
    def test_fraud_detection_under_load(self):
        """Test de detección de fraude bajo carga"""
        num_requests = 100
        
        async def make_fraud_detection_request(session, request_id):
            """Función asíncrona para request de detección de fraude"""
            transaction_data = {
                "transaction_id": f"load_tx_{request_id}",
                "amount": 500.0 + request_id * 5,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": f"user_{request_id % 30}",
                "merchant_id": f"merchant_{request_id % 15}",
                "merchant_category": "electronics",
                "location": "Load Test Location"
            }
            
            try:
                async with session.post(
                    "http://localhost:8000/api/v1/fraud/detect",
                    json=transaction_data
                ) as response:
                    return response.status == 200
            except Exception:
                return False
        
        async def run_load_test():
            """Ejecutar test de carga asíncrono"""
            connector = aiohttp.TCPConnector(limit=50)
            timeout = aiohttp.ClientTimeout(total=30)
            
            async with aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            ) as session:
                
                tasks = [make_fraud_detection_request(session, i) for i in range(num_requests)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                return results
        
        # Ejecutar test de carga
        start_time = time.time()
        results = asyncio.run(run_load_test())
        end_time = time.time()
        
        # Calcular métricas
        total_time = end_time - start_time
        successful_requests = sum(1 for r in results if r is True)
        success_rate = successful_requests / num_requests
        requests_per_second = num_requests / total_time
        
        # Validaciones
        assert success_rate >= 0.90, f"Success rate too low: {success_rate:.2%}"
        assert requests_per_second >= 10, f"Throughput too low: {requests_per_second:.1f} req/s"
    
    @pytest.mark.slow
    def test_memory_usage_under_load(self):
        """Test de uso de memoria bajo carga"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Ejecutar carga intensiva de detección de fraude
        for i in range(1000):
            transaction_data = {
                "transaction_id": f"mem_tx_{i}",
                "amount": 1000.0,
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": f"user_{i % 100}",
                "merchant_id": f"merchant_{i % 50}",
                "merchant_category": "electronics",
                "location": "Memory Test Location"
            }
            
            # Simular procesamiento de detección de fraude
            _ = prepare_feature_vector(transaction_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Validar que el aumento de memoria sea razonable (< 100MB)
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f}MB"
```

## Configuración de CI/CD para Sistemas de Detección de Fraude

### Pipeline de CI/CD con Validaciones de Detección de Fraude
```yaml
# .github/workflows/fraud-detection-ci.yml
name: Fraud Detection System CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  NODE_VERSION: '18'

jobs:
  code-quality:
    name: Code Quality & Security
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-dev.txt
    
    - name: Run linting (Black, Flake8, MyPy)
      run: |
        black --check src/
        flake8 src/
        mypy src/
    
    - name: Run security scanning (Bandit)
      run: |
        bandit -r src/ -f json -o bandit-report.json
    
    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: security-scan-results
        path: bandit-report.json

  unit-tests:
    name: Unit Tests
    runs-on: ubuntu-latest
    needs: code-quality
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src/cerverus --cov-report=xml --cov-report=html
    
    - name: Upload coverage reports
      uses: actions/upload-artifact@v3
      with:
        name: coverage-reports
        path: |
          coverage.xml
          htmlcov/

  integration-tests:
    name: Integration Tests
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Start test environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
    
    - name: Stop test environment
      if: always()
      run: |
        docker-compose -f docker-compose.test.yml down

  model-validation:
    name: Model Validation
    runs-on: ubuntu-latest
    needs: unit-tests
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Validate model performance
      run: |
        python scripts/validate_model_performance.py
    
    - name: Check model drift
      run: |
        python scripts/check_model_drift.py
    
    - name: Upload validation results
      uses: actions/upload-artifact@v3
      with:
        name: model-validation-results
        path: |
          model_performance_report.json
          model_drift_report.json

  performance-tests:
    name: Performance Tests
    runs-on: ubuntu-latest
    needs: integration-tests
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt
    
    - name: Start performance test environment
      run: |
        docker-compose -f docker-compose.performance.yml up -d
        sleep 30
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-only
    
    - name: Generate performance report
      run: |
        python scripts/generate_performance_report.py
    
    - name: Upload performance results
      uses: actions/upload-artifact@v3
      with:
        name: performance-results
        path: |
          performance_report.json
          benchmark_results.json

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: code-quality
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Run Snyk security scan
      uses: snyk/actions/python@master
      env:
        SNYK_TOKEN: ${{ secrets.SNYK_TOKEN }}
      with:
        args: --severity-threshold=high
    
    - name: Run OWASP dependency check
      run: |
        dependency-check --project "Fraud Detection System" --scan . --format JSON
    
    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: |
          snyk-report.json
          dependency-check-report.json

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, model-validation, performance-tests]
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Deploy to staging
      run: |
        python scripts/deploy_to_staging.py
    
    - name: Run smoke tests
      run: |
        python scripts/run_smoke_tests.py

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests, model-validation, performance-tests, security-scan]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Deploy to production
      run: |
        python scripts/deploy_to_production.py
    
    - name: Run production validation
      run: |
        python scripts/validate_production_deployment.py
