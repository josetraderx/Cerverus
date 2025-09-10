---
title: "Guía de Debugging para Cerverus"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# Guía de Debugging para Cerverus

## Checklist de Calidad para Debugging
- [ ] Logging estructurado y correlación configurados
- [ ] Debugging por etapa (1-7) implementado
- [ ] Profiling de performance activado
- [ ] Health check distribuido funcionando
- [ ] Circuit breaker debugging habilitado
- [ ] Tracing distribuido con OpenTelemetry configurado

## Logging Estructurado y Correlación

### Configuración de Logging por Componente
```python
import structlog
from pythonjsonlogger import jsonlogger

class CerverusLogger:
    """Logger estructurado para debugging de Cerverus"""
    
    @staticmethod
    def setup_logging():
        """Configuración central de logging"""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                CerverusLogger._add_trace_context,
                CerverusLogger._add_cerverus_context,
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    @staticmethod
    def _add_trace_context(logger, method_name, event_dict):
        """Añade contexto de tracing para correlación"""
        # Obtener trace ID de OpenTelemetry
        current_span = trace.get_current_span()
        if current_span.is_recording():
            span_context = current_span.get_span_context()
            event_dict["trace_id"] = f"{span_context.trace_id:032x}"
            event_dict["span_id"] = f"{span_context.span_id:016x}"
        return event_dict
    
    @staticmethod
    def _add_cerverus_context(logger, method_name, event_dict):
        """Añade contexto específico de Cerverus"""
        event_dict["service"] = "cerverus"
        event_dict["environment"] = os.getenv("CERVERUS_ENV", "development")
        event_dict["version"] = os.getenv("CERVERUS_VERSION", "unknown")
        return event_dict

# Loggers específicos por componente
fraud_detection_logger = structlog.get_logger("fraud_detection")
data_pipeline_logger = structlog.get_logger("data_pipeline")
ml_model_logger = structlog.get_logger("ml_model")
compliance_logger = structlog.get_logger("compliance")
```

### Patrones de Debugging por Etapa

#### Etapa 1-2: Debugging de Pipeline de Datos
```python
def debug_data_extraction(source: str, symbol: str):
    """Template de debugging para extracción de datos"""
    
    logger = data_pipeline_logger.bind(
        source=source,
        symbol=symbol,
        operation="data_extraction"
    )
    
    try:
        logger.info("Starting data extraction")
        
        # 1. Verificar conectividad
        if not check_source_connectivity(source):
            logger.error("Source connectivity failed")
            return
        
        # 2. Verificar rate limiting
        rate_limit_status = check_rate_limit(source)
        logger.info("Rate limit status", **rate_limit_status)
        
        # 3. Extraer datos con timing
        start_time = time.time()
        data = extract_data(source, symbol)
        extraction_time = time.time() - start_time
        
        logger.info(
            "Data extraction completed",
            records_extracted=len(data),
            extraction_time_seconds=extraction_time,
            data_size_mb=sys.getsizeof(data) / 1024 / 1024
        )
        
        # 4. Validar calidad de datos
        quality_check = validate_data_quality(data)
        logger.info("Data quality check", **quality_check)
        
        if quality_check['has_issues']:
            logger.warning(
                "Data quality issues detected",
                issues=quality_check['issues']
            )
            
    except Exception as e:
        logger.error(
            "Data extraction failed",
            error=str(e),
            error_type=type(e).__name__,
            traceback=traceback.format_exc()
        )
```

#### Etapa 5: Debugging de Modelos ML
```python
def debug_fraud_detection_inference(transaction_id: str, model_name: str):
    """Template de debugging para inferencia ML"""
    
    logger = ml_model_logger.bind(
        transaction_id=transaction_id,
        model_name=model_name,
        operation="fraud_inference"
    )
    
    try:
        # 1. Verificar disponibilidad del modelo
        model = load_model(model_name)
        logger.info(
            "Model loaded",
            model_version=model.version,
            model_size_mb=model.size_mb,
            load_time_ms=model.load_time_ms
        )
        
        # 2. Extraer y validar features
        features = extract_features(transaction_id)
        feature_validation = validate_features(features, model.required_features)
        
        logger.info(
            "Features extracted",
            feature_count=len(features),
            missing_features=feature_validation.get('missing', []),
            invalid_features=feature_validation.get('invalid', [])
        )
        
        if feature_validation['is_valid']:
            # 3. Ejecutar inferencia con timing
            start_time = time.time()
            prediction = model.predict(features)
            inference_time = time.time() - start_time
            
            logger.info(
                "Inference completed",
                fraud_score=prediction.fraud_score,
                confidence=prediction.confidence,
                inference_time_ms=inference_time * 1000,
                algorithm_scores=prediction.algorithm_breakdown
            )
            
            # 4. Validar resultado
            if prediction.fraud_score > 0.8:
                logger.warning(
                    "High fraud score detected",
                    fraud_score=prediction.fraud_score,
                    contributing_features=prediction.top_features
                )
        else:
            logger.error(
                "Feature validation failed",
                validation_errors=feature_validation['errors']
            )
            
    except Exception as e:
        logger.error(
            "Fraud detection inference failed",
            error=str(e),
            error_type=type(e).__name__,
            model_available=model is not None,
            traceback=traceback.format_exc()
        )
```

## Debugging de Performance y Métricas

### Profiling de Componentes Críticos
```python
import cProfile
import pstats
from functools import wraps

def profile_performance(func):
    """Decorator para profiling de performance"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            profiler.disable()
            
            # Analizar stats
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            
            # Log top 10 funciones más lentas
            stats_output = io.StringIO()
            stats.print_stats(10)
            
            performance_logger.info(
                "Performance profile",
                function=func.__name__,
                top_functions=stats_output.getvalue()
            )
    
    return wrapper

@profile_performance
def detect_fraud_with_profiling(transaction_data):
    """Función de detección con profiling automático"""
    return fraud_detector.detect(transaction_data)
```

### Debugging de Latencia Distribuida
```python
class LatencyTracker:
    """Tracker de latencia para debugging distribuido"""
    
    def __init__(self):
        self.spans = {}
        
    def start_span(self, operation: str, context: dict = None):
        """Inicia span para tracking de latencia"""
        span_id = str(uuid.uuid4())
        self.spans[span_id] = {
            'operation': operation,
            'start_time': time.time(),
            'context': context or {}
        }
        return span_id
    
    def end_span(self, span_id: str, result: dict = None):
        """Termina span y log latencia"""
        if span_id not in self.spans:
            return
            
        span = self.spans[span_id]
        duration = time.time() - span['start_time']
        
        performance_logger.info(
            "Operation completed",
            operation=span['operation'],
            duration_ms=duration * 1000,
            context=span['context'],
            result=result
        )
        
        del self.spans[span_id]

# Uso para debugging de latencia
latency_tracker = LatencyTracker()

def debug_end_to_end_latency(transaction_id: str):
    """Debug latencia end-to-end de detección de fraude"""
    
    total_span = latency_tracker.start_span(
        "fraud_detection_e2e",
        {"transaction_id": transaction_id}
    )
    
    try:
        # Feature extraction
        feature_span = latency_tracker.start_span("feature_extraction")
        features = extract_features(transaction_id)
        latency_tracker.end_span(feature_span, {"feature_count": len(features)})
        
        # Model inference
        inference_span = latency_tracker.start_span("model_inference")
        prediction = run_model_inference(features)
        latency_tracker.end_span(inference_span, {"fraud_score": prediction.score})
        
        # Ensemble combination
        ensemble_span = latency_tracker.start_span("ensemble_combination")
        final_result = combine_ensemble_results(prediction)
        latency_tracker.end_span(ensemble_span, {"final_score": final_result.score})
        
        latency_tracker.end_span(total_span, {"is_fraud": final_result.is_fraud})
        
    except Exception as e:
        latency_tracker.end_span(total_span, {"error": str(e)})
        raise
```

## Debugging de Integración y Dependencias

### Health Check Distribuido
```python
class CerverusHealthChecker:
    """Health checker para debugging de dependencias"""
    
    def __init__(self):
        self.dependencies = [
            {'name': 'database', 'check': self._check_database},
            {'name': 'redis', 'check': self._check_redis},
            {'name': 'model_registry', 'check': self._check_model_registry},
            {'name': 'feature_store', 'check': self._check_feature_store},
            {'name': 'external_apis', 'check': self._check_external_apis}
        ]
    
    async def run_comprehensive_health_check(self) -> dict:
        """Ejecuta health check completo para debugging"""
        results = {
            'overall_status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'components': {}
        }
        
        for dependency in self.dependencies:
            try:
                start_time = time.time()
                status = await dependency['check']()
                check_duration = time.time() - start_time
                
                results['components'][dependency['name']] = {
                    'status': 'healthy' if status['healthy'] else 'unhealthy',
                    'response_time_ms': check_duration * 1000,
                    'details': status.get('details', {}),
                    'last_checked': datetime.utcnow().isoformat()
                }
                
                if not status['healthy']:
                    results['overall_status'] = 'degraded'
                    
            except Exception as e:
                results['components'][dependency['name']] = {
                    'status': 'error',
                    'error': str(e),
                    'last_checked': datetime.utcnow().isoformat()
                }
                results['overall_status'] = 'unhealthy'
        
        return results
    
    async def _check_model_registry(self) -> dict:
        """Check específico para model registry"""
        try:
            # Verificar conectividad
            models = await model_registry.list_models()
            
            # Verificar modelos críticos
            critical_models = ['isolation_forest', 'ensemble_stacking']
            missing_models = []
            
            for model_name in critical_models:
                try:
                    model = await model_registry.get_model(model_name, 'latest')
                    if not model:
                        missing_models.append(model_name)
                except Exception:
                    missing_models.append(model_name)
            
            return {
                'healthy': len(missing_models) == 0,
                'details': {
                    'total_models': len(models),
                    'missing_critical_models': missing_models,
                    'registry_responsive': True
                }
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'details': {
                    'error': str(e),
                    'registry_responsive': False
                }
            }
```

### Debugging de Circuit Breaker
```python
class CircuitBreakerDebugger:
    """Debugger para circuit breakers"""
    
    def __init__(self, circuit_breaker):
        self.circuit_breaker = circuit_breaker
        self.logger = structlog.get_logger("circuit_breaker")
    
    def debug_circuit_state(self, service_name: str):
        """Debug estado actual del circuit breaker"""
        state = self.circuit_breaker.current_state
        
        self.logger.info(
            "Circuit breaker state",
            service=service_name,
            state=state,
            failure_count=self.circuit_breaker.failure_count,
            success_count=self.circuit_breaker.success_count,
            last_failure_time=self.circuit_breaker.last_failure_time,
            next_attempt_time=self.circuit_breaker.next_attempt_time
        )
        
        # Recommendations basadas en estado
        if state == 'OPEN':
            self.logger.warning(
                "Circuit breaker is OPEN",
                service=service_name,
                recommendation="Service is failing. Check service health and logs."
            )
        elif state == 'HALF_OPEN':
            self.logger.info(
                "Circuit breaker is HALF_OPEN",
                service=service_name,
                recommendation="Testing service recovery. Monitor next few requests."
            )
```