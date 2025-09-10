---
title: "Reglas Globales de Estilo de Código"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# Reglas Globales de Estilo de Código

## Checklist de Calidad para Estilo de Código
- [ ] Convenciones de nomenclatura aplicadas (PEP 8, snake_case, etc.)
- [ ] Linting automático configurado (Black, Flake8, MyPy)
- [ ] Docstrings completos según estándar definido
- [ ] Pre-commit hooks configurados y funcionando
- [ ] Code review checklist aplicado

## Convenciones Técnicas Globales

### Estándares Python (PEP 8 Extendido)

#### Nomenclatura
```python
# Variables y funciones: snake_case
user_data = get_user_information()
fraud_detection_score = 0.95

# Clases: PascalCase
class FraudDetectionEngine:
    pass

# Constantes: UPPER_SNAKE_CASE
MAX_TRANSACTION_AMOUNT = 100000
DEFAULT_THRESHOLD = 0.8

# Archivos y módulos: snake_case
# fraud_detector.py
# data_processor.py
```

#### Estructura de Funciones
```python
def process_transaction_data(
    transaction_data: dict,
    validation_rules: list,
    timeout_seconds: int = 30
) -> TransactionResult:
    """
    Procesa datos de transacción aplicando reglas de validación.
    
    Args:
        transaction_data: Diccionario con datos de transacción requeridos
        validation_rules: Lista de reglas de validación a aplicar  
        timeout_seconds: Timeout máximo para procesamiento
    
    Returns:
        TransactionResult con resultado de procesamiento y metadatos
        
    Raises:
        ValidationError: Si transaction_data no pasa validaciones
        TimeoutError: Si procesamiento excede timeout_seconds
    """
    # Validación temprana de parámetros
    if not transaction_data:
        raise ValueError("transaction_data no puede estar vacío")
    
    # Lógica principal con manejo de errores
    try:
        validated_data = _validate_transaction_data(transaction_data, validation_rules)
        processed_result = _process_validated_data(validated_data)
        return processed_result
    except Exception as e:
        logger.error(f"Error procesando transacción: {e}")
        raise
```

#### Imports y Organización
```python
# 1. Librerías estándar
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

# 2. Librerías de terceros
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# 3. Imports locales
from cerverus.core.base import BaseDetector
from cerverus.utils.logging import get_logger
from cerverus.models.transaction import Transaction
```

### Estándares SQL

#### Convenciones de Nomenclatura
```sql
-- Tablas: snake_case con prefijo descriptivo
fraud_detection_signals
market_data_daily
user_transaction_history

-- Columnas: snake_case descriptivo
transaction_id
fraud_probability_score
created_at_utc

-- Índices: descriptivos con sufijo
idx_transactions_symbol_date
idx_fraud_signals_timestamp
```

#### Formato de Queries
```sql
-- Queries complejas: formato multi-línea con indentación
SELECT 
    t.transaction_id,
    t.symbol,
    t.amount,
    t.timestamp,
    fs.fraud_score,
    fs.algorithm_used
FROM transactions t
LEFT JOIN fraud_signals fs 
    ON t.transaction_id = fs.transaction_id
WHERE t.timestamp >= CURRENT_DATE - INTERVAL '30 days'
    AND t.amount > 1000
ORDER BY t.timestamp DESC, fs.fraud_score DESC
LIMIT 1000;
```

### Estándares YAML/JSON

#### Configuración YAML
```yaml
# Usar 2 espacios para indentación
# Agrupar configuraciones relacionadas
database:
  host: localhost
  port: 5432
  name: cerverus_production
  
fraud_detection:
  algorithms:
    - name: isolation_forest
      enabled: true
      parameters:
        contamination: 0.01
        n_estimators: 200
    
    - name: z_score_adaptive  
      enabled: true
      parameters:
        threshold: 3.0
        window_size: 30

# Comentarios descriptivos para configuraciones críticas
monitoring:
  # Intervalo de health checks en segundos
  health_check_interval: 30
  
  # Número máximo de reintentos antes de marcar como failed
  max_retry_attempts: 3
```

## Linting y Formateo Automático

### Configuración Black (Python)
```toml
# pyproject.toml
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
extend-exclude = '''
/(
  # Directorios a excluir
  \.eggs
  | \.git
  | \.venv
  | \.mypy_cache
  | build
  | dist
)/
'''
```

### Configuración Flake8
```ini
# .flake8
[flake8]
max-line-length = 88
extend-ignore = E203, E501, W503
exclude = 
    .git,
    __pycache__,
    .venv,
    .eggs,
    *.egg,
    build,
    dist

per-file-ignores =
    __init__.py:F401
    */migrations/*:E501
```

### Configuración MyPy
```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True

[mypy-pandas.*]
ignore_missing_imports = True

[mypy-numpy.*]
ignore_missing_imports = True
```

## Docstrings y Documentación

### Estándar para Funciones
```python
def calculate_isolation_forest_score(
    features: np.ndarray, 
    model_path: str,
    contamination: float = 0.01
) -> float:
    """
    Calcula score de anomalía usando Isolation Forest.
    
    Implementa detección de anomalías usando el algoritmo Isolation Forest
    para identificar transacciones potencialmente fraudulentas basado en
    características extraídas.
    
    Args:
        features: Array NumPy con características normalizadas de la transacción.
                 Shape esperado: (n_features,) donde n_features >= 10.
        model_path: Ruta al archivo del modelo Isolation Forest serializado.
                   Debe existir y ser un archivo .pkl válido.
        contamination: Proporción esperada de outliers en los datos.
                      Rango válido: 0.0 < contamination < 0.5.
                      Default: 0.01 (1% de datos son outliers).
    
    Returns:
        Score de anomalía entre -1.0 y 1.0. Valores más negativos indican
        mayor probabilidad de anomalía/fraude.
        
    Raises:
        FileNotFoundError: Si model_path no existe.
        ValueError: Si features tiene shape inválido o contamination fuera de rango.
        ModelLoadError: Si el modelo no puede ser deserializado correctamente.
        
    Example:
        >>> features = np.array([1.2, 0.8, -0.3, 2.1, 0.0])
        >>> score = calculate_isolation_forest_score(features, "models/fraud_model.pkl")
        >>> print(f"Anomaly score: {score:.3f}")
        Anomaly score: -0.142
        
    Note:
        - Scores < -0.5 se consideran alta probabilidad de fraude
        - El modelo debe ser entrenado con datos similares a los de producción
        - Features deben estar normalizadas usando el mismo scaler del entrenamiento
    """
```

### Estándar para Clases
```python
class CerverusFraudDetector:
    """
    Detector de fraude multi-algoritmo para transacciones financieras.
    
    Esta clase implementa un ensemble de algoritmos de detección de fraude
    incluyendo métodos estadísticos (Z-Score), machine learning no supervisado  
    (Isolation Forest) y deep learning (LSTM Autoencoder).
    
    Attributes:
        algorithms (List[str]): Lista de algoritmos habilitados para detección.
        threshold (float): Umbral de score para clasificar como fraude.
        model_registry (ModelRegistry): Registro de modelos ML cargados.
        
    Example:
        >>> detector = CerverusFraudDetector(algorithms=['isolation_forest', 'z_score'])
        >>> result = detector.detect_fraud(transaction_data)
        >>> if result.is_fraud:
        ...     print(f"Fraude detectado con confianza {result.confidence:.2f}")
    """
    
    def __init__(
        self, 
        algorithms: List[str],
        threshold: float = 0.8,
        config_path: Optional[str] = None
    ):
        """
        Inicializa detector de fraude con algoritmos especificados.
        
        Args:
            algorithms: Lista de algoritmos a usar. Opciones válidas:
                       ['z_score', 'isolation_forest', 'lstm_autoencoder']
            threshold: Umbral para clasificación de fraude (0.0-1.0)
            config_path: Ruta opcional a archivo de configuración YAML
            
        Raises:
            ValueError: Si algorithms contiene algoritmos no soportados
            ConfigError: Si config_path existe pero no es válido
        """
```

## Pre-commit Hooks

### Configuración .pre-commit-config.yaml
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.11
        
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
      
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.6.1
    hooks:
      - id: mypy
        additional_dependencies: [types-requests, types-PyYAML]
        
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: ["--profile", "black"]
        
  - repo: https://github.com/adrienverge/yamllint
    rev: v1.32.0
    hooks:
      - id: yamllint
        args: [--format, parsable, --strict]
```