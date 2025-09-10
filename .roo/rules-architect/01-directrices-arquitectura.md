---
title: "Arquitectura de 7 Etapas - Especificaciones Técnicas"
version: "1.0"
owner: "Arquitecto de Sistemas"
contact: "#team-architecture"
last_updated: "2025-09-09"
---

# Arquitectura de 7 Etapas - Especificaciones Técnicas

## Checklist de Calidad para Arquitectura
- [ ] Patrones de diseño fundamentales aplicados
- [ ] Escalabilidad horizontal verificada
- [ ] Cada etapa como servicio independiente deployable
- [ ] Event-driven communication implementada
- [ ] Observabilidad trifásica configurada
- [ ] Infraestructura cloud-native desplegada

## Principios Arquitectónicos Generales

### Patrones de Diseño Fundamentales
- **Microservicios**: Cada etapa como servicio independiente deployable
- **Event-Driven**: Comunicación asíncrona entre etapas via Kafka
- **CQRS**: Separación de comandos y queries para optimización
- **Circuit Breaker**: Protección ante fallos en cascada
- **Saga Pattern**: Coordinación de transacciones distribuidas

### Principios de Escalabilidad
- **Horizontal Scaling**: Auto-scaling basado en métricas de negocio
- **Stateless Services**: Servicios sin estado para facilitar escalado
- **Data Partitioning**: Particionamiento por símbolo y tiempo
- **Caching Strategy**: Cache multinivel (L1: Redis, L2: CDN, L3: S3)

## Etapa 1: Recolección de Datos - Especificaciones

### Arquitectura de Componentes
```
┌─────────────────────────────────────────────────────────────────┐
│                   Data Extraction Layer                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────┐│
│  │   Yahoo      │  │    SEC      │  │   FINRA     │  │ Alpha  ││
│  │   Finance    │  │    EDGAR    │  │   Data      │  │Vantage ││
│  └──────┬───────┘  └──────┬──────┘  └──────┬──────┘  └───┬────┘│
│         │                 │                │              │     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                Orchestration Engine                      │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐│ │
│  │  │Circuit      │  │Rate Limiter  │  │Cache Manager    ││ │
│  │  │Breaker      │  │Adaptive      │  │Multi-level      ││ │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘│ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              │                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    S3 Data Lake                             │ │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────────┐    │ │
│  │  │   Raw Data  │  │ Processed    │  │   ML Features   │    │ │
│  │  │   (Bronze)  │  │ Data (Silver)│  │     (Gold)      │    │ │
│  │  └─────────────┘  └──────────────┘  └─────────────────┘    │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Implementaciones Específicas

#### Adaptador Polimórfico para Yahoo Finance
```python
class YahooFinanceAdapter(DataSourceAdapter):
    def __init__(self):
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            expected_exception=RequestException
        )
        self.rate_limiter = RateLimiter(requests_per_second=10)
        self.cache = RedisCache(ttl_seconds=300)
    
    @circuit_breaker.protect
    @rate_limiter.limit
    def extract_data(self, symbol: str, start_date: datetime, end_date: datetime) -> CerverusDataFrame:
        cache_key = f"yahoo:{symbol}:{start_date}:{end_date}"
        
        # Intentar cache primero
        cached_data = self.cache.get(cache_key)
        if cached_data and not self._is_stale(cached_data):
            return cached_data
        
        # Extraer de Yahoo Finance
        raw_data = yfinance.download(symbol, start=start_date, end=end_date)
        
        # Validar y transformar
        validated_data = self._validate_market_data(raw_data)
        transformed_data = self._transform_to_cerverus_format(validated_data)
        
        # Cache resultado
        self.cache.set(cache_key, transformed_data)
        
        return transformed_data
```

#### Reconciliador de Datos Multi-Fuente
```python
class DataReconciler:
    def reconcile_market_data(self, sources_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Reconcilia datos de múltiples fuentes priorizando por confiabilidad.
        Prioridad: Yahoo Finance > Alpha Vantage > SEC EDGAR
        """
        reconciled_records = []
        
        for symbol in self._get_all_symbols(sources_data):
            for date in self._get_all_dates(sources_data, symbol):
                record = self._reconcile_single_record(symbol, date, sources_data)
                if record:
                    reconciled_records.append(record)
        
        return pd.DataFrame(reconciled_records)
    
    def _reconcile_single_record(self, symbol: str, date: datetime, sources_data: Dict) -> Optional[Dict]:
        yahoo_data = self._get_record(sources_data.get('yahoo'), symbol, date)
        alpha_data = self._get_record(sources_data.get('alpha_vantage'), symbol, date)
        
        if yahoo_data and alpha_data:
            # Validar consistencia entre fuentes
            price_diff = abs(yahoo_data['close'] - alpha_data['close']) / yahoo_data['close']
            if price_diff > 0.02:  # Diferencia > 2%
                self._flag_data_discrepancy(symbol, date, yahoo_data, alpha_data)
            
            # Usar Yahoo Finance como principal
            return self._create_reconciled_record(yahoo_data, alpha_data, primary_source='yahoo')
        
        return yahoo_data or alpha_data
```

## Etapa 2: Almacenamiento - Medallion Architecture

### Bronze Layer - Almacenamiento Crudo
```python
class BronzeLayerManager:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        self.bucket_name = 'cerverus-bronze-data'
        
    def store_raw_data(self, source: str, symbol: str, data: pd.DataFrame) -> str:
        """
        Almacena datos crudos con particionamiento automático.
        
        Estructura S3:
        s3://cerverus-bronze-data/
        ├── source=yahoo_finance/
        │   └── year=2024/month=03/day=15/
        │       └── symbol=AAPL/
        │           └── data.parquet
        """
        partition_path = self._generate_partition_path(source, symbol, datetime.utcnow())
        
        # Comprimir con Parquet
        parquet_buffer = self._dataframe_to_parquet(data)
        
        # Generar metadatos
        metadata = {
            'source': source,
            'symbol': symbol,
            'records_count': str(len(data)),
            'schema_version': '1.0',
            'ingestion_timestamp': datetime.utcnow().isoformat()
        }
        
        # Almacenar en S3
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=partition_path,
            Body=parquet_buffer,
            Metadata=metadata
        )
        
        return f"s3://{self.bucket_name}/{partition_path}"
```

### Silver Layer - Datos Curados con dbt
```sql
-- models/silver/silver_market_data.sql
{{ 
  config(
    materialized='incremental',
    unique_key='market_data_id',
    partition_by={
      'field': 'trade_date',
      'data_type': 'date'
    },
    cluster_by=['symbol', 'trade_date']
  ) 
}}

WITH bronze_reconciled AS (
  SELECT 
    symbol,
    trade_date,
    open_price,
    high_price,
    low_price,
    close_price,
    volume,
    adj_close_price,
    data_quality_score,
    source_primary,
    source_secondary,
    created_at
  FROM {{ ref('bronze_reconciled_data') }}
  {% if is_incremental() %}
    WHERE created_at > (SELECT MAX(created_at) FROM {{ this }})
  {% endif %}
),

enhanced_data AS (
  SELECT 
    *,
    -- Generar ID único
    {{ generate_market_data_id('symbol', 'trade_date') }} AS market_data_id,
    
    -- Calcular características técnicas
    {{ calculate_returns('close_price') }} AS daily_return,
    {{ calculate_volatility('close_price', 20) }} AS volatility_20d,
    {{ calculate_rsi('close_price', 14) }} AS rsi_14d,
    
    -- Detectar anomalías básicas
    {{ detect_price_gaps('high_price', 'low_price') }} AS price_gap_flag,
    {{ detect_volume_spikes('volume', 'symbol') }} AS volume_spike_flag
    
  FROM bronze_reconciled
)

SELECT * FROM enhanced_data

-- Tests de calidad
{{ test_not_null(['market_data_id', 'symbol', 'trade_date']) }}
{{ test_unique(['market_data_id']) }}
{{ test_relationships('symbol', ref('dim_symbols'), 'symbol') }}
```

## Etapa 3: Procesamiento - Pipeline Híbrido

### Procesamiento Batch con dbt
```sql
-- models/features/features_statistical_analysis.sql
WITH base_data AS (
  SELECT * FROM {{ ref('silver_market_data') }}
  WHERE trade_date >= CURRENT_DATE - 90  -- Rolling 90-day window
),

statistical_features AS (
  SELECT 
    symbol,
    trade_date,
    close_price,
    volume,
    
    -- Z-Score Adaptativo
    (close_price - AVG(close_price) OVER (
      PARTITION BY symbol 
      ORDER BY trade_date 
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    )) / NULLIF(STDDEV(close_price) OVER (
      PARTITION BY symbol 
      ORDER BY trade_date 
      ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ), 0) AS price_zscore_30d,
    
    -- Grubbs Test Statistic
    {{ grubbs_test_statistic('close_price', 'symbol', 30) }} AS grubbs_stat_price,
    
    -- CUSUM para detectar cambios de tendencia
    {{ cusum_statistic('daily_return', 'symbol') }} AS cusum_returns,
    
    -- Percentiles para detección de outliers
    PERCENT_RANK() OVER (
      PARTITION BY symbol 
      ORDER BY close_price
    ) AS price_percentile_rank,
    
    PERCENT_RANK() OVER (
      PARTITION BY symbol 
      ORDER BY volume
    ) AS volume_percentile_rank
    
  FROM base_data
)

SELECT * FROM statistical_features
```

### Procesamiento Streaming con Apache Flink
```java
public class CerverusRealtimeFraudDetector extends ProcessWindowFunction<MarketData, FraudSignal, String, TimeWindow> {
    
    // Algoritmos de detección en tiempo real
    private transient ZScoreDetector zScoreDetector;
    private transient IsolationForestDetector isolationDetector;
    private transient PriceManipulationDetector priceManipDetector;
    
    @Override
    public void open(Configuration parameters) {
        // Inicializar detectores
        zScoreDetector = new ZScoreDetector(windowSize = 30, threshold = 3.0);
        isolationDetector = new IsolationForestDetector("models/isolation_forest.pkl");
        priceManipDetector = new PriceManipulationDetector();
    }
    
    @Override
    public void process(String symbol, Context context, 
                       Iterable<MarketData> elements, 
                       Collector<FraudSignal> out) {
        
        List<MarketData> windowData = Lists.newArrayList(elements);
        if (windowData.size() < 10) return; // Datos insuficientes
        
        // Extraer características para ventana actual
        FeatureVector features = extractRealtimeFeatures(windowData);
        
        // Aplicar detectores en paralelo
        CompletableFuture<Double> zScoreFuture = CompletableFuture.supplyAsync(
            () -> zScoreDetector.detectAnomaly(features)
        );
        
        CompletableFuture<Double> isolationFuture = CompletableFuture.supplyAsync(
            () -> isolationDetector.detectAnomaly(features)
        );
        
        CompletableFuture<Double> manipulationFuture = CompletableFuture.supplyAsync(
            () -> priceManipDetector.detectManipulation(features)
        );
        
        // Combinar resultados con ensemble
        try {
            double zScore = zScoreFuture.get(100, TimeUnit.MILLISECONDS);
            double isoScore = isolationFuture.get(100, TimeUnit.MILLISECONDS);
            double manipScore = manipulationFuture.get(100, TimeUnit.MILLISECONDS);
            
            double ensembleScore = calculateEnsembleScore(zScore, isoScore, manipScore);
            
            if (ensembleScore > FRAUD_THRESHOLD) {
                FraudSignal signal = FraudSignal.builder()
                    .symbol(symbol)
                    .timestamp(context.window().getEnd())
                    .confidenceScore(ensembleScore)
                    .algorithmScores(Map.of(
                        "zscore", zScore,
                        "isolation_forest", isoScore,
                        "price_manipulation", manipScore
                    ))
                    .features(features)
                    .build();
                
                out.collect(signal);
            }
            
        } catch (TimeoutException e) {
            // Log timeout pero continuar procesamiento
            logger.warn("Detection timeout for symbol: {}", symbol);
        }
    }
}
```

## Etapa 4: Orquestación - DAGs Dinámicos

### Generador de DAGs Dinámicos
```python
# dags/cerverus_dag_factory.py
class CerverusDynamicDAGFactory:
    def __init__(self):
        self.symbols_config = Variable.get("symbols_config", deserialize_json=True)
        self.market_activity_monitor = MarketActivityMonitor()
    
    def generate_fraud_detection_dags(self) -> List[DAG]:
        generated_dags = []
        
        for symbol, config in self.symbols_config.items():
            if config.get('enabled', True):
                dag = self._create_symbol_specific_dag(symbol, config)
                generated_dags.append(dag)
        
        return generated_dags
    
    def _create_symbol_specific_dag(self, symbol: str, config: dict) -> DAG:
        risk_level = config.get('risk_level', 'medium')
        
        # Configuración adaptativa por riesgo
        schedule_map = {
            'high': timedelta(minutes=5),
            'medium': timedelta(minutes=15),
            'low': timedelta(hours=1)
        }
        
        dag = DAG(
            dag_id=f"fraud_detection_{symbol.lower()}",
            schedule_interval=schedule_map[risk_level],
            max_active_runs=3 if risk_level == 'high' else 1,
            catchup=False,
            tags=['fraud-detection', f'risk-{risk_level}', symbol]
        )
        
        with dag:
            # TaskGroup para extracción de datos
            with TaskGroup('data_extraction') as extraction_group:
                extract_realtime = PythonOperator(
                    task_id='extract_realtime_data',
                    python_callable=self._extract_realtime_data,
                    op_kwargs={'symbol': symbol}
                )
                
                extract_historical = PythonOperator(
                    task_id='extract_historical_context',
                    python_callable=self._extract_historical_context,
                    op_kwargs={'symbol': symbol, 'days': config.get('lookback_days', 30)}
                )
            
            # TaskGroup para detección de fraude
            with TaskGroup('fraud_detection') as detection_group:
                if risk_level == 'high':
                    # Para símbolos de alto riesgo, usar todos los algoritmos
                    run_statistical = PythonOperator(
                        task_id='run_statistical_detection',
                        python_callable=self._run_statistical_algorithms,
                        op_kwargs={'symbol': symbol, 'algorithms': ['zscore', 'grubbs', 'cusum']}
                    )
                    
                    run_ml = PythonOperator(
                        task_id='run_ml_detection', 
                        python_callable=self._run_ml_algorithms,
                        op_kwargs={'symbol': symbol, 'algorithms': ['isolation_forest', 'lof']}
                    )
                    
                    run_ensemble = PythonOperator(
                        task_id='run_ensemble_detection',
                        python_callable=self._run_ensemble_algorithms,
                        op_kwargs={'symbol': symbol}
                    )
                    
                    [run_statistical, run_ml] >> run_ensemble
                else:
                    # Para símbolos de menor riesgo, solo algoritmos básicos
                    run_basic_detection = PythonOperator(
                        task_id='run_basic_detection',
                        python_callable=self._run_basic_algorithms,
                        op_kwargs={'symbol': symbol}
                    )
            
            # Dependencias principales
            extraction_group >> detection_group
        
        return dag

# Generar DAGs
dag_factory = CerverusDynamicDAGFactory()
for dag in dag_factory.generate_fraud_detection_dags():
    globals()[dag.dag_id] = dag
```

## Etapa 5: ML y Calidad de Datos - 4 Tiers

### Tier 1: Algoritmos Estadísticos
```python
class ZScoreAdaptiveDetector:
    def __init__(self, base_window=30, adaptation_factor=0.1):
        self.base_window = base_window
        self.adaptation_factor = adaptation_factor
        self.adaptive_threshold = 3.0
        
    def detect_anomalies(self, data: pd.Series) -> pd.Series:
        """Detección con Z-Score adaptativo que ajusta threshold dinámicamente"""
        rolling_mean = data.rolling(window=self.base_window).mean()
        rolling_std = data.rolling(window=self.base_window).std()
        
        # Calcular Z-scores
        z_scores = (data - rolling_mean) / rolling_std
        
        # Adaptar threshold basado en volatilidad
        volatility = rolling_std.rolling(window=5).mean()
        avg_volatility = volatility.mean()
        
        # Threshold más alto durante alta volatilidad
        adapted_threshold = self.adaptive_threshold * (1 + volatility / avg_volatility)
        
        # Detectar anomalías
        anomalies = abs(z_scores) > adapted_threshold
        
        return anomalies
```

### Tier 2: ML No Supervisado
```python
class EnsembleIsolationForest:
    def __init__(self, n_models=5, contamination=0.01):
        self.n_models = n_models
        self.contamination = contamination
        self.models = []
        self.feature_subsets = []
        
    def fit(self, X: np.ndarray):
        """Entrenar ensemble de modelos con feature subsampling"""
        n_features = X.shape[1]
        
        for i in range(self.n_models):
            # Submuestreo de features para diversidad
            n_selected = int(n_features * 0.8)
            selected_features = np.random.choice(n_features, n_selected, replace=False)
            
            # Entrenar modelo con subset de features
            model = IsolationForest(
                contamination=self.contamination,
                n_estimators=200,
                random_state=42 + i
            )
            model.fit(X[:, selected_features])
            
            self.models.append(model)
            self.feature_subsets.append(selected_features)
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Predecir score de anomalía usando ensemble"""
        scores = []
        
        for model, features in zip(self.models, self.feature_subsets):
            score = model.decision_function(X[:, features])
            scores.append(score)
        
        # Promedio ponderado de scores
        ensemble_score = np.mean(scores, axis=0)
        return ensemble_score
```

### Tier 3: Deep Learning
```python
class LSTMFraudDetector(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout
        )
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM para capturar dependencias temporales
        lstm_out, _ = self.lstm(x)
        
        # Attention para enfocarse en patrones importantes
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Clasificación usando última salida
        fraud_score = self.classifier(attn_out[:, -1, :])
        
        return fraud_score

class GraphNeuralNetwork(nn.Module):
    """GNN para detectar redes de fraude coordinado"""
    
    def __init__(self, node_features, hidden_dim=64):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, edge_index):
        # Convoluciones de grafo
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        
        # Clasificación por nodo
        fraud_scores = torch.sigmoid(self.classifier(x))
        
        return fraud_scores
```

### Tier 4: Ensemble Final
```python
class CerverusEnsembleClassifier:
    def __init__(self):
        self.statistical_weight = 0.2
        self.ml_weight = 0.3
        self.dl_weight = 0.3
        self.meta_weight = 0.2
        
        # Meta-learner para combinar predicciones
        self.meta_classifier = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1
        )
    
    def predict(self, transaction_features: np.ndarray) -> Dict[str, float]:
        """Predicción usando ensemble completo"""
        
        # Tier 1: Algoritmos estadísticos
        statistical_scores = self._get_statistical_scores(transaction_features)
        
        # Tier 2: ML no supervisado
        ml_scores = self._get_ml_scores(transaction_features)
        
        # Tier 3: Deep learning
        dl_scores = self._get_dl_scores(transaction_features)
        
        # Combinar scores con pesos
        base_ensemble_score = (
            self.statistical_weight * statistical_scores +
            self.ml_weight * ml_scores +
            self.dl_weight * dl_scores
        )
        
        # Meta-learning para refinamiento final
        meta_features = np.column_stack([
            statistical_scores, ml_scores, dl_scores,
            transaction_features  # Features originales
        ])
        
        meta_score = self.meta_classifier.predict_proba(meta_features)[:, 1]
        
        # Score final combinado
        final_score = (
            (1 - self.meta_weight) * base_ensemble_score +
            self.meta_weight * meta_score
        )
        
        return {
            'final_score': final_score[0],
            'statistical_score': statistical_scores[0],
            'ml_score': ml_scores[0],
            'dl_score': dl_scores[0],
            'meta_score': meta_score[0],
            'confidence': self._calculate_confidence(statistical_scores, ml_scores, dl_scores)
        }
```

## Etapa 6: Infraestructura - Cloud Native

### Kubernetes Deployment con Auto-scaling
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cerverus-fraud-detection
  namespace: cerverus-system
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: fraud-detection
  template:
    metadata:
      labels:
        app: fraud-detection
        version: v1
    spec:
      containers:
      - name: fraud-detection
        image: cerverus/fraud-detection:v1.2.0
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: cerverus-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fraud-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cerverus-fraud-detection
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: fraud_detection_requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
```

## Etapa 7: Monitoreo - Observabilidad Trifásica

### Métricas de Negocio
```python
# Métricas específicas para detección de fraude
FRAUD_SIGNALS_TOTAL = Counter(
    'fraud_signals_total',
    'Total fraud signals generated',
    ['symbol', 'algorithm', 'confidence_level']
)

FRAUD_DETECTION_LATENCY = Histogram(
    'fraud_detection_duration_seconds',
    'Time spent processing fraud detection',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

MODEL_ACCURACY = Gauge(
    'fraud_model_accuracy',
    'Current model accuracy',
    ['model_name', 'model_version']
)
```

### Logging Estructurado con Correlación
```python
import structlog

logger = structlog.get_logger()

def detect_fraud_with_observability(transaction_data: dict) -> dict:
    trace_id = generate_trace_id()
    
    with tracer.start_as_current_span("fraud_detection") as span:
        span.set_attribute("transaction_id", transaction_data["id"])
        span.set_attribute("symbol", transaction_data["symbol"])
        
        start_time = time.time()
        
        try:
            result = fraud_detector.predict(transaction_data)
            
            # Log éxito con contexto
            logger.info(
                "Fraud detection completed",
                transaction_id=transaction_data["id"],
                symbol=transaction_data["symbol"],
                fraud_score=result["final_score"],
                is_fraud=result["final_score"] > 0.8,
                processing_time=time.time() - start_time,
                trace_id=trace_id
            )
            
            # Actualizar métricas
            FRAUD_DETECTION_LATENCY.observe(time.time() - start_time)
            
            if result["final_score"] > 0.8:
                FRAUD_SIGNALS_TOTAL.labels(
                    symbol=transaction_data["symbol"],
                    algorithm="ensemble",
                    confidence_level="high"
                ).inc()
            
            return result
            
        except Exception as e:
            logger.error(
                "Fraud detection failed",
                transaction_id=transaction_data["id"],
                error=str(e),
                trace_id=trace_id
            )
            raise
```