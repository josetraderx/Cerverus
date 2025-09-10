# metrics utilities for Prometheus / OpenTelemetry
"""
Metrics and Monitoring System for Cerverus
==========================================

Implements comprehensive metrics with Prometheus for all system stages.
Provides full observability of the fraud detection pipeline.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, CollectorRegistry, 
    multiprocess, generate_latest, start_http_server
)
import structlog
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import psutil
import asyncio
from contextlib import contextmanager
import json

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of metrics available in the system."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Definition of a system metric."""
    name: str
    help_text: str
    labels: List[str]
    metric_type: MetricType
    buckets: Optional[List[float]] = None  # For histograms


class CerverusMetricsCollector:
    """
    Central metrics collector for the Cerverus System.
    
    Provides performance, data quality, fraud detection,
    and infrastructure metrics in a centralized and standardized way.
    """
    
    def __init__(self, environment: str = "production"):
        self.environment = environment
        self.registry = CollectorRegistry()
        self._metrics: Dict[str, Any] = {}
        self._initialize_metrics()
        self._start_background_collection()
        
        logger.info(
            "Metrics collector initialized",
            environment=self.environment,
            metrics_count=len(self._metrics)
        )
    
    def _initialize_metrics(self):
        """Initializes all system metrics."""
        
        # ========== DATA INGESTION METRICS ==========
        self._metrics['data_extraction_total'] = Counter(
            'cerverus_data_extraction_total',
            'Total number of data extraction operations',
            ['source', 'status', 'environment'],
            registry=self.registry
        )
        
        self._metrics['data_extraction_duration'] = Histogram(
            'cerverus_data_extraction_duration_seconds',
            'Time spent extracting data from sources',
            ['source', 'environment'],
            buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')),
            registry=self.registry
        )
        
        self._metrics['data_records_processed'] = Counter(
            'cerverus_data_records_processed_total',
            'Total number of data records processed',
            ['source', 'layer', 'environment'],
            registry=self.registry
        )
        
        # ========== DATA QUALITY METRICS ==========
        self._metrics['data_quality_score'] = Gauge(
            'cerverus_data_quality_score',
            'Data quality score by source (0-1)',
            ['source', 'quality_dimension', 'environment'],
            registry=self.registry
        )
        
        self._metrics['data_validation_errors'] = Counter(
            'cerverus_data_validation_errors_total',
            'Total number of data validation errors',
            ['source', 'error_type', 'environment'],
            registry=self.registry
        )
        
        self._metrics['data_freshness_minutes'] = Gauge(
            'cerverus_data_freshness_minutes',
            'Data freshness in minutes since last update',
            ['source', 'environment'],
            registry=self.registry
        )
        
        # ========== FRAUD DETECTION METRICS ==========
        self._metrics['fraud_signals_total'] = Counter(
            'cerverus_fraud_signals_total',
            'Total number of fraud signals generated',
            ['algorithm', 'severity', 'symbol', 'environment'],
            registry=self.registry
        )
        
        self._metrics['fraud_detection_latency'] = Histogram(
            'cerverus_fraud_detection_latency_seconds',
            'Time to detect fraud from data ingestion',
            ['algorithm', 'environment'],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, float('inf')),
            registry=self.registry
        )
        
        self._metrics['model_accuracy'] = Gauge(
            'cerverus_model_accuracy',
            'Model accuracy score for fraud detection',
            ['model_name', 'metric_type', 'environment'],
            registry=self.registry
        )
        
        # ========== PIPELINE METRICS ==========
        self._metrics['pipeline_execution_total'] = Counter(
            'cerverus_pipeline_execution_total',
            'Total number of pipeline executions',
            ['pipeline_name', 'status', 'environment'],
            registry=self.registry
        )
        
        self._metrics['pipeline_duration'] = Histogram(
            'cerverus_pipeline_duration_seconds',
            'Pipeline execution duration',
            ['pipeline_name', 'environment'],
            buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 900.0, 1800.0, float('inf')),
            registry=self.registry
        )
        
        # ========== INFRASTRUCTURE METRICS ==========
        self._metrics['system_cpu_usage'] = Gauge(
            'cerverus_system_cpu_usage_percent',
            'System CPU usage percentage',
            ['instance', 'environment'],
            registry=self.registry
        )
        
        self._metrics['system_memory_usage'] = Gauge(
            'cerverus_system_memory_usage_percent',
            'System memory usage percentage',
            ['instance', 'environment'],
            registry=self.registry
        )
        
        self._metrics['active_connections'] = Gauge(
            'cerverus_active_connections',
            'Number of active connections by service',
            ['service', 'environment'],
            registry=self.registry
        )
        
        # ========== BUSINESS KPIs ==========
        self._metrics['investigations_completed'] = Counter(
            'cerverus_investigations_completed_total',
            'Total number of fraud investigations completed',
            ['status', 'priority', 'environment'],
            registry=self.registry
        )
        
        self._metrics['false_positive_rate'] = Gauge(
            'cerverus_false_positive_rate',
            'False positive rate for fraud detection',
            ['time_window', 'environment'],
            registry=self.registry
        )
    
    def _start_background_collection(self):
        """Starts background system metrics collection."""
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self._collection_thread.start()
    
    def _collect_system_metrics(self):
        """Collects system metrics every 30 seconds."""
        while True:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                self._metrics['system_cpu_usage'].labels(
                    instance='main',
                    environment=self.environment
                ).set(cpu_percent)
                
                # Memory metrics
                memory = psutil.virtual_memory()
                self._metrics['system_memory_usage'].labels(
                    instance='main',
                    environment=self.environment
                ).set(memory.percent)
                
                time.sleep(30)
                
            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
                time.sleep(30)
    
    @contextmanager
    def track_operation(self, operation_name: str, labels: Dict[str, str] = None):
        """
        Context manager to track operation duration.
        
        Args:
            operation_name: Name of the operation to track
            labels: Additional labels for the metric
        """
        labels = labels or {}
        labels['environment'] = self.environment
        
        start_time = time.time()
        
        try:
            yield
            # Success
            labels['status'] = 'success'
            
        except Exception as e:
            # Error
            labels['status'] = 'error'
            labels['error_type'] = type(e).__name__
            raise
            
        finally:
            duration = time.time() - start_time
            
            # Register duration if the metric exists
            if f"{operation_name}_duration" in self._metrics:
                self._metrics[f"{operation_name}_duration"].labels(**labels).observe(duration)
            
            # Register total counter
            if f"{operation_name}_total" in self._metrics:
                self._metrics[f"{operation_name}_total"].labels(**labels).inc()
