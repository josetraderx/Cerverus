"""
SLA and Business Metrics Monitoring System for Cerverus System
=============================================================

Implements monitoring of SLOs (Service Level Objectives), calculation of error budgets,
and tracking of critical business KPIs for the fraud detection system.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import structlog
import asyncio
from prometheus_client import Gauge, Counter, Histogram
import numpy as np
from collections import defaultdict, deque
import json

logger = structlog.get_logger()


class SLOStatus(Enum):
    """Possible states of an SLO."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACHED = "breached"


@dataclass
class SLODefinition:
    """Definition of a Service Level Objective."""
    name: str
    description: str
    target_percentage: float  # e.g., 99.9 for 99.9%
    time_window_hours: int    # e.g., 24 for 24-hour window
    metric_query: str         # Prometheus query to measure the SLO
    error_budget_policy: str  # Policy when error budget is exhausted
    team_responsible: str
    business_impact: str


@dataclass
class SLOMeasurement:
    """Current measurement of an SLO."""
    timestamp: datetime
    actual_percentage: float
    target_percentage: float
    error_budget_remaining: float  # Remaining error budget percentage
    status: SLOStatus
    time_to_exhaustion: Optional[timedelta] = None  # Estimated time to budget exhaustion


@dataclass
class BusinessKPI:
    """Definition of a business KPI."""
    name: str
    description: str
    target_value: float
    unit: str
    measurement_frequency: str  # daily, hourly, real-time
    team_responsible: str
    business_criticality: str  # low, medium, high, critical


class SLAMonitor:
    """
    Main SLA and SLO monitor for the Cerverus System.
    
    Tracks service level objectives, calculates error budgets,
    and provides early alerts for service degradation.
    """
    
    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector
        self.slo_definitions = self._initialize_slos()
        self.business_kpis = self._initialize_business_kpis()
        self.measurements_history = defaultdict(deque)
        self._initialize_metrics()
        
        logger.info(
            "SLA Monitor initialized",
            slos_count=len(self.slo_definitions),
            kpis_count=len(self.business_kpis)
        )
    
    def _initialize_slos(self) -> Dict[str, SLODefinition]:
        """Initializes the system's SLO definitions."""
        return {
            "fraud_detection_availability": SLODefinition(
                name="fraud_detection_availability",
                description="Availability of the fraud detection system during market hours",
                target_percentage=99.9,
                time_window_hours=24,
                metric_query='avg_over_time(up{job="cerverus-fraud-detection"}[24h]) * 100',
                error_budget_policy="halt_deployments",
                team_responsible="platform",
                business_impact="Automatic trading disabled without fraud detection"
            ),
            
            "fraud_detection_latency": SLODefinition(
                name="fraud_detection_latency",
                description="P95 fraud detection latency < 100ms",
                target_percentage=95.0,
                time_window_hours=1,
                metric_query='histogram_quantile(0.95, rate(cerverus_fraud_detection_latency_seconds_bucket[1h])) < 0.1',
                error_budget_policy="throttle_traffic",
                team_responsible="ml-engineering",
                business_impact="Late detection allows more successful frauds"
            ),
            
            "data_ingestion_success": SLODefinition(
                name="data_ingestion_success",
                description="Data ingestion success rate > 99.5%",
                target_percentage=99.5,
                time_window_hours=24,
                metric_query='(sum(rate(cerverus_data_extraction_total{status="success"}[24h])) / sum(rate(cerverus_data_extraction_total[24h]))) * 100',
                error_budget_policy="escalate_to_oncall",
                team_responsible="data-engineering",
                business_impact="Insufficient data degrades detection accuracy"
            ),
            
            "data_freshness": SLODefinition(
                name="data_freshness",
                description="Data no older than 5 minutes 99% of the time",
                target_percentage=99.0,
                time_window_hours=24,
                metric_query='(sum(rate(cerverus_data_freshness_minutes <= 5)[24h]) / sum(rate(cerverus_data_freshness_minutes)[24h])) * 100',
                error_budget_policy="alert_business_users",
                team_responsible="data-engineering",
                business_impact="Decisions based on stale data"
            ),
            
            "model_accuracy": SLODefinition(
                name="model_accuracy",
                description="ML model accuracy > 90%",
                target_percentage=90.0,
                time_window_hours=168,
                metric_query='avg_over_time(cerverus_model_accuracy{metric_type="accuracy"}[168h]) * 100',
                error_budget_policy="trigger_model_retraining",
                team_responsible="ml-engineering",
                business_impact="Increase in false positives and undetected frauds"
            ),
            
            "investigation_completion": SLODefinition(
                name="investigation_completion",
                description="Investigations completed in < 2 hours on average",
                target_percentage=95.0,
                time_window_hours=24,
                metric_query='histogram_quantile(0.95, rate(cerverus_investigation_duration_seconds_bucket[24h])) < 7200',
                error_budget_policy="assign_more_analysts",
                team_responsible="fraud-analysts",
                business_impact="Delayed response allows fraud escalation"
            )
        }
    
    def _initialize_business_kpis(self) -> Dict[str, BusinessKPI]:
        """Initializes the system's business KPIs."""
        return {
            "fraud_detection_rate": BusinessKPI(
                name="fraud_detection_rate",
                description="Rate of confirmed frauds detected",
                target_value=95.0,
                unit="percentage",
                measurement_frequency="daily",
                team_responsible="fraud-analysts",
                business_criticality="critical"
            ),
            
            "false_positive_rate": BusinessKPI(
                name="false_positive_rate",
                description="Rate of false positives in detection",
                target_value=5.0,
                unit="percentage",
                measurement_frequency="daily",
                team_responsible="ml-engineering",
                business_criticality="high"
            ),
            
            "investigation_efficiency": BusinessKPI(
                name="investigation_efficiency",
                description="Average fraud investigation time",
                target_value=1.5,
                unit="hours",
                measurement_frequency="daily",
                team_responsible="fraud-analysts",
                business_criticality="high"
            ),
            
            "system_cost_per_transaction": BusinessKPI(
                name="system_cost_per_transaction",
                description="System cost per processed transaction",
                target_value=0.001,
                unit="usd",
                measurement_frequency="daily",
                team_responsible="platform",
                business_criticality="medium"
            ),
            
            "fraud_prevention_savings": BusinessKPI(
                name="fraud_prevention_savings",
                description="Money saved by preventing frauds",
                target_value=1000000.0,
                unit="usd",
                measurement_frequency="monthly",
                team_responsible="fraud-analysts",
                business_criticality="critical"
            ),
            
            "data_quality_score": BusinessKPI(
                name="data_quality_score",
                description="Average data quality score",
                target_value=95.0,
                unit="percentage",
                measurement_frequency="hourly",
                team_responsible="data-quality",
                business_criticality="high"
            )
        }
    
    def _initialize_metrics(self):
        """Initializes Prometheus metrics for SLAs."""
        self.slo_status_gauge = Gauge(
            'cerverus_slo_status',
            'Current SLO status (0=healthy, 1=warning, 2=critical, 3=breached)',
            ['slo_name', 'team'],
            registry=self.metrics_collector.registry
        )
        
        self.error_budget_gauge = Gauge(
            'cerverus_error_budget_remaining',
            'Remaining error budget percentage',
            ['slo_name', 'team'],
            registry=self.metrics_collector.registry
        )
        
        self.business_kpi_gauge = Gauge(
            'cerverus_business_kpi_value',
            'Current business KPI value',
            ['kpi_name', 'team', 'criticality'],
            registry=self.metrics_collector.registry
        )
        
        self.slo_compliance_counter = Counter(
            'cerverus_slo_compliance_total',
            'Total SLO compliance measurements',
            ['slo_name', 'status'],
            registry=self.metrics_collector.registry
        )
    
    async def measure_slo(self, slo_name: str) -> SLOMeasurement:
        """
        Measures the current state of a specific SLO.
        
        Args:
            slo_name: Name of the SLO to measure
            
        Returns:
            Current SLO measurement
        """
        if slo_name not in self.slo_definitions:
            raise ValueError(f"SLO {slo_name} is not defined")
        
        slo_def = self.slo_definitions[slo_name]
        
        # In production, this would query Prometheus
        # For now we simulate the measurement
        actual_percentage = await self._query_prometheus_metric(slo_def.metric_query)
        
        # Calculate remaining error budget
        error_budget_remaining = self._calculate_error_budget(
            actual_percentage, 
            slo_def.target_percentage, 
            slo_def.time_window_hours
        )
        
        # Determine status
        status = self._determine_slo_status(actual_percentage, slo_def.target_percentage, error_budget_remaining)
        
        # Estimate time to error budget exhaustion
        time_to_exhaustion = self._estimate_time_to_exhaustion(
            slo_name, 
            error_budget_remaining, 
            actual_percentage, 
            slo_def.target_percentage
        )
        
        measurement = SLOMeasurement(
            timestamp=datetime.utcnow(),
            actual_percentage=actual_percentage,
            target_percentage=slo_def.target_percentage,
            error_budget_remaining=error_budget_remaining,
            status=status,
            time_to_exhaustion=time_to_exhaustion
        )
        
        # Save in history
        self.measurements_history[slo_name].append(measurement)
        if len(self.measurements_history[slo_name]) > 1000:
            self.measurements_history[slo_name].popleft()
        
        # Update Prometheus metrics
        self._update_slo_metrics(slo_name, measurement)
        
        logger.info(
            "SLO measured",
            slo_name=slo_name,
            actual_percentage=actual_percentage,
            target_percentage=slo_def.target_percentage,
            error_budget_remaining=error_budget_remaining,
            status=status.value
        )
        
        return measurement
    
    def _calculate_error_budget(self, actual_percentage: float, target_percentage: float, time_window_hours: int) -> float:
        """
        Calculates the remaining error budget.
        
        Args:
            actual_percentage: Current measured percentage
            target_percentage: Target percentage
            time_window_hours: Time window in hours
            
        Returns:
            Remaining error budget percentage (0-100)
        """
        if actual_percentage >= target_percentage:
            return 100.0  # No error budget consumed
        
        total_error_budget = 100 - target_percentage
        error_consumed = target_percentage - actual_percentage
        error_budget_remaining = max(0, (total_error_budget - error_consumed) / total_error_budget * 100)
        
        return error_budget_remaining
    
    def _determine_slo_status(self, actual_percentage: float, target_percentage: float, error_budget_remaining: float) -> SLOStatus:
        """Determines SLO status based on measurements."""
        if actual_percentage >= target_percentage:
            return SLOStatus.HEALTHY
        elif error_budget_remaining > 50:
            return SLOStatus.WARNING
        elif error_budget_remaining > 10:
            return SLOStatus.CRITICAL
        else:
            return SLOStatus.BREACHED
    
    def _estimate_time_to_exhaustion(self, slo_name: str, error_budget_remaining: float, actual_percentage: float, target_percentage: float) -> Optional[timedelta]:
        """Estimates time to error budget exhaustion."""
        if error_budget_remaining >= 100 or len(self.measurements_history[slo_name]) < 10:
            return None
        
        recent_measurements = list(self.measurements_history[slo_name])[-10:]
        error_budgets = [m.error_budget_remaining for m in recent_measurements]
        
        x = np.arange(len(error_budgets))
        z = np.polyfit(x, error_budgets, 1)
        slope = z[0]
        
        if slope >= 0:
            return None
        
        measurements_to_exhaustion = error_budget_remaining / abs(slope)
        minutes_to_exhaustion = measurements_to_exhaustion * 5
        return timedelta(minutes=minutes_to_exhaustion)
    
    async def _query_prometheus_metric(self, query: str) -> float:
        """Queries a Prometheus metric. Returns simulated values for now."""
        import random
        
        if "availability" in query:
            return random.uniform(99.8, 99.99)
        elif "latency" in query:
            return random.uniform(92, 98)
        elif "success" in query:
            return random.uniform(99.2, 99.8)
        elif "freshness" in query:
            return random.uniform(97, 99.5)
        elif "accuracy" in query:
            return random.uniform(88, 95)
        else:
            return random.uniform(90, 99)
    
    def _update_slo_metrics(self, slo_name: str, measurement: SLOMeasurement):
        """Updates Prometheus metrics for the SLO."""
        slo_def = self.slo_definitions[slo_name]
        status_value = {
            SLOStatus.HEALTHY: 0,
            SLOStatus.WARNING: 1,
            SLOStatus.CRITICAL: 2,
            SLOStatus.BREACHED: 3
        }[measurement.status]
        
        self.slo_status_gauge.labels(slo_name=slo_name, team=slo_def.team_responsible).set(status_value)
        self.error_budget_gauge.labels(slo_name=slo_name, team=slo_def.team_responsible).set(measurement.error_budget_remaining)
        self.slo_compliance_counter.labels(slo_name=slo_name, status=measurement.status.value).inc()
    
    async def measure_business_kpi(self, kpi_name: str, value: float):
        """Records a business KPI measurement."""
        if kpi_name not in self.business_kpis:
            raise ValueError(f"Business KPI {kpi_name} is not defined")
        
        kpi_def = self.business_kpis[kpi_name]
        self.business_kpi_gauge.labels(kpi_name=kpi_name, team=kpi_def.team_responsible, criticality=kpi_def.business_criticality).set(value)
        
        target_met = self._check_kpi_target(kpi_def, value)
        logger.info(
            "Business KPI measured",
            kpi_name=kpi_name,
            value=value,
            target_value=kpi_def.target_value,
            unit=kpi_def.unit,
            target_met=target_met
        )
        
        if not target_met:
            await self._alert_kpi_miss(kpi_def, value)
    
    def _check_kpi_target(self, kpi_def: BusinessKPI, value: float) -> bool:
        """Checks whether a KPI meets its target."""
        if kpi_def.name in ["false_positive_rate", "investigation_efficiency", "system_cost_per_transaction"]:
            return value <= kpi_def.target_value
        else:
            return value >= kpi_def.target_value
    
    async def _alert_kpi_miss(self, kpi_def: BusinessKPI, actual_value: float):
        """Sends an alert when a KPI misses its target."""
        logger.warning(
            "Business KPI target missed",
            kpi_name=kpi_def.name,
            actual_value=actual_value,
            target_value=kpi_def.target_value,
            team=kpi_def.team_responsible
        )


async def main():
    """Main loop simulating SLA and KPI monitoring."""
    class DummyCollector:
        registry = None
    
    monitor = SLAMonitor(metrics_collector=DummyCollector())
    
    while True:
        for slo_name in monitor.slo_definitions.keys():
            await monitor.measure_slo(slo_name)
        
        # Simulate KPI measurements
        for kpi_name in monitor.business_kpis.keys():
            value = np.random.uniform(90, 105)
            await monitor.measure_business_kpi(kpi_name, value)
        
        await asyncio.sleep(300)  # Wait 5 minutes


if __name__ == "__main__":
    asyncio.run(main())
