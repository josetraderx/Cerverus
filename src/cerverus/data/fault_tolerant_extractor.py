"""
Cerverus System - Stage 1: Data Collection
Circuit Breaker Pattern and Fault Tolerant Data Extractor
Following documentation requirements exactly as specified
"""

import time
import asyncio
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import json
import random
from pathlib import Path

# Import our data source adapters
from .extraction import (
    DataSourceAdapter, 
    DataExtractionResult,
    YahooFinanceAdapter,
    SECEdgarAdapter, 
    FINRAAdapter,
    AlphaVantageAdapter
)

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states as per documentation."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery

@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring as specified in documentation."""
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    state_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_failure(self):
        """Record a failure."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
    
    def add_success(self):
        """Record a success."""
        self.success_count += 1
        self.last_success_time = datetime.now()
    
    def reset_failures(self):
        """Reset failure count."""
        self.failure_count = 0
    
    def record_state_change(self, from_state: CircuitBreakerState, 
                           to_state: CircuitBreakerState, reason: str):
        """Record state change for monitoring."""
        self.state_changes.append({
            "timestamp": datetime.now().isoformat(),
            "from_state": from_state.value,
            "to_state": to_state.value,
            "reason": reason
        })

class CircuitBreaker:
    """
    Circuit Breaker implementation for data source resilience.
    Implements states: Closed (normal) -> Open (failing) -> Half-Open (testing)
    As specified in Stage 1 documentation.
    """
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
    
    async def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._move_to_half_open()
            else:
                raise Exception("Circuit breaker is OPEN - failing fast")
        
        try:
            result = await self._execute_function(func, *args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    async def _execute_function(self, func: Callable, *args, **kwargs):
        """Execute the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.metrics.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.metrics.last_failure_time
        return time_since_failure.total_seconds() >= self.recovery_timeout
    
    def _move_to_half_open(self):
        """Move circuit breaker to half-open state."""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.metrics.record_state_change(
            old_state, 
            self.state, 
            "Recovery timeout reached, testing service"
        )
        self.logger.info("Circuit breaker moved to HALF-OPEN state")
    
    def _on_success(self):
        """Handle successful function execution."""
        self.metrics.add_success()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._move_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success in closed state
            self.metrics.reset_failures()
    
    def _on_failure(self):
        """Handle failed function execution."""
        self.metrics.add_failure()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._move_to_open()
        elif (self.state == CircuitBreakerState.CLOSED and 
              self.metrics.failure_count >= self.failure_threshold):
            self._move_to_open()
    
    def _move_to_closed(self):
        """Move circuit breaker to closed state."""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        self.metrics.record_state_change(
            old_state, 
            self.state, 
            "Service recovered, normal operation resumed"
        )
        self.metrics.reset_failures()
        self.logger.info("Circuit breaker moved to CLOSED state")
    
    def _move_to_open(self):
        """Move circuit breaker to open state."""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.metrics.record_state_change(
            old_state, 
            self.state, 
            f"Failure threshold reached: {self.metrics.failure_count}/{self.failure_threshold}"
        )
        self.logger.error(f"Circuit breaker moved to OPEN state - failing fast")
    
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self.state
    
    def get_metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self.metrics


class RetryStrategy:
    """
    Retry strategy with exponential backoff and jitter.
    Implements Pattern Strategy for Rate Limiting as per documentation.
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logging.getLogger(f"{__name__}.RetryStrategy")
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic and exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                if attempt > 0:
                    self.logger.info(f"Function succeeded on attempt {attempt + 1}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    self.logger.error(f"Function failed after {self.max_retries + 1} attempts")
                    break
                
                # Distinguish between recoverable and non-recoverable errors
                if not self._is_recoverable_error(e):
                    self.logger.error(f"Non-recoverable error: {str(e)}")
                    break
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f}s"
                )
                
                await asyncio.sleep(delay)
        
        raise last_exception
    
    def _is_recoverable_error(self, error: Exception) -> bool:
        """Distinguish between recoverable and non-recoverable errors as per documentation."""
        # Rate limiting errors are recoverable
        if "rate limit" in str(error).lower():
            return True
        
        # Connection errors are recoverable
        if any(error_type in str(type(error)) for error_type in 
               ['ConnectionError', 'TimeoutError', 'HTTPError']):
            return True
        
        # Server errors (5xx) are recoverable
        if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
            return 500 <= error.response.status_code < 600
        
        # Authentication errors are not recoverable
        if any(auth_error in str(error).lower() for auth_error in 
               ['unauthorized', 'forbidden', 'invalid api key']):
            return False
        
        # Default to recoverable for unknown errors
        return True
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and optional jitter."""
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter (±25% of delay) to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get retry strategy metrics."""
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "exponential_base": self.exponential_base,
            "jitter_enabled": self.jitter
        }


class RateLimitStrategy:
    """
    Rate limiting strategy implementation.
    Implements Pattern Strategy for Rate Limiting as per documentation.
    """
    
    def __init__(self, algorithm: str = "token_bucket"):
        self.algorithm = algorithm
        self.algorithms = {
            "token_bucket": self._token_bucket,
            "sliding_window": self._sliding_window
        }
        self.logger = logging.getLogger(f"{__name__}.RateLimitStrategy")
    
    def apply_rate_limit(self, adapter: DataSourceAdapter) -> bool:
        """Apply rate limiting algorithm to adapter."""
        if self.algorithm in self.algorithms:
            return self.algorithms[self.algorithm](adapter)
        else:
            self.logger.warning(f"Unknown rate limiting algorithm: {self.algorithm}")
            return True
    
    def _token_bucket(self, adapter: DataSourceAdapter) -> bool:
        """Token Bucket Algorithm implementation."""
        rate_limit = adapter.get_rate_limits()
        
        # Simplified token bucket - in production would maintain token state
        current_time = time.time()
        time_since_last = current_time - adapter.last_request_time
        
        # Replenish tokens based on time passed
        tokens_to_add = time_since_last * (rate_limit.requests_per_minute / 60.0)
        available_tokens = min(rate_limit.burst_limit, tokens_to_add)
        
        return available_tokens >= 1.0
    
    def _sliding_window(self, adapter: DataSourceAdapter) -> bool:
        """Sliding Window Algorithm implementation."""
        # Simplified sliding window implementation
        current_time = time.time()
        window_start = current_time - 60  # 1-minute window
        
        # In production, would maintain a queue of request timestamps
        # For now, use the basic rate limiting from adapter
        return adapter._check_rate_limit()


class FaultTolerantDataExtractor:
    """
    Fault-tolerant data extractor with circuit breaker pattern.
    Orchestrates multiple data source adapters with resilience patterns.
    Implements FaultTolerantDataExtractor as specified in documentation.
    """
    
    def __init__(self, alpha_vantage_api_key: Optional[str] = None):
        self.adapters: Dict[str, DataSourceAdapter] = {
            "yahoo_finance": YahooFinanceAdapter(),
            "sec_edgar": SECEdgarAdapter(),
            "finra": FINRAAdapter(),
        }
        
        # Add Alpha Vantage only if API key is provided
        if alpha_vantage_api_key:
            self.adapters["alpha_vantage"] = AlphaVantageAdapter(alpha_vantage_api_key)
        
        # Circuit breakers per adapter with configuration from documentation
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            name: CircuitBreaker(
                failure_threshold=5,  # As specified in documentation
                recovery_timeout=60,  # As specified in documentation
                expected_exception=Exception
            )
            for name in self.adapters.keys()
        }
        
        # Retry strategies per adapter
        self.retry_strategies: Dict[str, RetryStrategy] = {
            name: RetryStrategy(
                max_retries=3,
                base_delay=1.0,
                max_delay=30.0
            )
            for name in self.adapters.keys()
        }
        
        # Rate limiting strategies
        self.rate_limit_strategies: Dict[str, RateLimitStrategy] = {
            name: RateLimitStrategy(algorithm="token_bucket")
            for name in self.adapters.keys()
        }
        
        self.logger = logging.getLogger(f"{__name__}.FaultTolerantDataExtractor")
        self.extraction_history: List[Dict[str, Any]] = []
    
    async def extract_from_all_sources(self, 
                                     symbols: List[str], 
                                     start_date: datetime, 
                                     end_date: datetime) -> Dict[str, DataExtractionResult]:
        """
        Extract data from all available sources with fault tolerance.
        Returns results from successful extractions, logs failures.
        """
        extraction_id = f"extraction_{int(time.time())}"
        self.logger.info(f"Starting fault-tolerant extraction {extraction_id}")
        
        results = {}
        extraction_summary = {
            "extraction_id": extraction_id,
            "timestamp": datetime.now().isoformat(),
            "symbols": symbols,
            "sources_attempted": list(self.adapters.keys()),
            "sources_successful": [],
            "sources_failed": [],
            "total_records": 0,
            "circuit_breaker_states": {}
        }
        
        # Extract from each source concurrently
        tasks = []
        for source_name, adapter in self.adapters.items():
            task = asyncio.create_task(
                self._extract_from_source_with_resilience(
                    source_name, adapter, symbols, start_date, end_date
                )
            )
            tasks.append((source_name, task))
        
        # Wait for all extractions to complete
        for source_name, task in tasks:
            try:
                result = await task
                results[source_name] = result
                
                # Record circuit breaker state
                cb_state = self.circuit_breakers[source_name].get_state()
                extraction_summary["circuit_breaker_states"][source_name] = cb_state.value
                
                if result.success:
                    extraction_summary["sources_successful"].append(source_name)
                    extraction_summary["total_records"] += result.records_count
                else:
                    extraction_summary["sources_failed"].append(source_name)
                    
            except Exception as e:
                self.logger.error(f"Unexpected error in {source_name}: {str(e)}")
                extraction_summary["sources_failed"].append(source_name)
        
        # Log extraction summary with detailed metrics
        self.extraction_history.append(extraction_summary)
        self._log_extraction_summary(extraction_summary)
        
        return results
    
    async def _extract_from_source_with_resilience(self,
                                                 source_name: str,
                                                 adapter: DataSourceAdapter,
                                                 symbols: List[str],
                                                 start_date: datetime,
                                                 end_date: datetime) -> DataExtractionResult:
        """Extract from single source with circuit breaker and retry."""
        circuit_breaker = self.circuit_breakers[source_name]
        retry_strategy = self.retry_strategies[source_name]
        rate_limit_strategy = self.rate_limit_strategies[source_name]
        
        try:
            # Check rate limiting before attempting extraction
            if not rate_limit_strategy.apply_rate_limit(adapter):
                return DataExtractionResult(
                    success=False,
                    data=None,
                    records_count=0,
                    source=source_name,
                    timestamp=datetime.now(),
                    s3_path=None,
                    error_message="Rate limit exceeded"
                )
            
            # Wrap extraction with circuit breaker and retry
            result = await circuit_breaker.call(
                retry_strategy.execute_with_retry,
                adapter.extract_data,
                symbols,
                start_date,
                end_date
            )
            return result
            
        except Exception as e:
            # Create failed result with detailed error information
            return DataExtractionResult(
                success=False,
                data=None,
                records_count=0,
                source=source_name,
                timestamp=datetime.now(),
                s3_path=None,
                error_message=f"Extraction failed after all retries: {str(e)}"
            )
    
    def _log_extraction_summary(self, summary: Dict[str, Any]):
        """Log extraction summary with structured format and circuit breaker states."""
        success_rate = len(summary["sources_successful"]) / len(summary["sources_attempted"]) * 100
        
        self.logger.info(
            "Extraction completed",
            extra={
                "extraction_id": summary["extraction_id"],
                "success_rate": f"{success_rate:.1f}%",
                "successful_sources": summary["sources_successful"],
                "failed_sources": summary["sources_failed"],
                "total_records": summary["total_records"],
                "circuit_breaker_states": summary["circuit_breaker_states"]
            }
        )
    
    async def validate_all_connections(self) -> Dict[str, bool]:
        """Validate connections to all data sources."""
        self.logger.info("Validating connections to all data sources")
        
        validation_results = {}
        tasks = []
        
        for source_name, adapter in self.adapters.items():
            task = asyncio.create_task(
                self._validate_connection_with_timeout(source_name, adapter)
            )
            tasks.append((source_name, task))
        
        for source_name, task in tasks:
            try:
                is_valid = await asyncio.wait_for(task, timeout=10.0)
                validation_results[source_name] = is_valid
            except asyncio.TimeoutError:
                validation_results[source_name] = False
                self.logger.warning(f"Connection validation timeout for {source_name}")
        
        return validation_results
    
    async def _validate_connection_with_timeout(self, 
                                              source_name: str, 
                                              adapter: DataSourceAdapter) -> bool:
        """Validate connection with timeout handling."""
        try:
            return adapter.validate_connection()
        except Exception as e:
            self.logger.error(f"Connection validation failed for {source_name}: {str(e)}")
            return False
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers with detailed metrics."""
        status = {}
        for source_name, cb in self.circuit_breakers.items():
            metrics = cb.get_metrics()
            status[source_name] = {
                "state": cb.get_state().value,
                "failure_count": metrics.failure_count,
                "success_count": metrics.success_count,
                "last_failure": metrics.last_failure_time.isoformat() if metrics.last_failure_time else None,
                "last_success": metrics.last_success_time.isoformat() if metrics.last_success_time else None,
                "failure_threshold": cb.failure_threshold,
                "recovery_timeout": cb.recovery_timeout,
                "state_changes": metrics.state_changes[-5:]  # Last 5 state changes
            }
        return status
    
    def get_extraction_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent extraction history."""
        return self.extraction_history[-limit:]
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check of the extraction system.
        Returns detailed status as per documentation monitoring requirements.
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "data_sources": {},
            "circuit_breakers": {},
            "recent_extractions": len(self.extraction_history),
            "system_metrics": {
                "total_adapters": len(self.adapters),
                "active_adapters": 0,
                "failed_adapters": 0
            }
        }
        
        # Check connections
        connection_results = await self.validate_all_connections()
        
        # Check circuit breakers
        cb_status = self.get_circuit_breaker_status()
        
        unhealthy_sources = []
        
        for source_name in self.adapters.keys():
            source_health = {
                "connection": connection_results.get(source_name, False),
                "circuit_breaker_state": cb_status[source_name]["state"],
                "recent_failures": cb_status[source_name]["failure_count"],
                "last_success": cb_status[source_name]["last_success"],
                "rate_limit_algorithm": self.rate_limit_strategies[source_name].algorithm
            }
            
            # Determine if source is healthy
            is_healthy = (
                source_health["connection"] and 
                source_health["circuit_breaker_state"] != "open" and
                source_health["recent_failures"] < 3
            )
            
            source_health["status"] = "healthy" if is_healthy else "unhealthy"
            
            if is_healthy:
                health_status["system_metrics"]["active_adapters"] += 1
            else:
                health_status["system_metrics"]["failed_adapters"] += 1
                unhealthy_sources.append(source_name)
            
            health_status["data_sources"][source_name] = source_health
            health_status["circuit_breakers"][source_name] = cb_status[source_name]
        
        # Overall system status
        if unhealthy_sources:
            if len(unhealthy_sources) == len(self.adapters):
                health_status["overall_status"] = "critical"
            else:
                health_status["overall_status"] = "degraded"
        
        return health_status