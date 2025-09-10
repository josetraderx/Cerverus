"""
Cerverus System - Stage 1: Data Collection
Dead Letter Queue and Structured Logging System
Following documentation requirements for error handling and forensic analysis
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import logging.handlers
from pathlib import Path
import gzip
import pickle
from collections import deque, defaultdict
import threading
import asyncio
from kafka import KafkaProducer, KafkaConsumer, KafkaAdminClient
from kafka.admin import ConfigResource, ConfigResourceType, NewTopic
import structlog

from .extraction import DataExtractionResult

class ErrorCategory(Enum):
    """Error categorization for automatic processing."""
    TEMPORARY_FAILURE = "temporary_failure"
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    DATA_QUALITY = "data_quality"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"

class ErrorSeverity(Enum):
    """Error severity levels for prioritization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class DLQMessage:
    """Dead Letter Queue message structure."""
    message_id: str
    timestamp: datetime
    source: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    original_data: Any
    error_message: str
    error_details: Dict[str, Any]
    retry_count: int = 0
    max_retries: int = 3
    correlation_id: Optional[str] = None
    processing_context: Optional[Dict[str, Any]] = None

class DeadLetterQueue:
    """
    Dead Letter Queue implementation with Apache Kafka.
    Provides forensic analysis and automatic retry capabilities as per documentation.
    """
    
    def __init__(self, 
                 kafka_bootstrap_servers: str = "localhost:9092",
                 dlq_topic: str = "cerverus-dlq",
                 retry_topic: str = "cerverus-retry"):
        
        self.kafka_servers = kafka_bootstrap_servers
        self.dlq_topic = dlq_topic
        self.retry_topic = retry_topic
        
        # Initialize Kafka components
        self._initialize_kafka()
        
        # Message storage and categorization
        self.message_categories = defaultdict(list)
        self.error_patterns = defaultdict(int)
        self.retry_queue = deque()
        
        self.logger = structlog.get_logger("cerverus.dlq")
        
        # Background processing
        self._processing_thread = None
        self._stop_processing = threading.Event()
        self._start_background_processing()
    
    def _initialize_kafka(self):
        """Initialize Kafka producer, consumer, and topics."""
        try:
            # Initialize Kafka producer
            self.producer = KafkaProducer(
                bootstrap_servers=self.kafka_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                acks='all',
                retries=3,
                retry_backoff_ms=1000
            )
            
            # Initialize admin client for topic management
            self.admin_client = KafkaAdminClient(
                bootstrap_servers=self.kafka_servers,
                client_id='cerverus_dlq_admin'
            )
            
            # Create topics if they don't exist
            self._create_topics()
            
            # Initialize consumer for retry processing
            self.consumer = KafkaConsumer(
                self.retry_topic,
                bootstrap_servers=self.kafka_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                group_id='cerverus_dlq_processor',
                enable_auto_commit=True,
                auto_offset_reset='latest'
            )
            
            self.logger.info("Kafka DLQ system initialized successfully")
            
        except Exception as e:
            self.logger.error("Failed to initialize Kafka DLQ", error=str(e))
            # Fallback to in-memory implementation
            self._initialize_memory_fallback()
    
    def _initialize_memory_fallback(self):
        """Initialize memory-based fallback for development."""
        class MockKafkaProducer:
            def __init__(self):
                self.messages = []
            
            def send(self, topic, value=None, key=None):
                self.messages.append({"topic": topic, "key": key, "value": value})
                return self
            
            def get(self, timeout=None):
                return self
            
            def flush(self):
                pass
        
        class MockKafkaConsumer:
            def __init__(self, *topics, **kwargs):
                self.messages = []
            
            def poll(self, timeout_ms=1000):
                return {}
        
        self.producer = MockKafkaProducer()
        self.consumer = MockKafkaConsumer()
        self.admin_client = None
        
        self.logger.warning("Using in-memory DLQ fallback - configure Kafka for production")
    
    def _create_topics(self):
        """Create Kafka topics for DLQ and retry processing."""
        if not self.admin_client:
            return
        
        try:
            topics = [
                NewTopic(
                    name=self.dlq_topic,
                    num_partitions=3,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '604800000',  # 7 days
                        'compression.type': 'gzip'
                    }
                ),
                NewTopic(
                    name=self.retry_topic,
                    num_partitions=3,
                    replication_factor=1,
                    topic_configs={
                        'retention.ms': '86400000',  # 24 hours
                        'compression.type': 'gzip'
                    }
                )
            ]
            
            self.admin_client.create_topics(topics, validate_only=False)
            self.logger.info("DLQ topics created successfully")
            
        except Exception as e:
            self.logger.warning("Failed to create DLQ topics", error=str(e))
    
    def send_to_dlq(self, 
                   source: str,
                   original_data: Any,
                   error_message: str,
                   error_category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                   error_severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                   correlation_id: Optional[str] = None,
                   processing_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Send failed message to Dead Letter Queue for forensic analysis.
        Implements categorization and correlation as per documentation.
        """
        
        message_id = str(uuid.uuid4())
        
        dlq_message = DLQMessage(
            message_id=message_id,
            timestamp=datetime.now(),
            source=source,
            error_category=error_category,
            error_severity=error_severity,
            original_data=original_data,
            error_message=error_message,
            error_details=self._extract_error_details(error_message, processing_context),
            correlation_id=correlation_id,
            processing_context=processing_context
        )
        
        try:
            # Send to Kafka DLQ topic
            message_dict = asdict(dlq_message)
            message_dict['timestamp'] = dlq_message.timestamp.isoformat()
            
            future = self.producer.send(
                self.dlq_topic,
                value=message_dict,
                key=source
            )
            
            # Wait for confirmation
            future.get(timeout=10)
            
            # Update internal tracking
            self.message_categories[error_category].append(dlq_message)
            self.error_patterns[error_category] += 1
            
            # Determine if this should be retried
            if self._should_retry(dlq_message):
                self._schedule_retry(dlq_message)
            
            self.logger.info(
                "Message sent to DLQ",
                message_id=message_id,
                source=source,
                error_category=error_category.value,
                error_severity=error_severity.value
            )
            
            return message_id
            
        except Exception as e:
            self.logger.error(
                "Failed to send message to DLQ",
                error=str(e),
                source=source,
                original_error=error_message
            )
            return None
    
    def _extract_error_details(self, error_message: str, 
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract structured error details for analysis."""
        details = {
            "timestamp": datetime.now().isoformat(),
            "error_hash": hash(error_message),
            "context": context or {}
        }
        
        # Extract specific error patterns
        error_lower = error_message.lower()
        
        if "rate limit" in error_lower or "429" in error_message:
            details["error_type"] = "rate_limit"
            details["retry_recommended"] = True
        elif "timeout" in error_lower or "connection" in error_lower:
            details["error_type"] = "network_error"
            details["retry_recommended"] = True
        elif "authentication" in error_lower or "401" in error_message or "403" in error_message:
            details["error_type"] = "authentication_error"
            details["retry_recommended"] = False
        elif "validation" in error_lower or "schema" in error_lower:
            details["error_type"] = "validation_error"
            details["retry_recommended"] = False
        else:
            details["error_type"] = "unknown"
            details["retry_recommended"] = True
        
        return details
    
    def _should_retry(self, dlq_message: DLQMessage) -> bool:
        """Determine if message should be automatically retried."""
        # Don't retry if max retries exceeded
        if dlq_message.retry_count >= dlq_message.max_retries:
            return False
        
        # Don't retry authentication errors
        if dlq_message.error_category == ErrorCategory.AUTHENTICATION:
            return False
        
        # Don't retry validation errors
        if dlq_message.error_category == ErrorCategory.VALIDATION_ERROR:
            return False
        
        # Retry temporary failures and rate limits
        retry_categories = [
            ErrorCategory.TEMPORARY_FAILURE,
            ErrorCategory.RATE_LIMIT,
            ErrorCategory.NETWORK_ERROR
        ]
        
        return dlq_message.error_category in retry_categories
    
    def _schedule_retry(self, dlq_message: DLQMessage):
        """Schedule message for automatic retry."""
        # Calculate retry delay with exponential backoff
        base_delay = 60  # 1 minute base delay
        delay_seconds = base_delay * (2 ** dlq_message.retry_count)
        max_delay = 3600  # Maximum 1 hour delay
        delay_seconds = min(delay_seconds, max_delay)
        
        retry_time = datetime.now() + timedelta(seconds=delay_seconds)
        
        retry_message = {
            "original_message": asdict(dlq_message),
            "retry_time": retry_time.isoformat(),
            "retry_delay_seconds": delay_seconds
        }
        
        try:
            # Send to retry topic
            self.producer.send(
                self.retry_topic,
                value=retry_message,
                key=dlq_message.source
            )
            
            self.logger.info(
                "Message scheduled for retry",
                message_id=dlq_message.message_id,
                retry_count=dlq_message.retry_count + 1,
                retry_delay_seconds=delay_seconds
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to schedule retry",
                message_id=dlq_message.message_id,
                error=str(e)
            )
    
    def _start_background_processing(self):
        """Start background thread for retry processing."""
        def process_retries():
            while not self._stop_processing.is_set():
                try:
                    # Poll for retry messages
                    message_batch = self.consumer.poll(timeout_ms=1000)
                    
                    for topic_partition, messages in message_batch.items():
                        for message in messages:
                            self._process_retry_message(message.value)
                    
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error("Error in retry processing", error=str(e))
                    time.sleep(5)
        
        self._processing_thread = threading.Thread(target=process_retries, daemon=True)
        self._processing_thread.start()
    
    def _process_retry_message(self, retry_message: Dict[str, Any]):
        """Process a retry message from the queue."""
        try:
            retry_time = datetime.fromisoformat(retry_message["retry_time"])
            
            # Check if it's time to retry
            if datetime.now() < retry_time:
                return
            
            original_msg_dict = retry_message["original_message"]
            
            # Reconstruct DLQ message
            dlq_message = DLQMessage(
                message_id=original_msg_dict["message_id"],
                timestamp=datetime.fromisoformat(original_msg_dict["timestamp"]),
                source=original_msg_dict["source"],
                error_category=ErrorCategory(original_msg_dict["error_category"]),
                error_severity=ErrorSeverity(original_msg_dict["error_severity"]),
                original_data=original_msg_dict["original_data"],
                error_message=original_msg_dict["error_message"],
                error_details=original_msg_dict["error_details"],
                retry_count=original_msg_dict["retry_count"] + 1,
                max_retries=original_msg_dict["max_retries"],
                correlation_id=original_msg_dict.get("correlation_id"),
                processing_context=original_msg_dict.get("processing_context")
            )
            
            # TODO: Implement actual retry logic here
            # This would involve calling the original extraction function again
            
            self.logger.info(
                "Processing retry message",
                message_id=dlq_message.message_id,
                retry_count=dlq_message.retry_count
            )
            
        except Exception as e:
            self.logger.error("Failed to process retry message", error=str(e))
    
    def get_dlq_analytics(self) -> Dict[str, Any]:
        """Get analytics and patterns from DLQ for forensic analysis."""
        total_messages = sum(len(messages) for messages in self.message_categories.values())
        
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "total_messages": total_messages,
            "messages_by_category": {
                category.value: len(messages) 
                for category, messages in self.message_categories.items()
            },
            "error_patterns": {
                category.value: count 
                for category, count in self.error_patterns.items()
            },
            "top_error_sources": self._get_top_error_sources(),
            "recommendations": self._generate_dlq_recommendations()
        }
        
        return analytics
    
    def _get_top_error_sources(self) -> List[Dict[str, Any]]:
        """Get top sources generating errors."""
        source_counts = defaultdict(int)
        
        for messages in self.message_categories.values():
            for message in messages:
                source_counts[message.source] += 1
        
        # Sort by count and return top 10
        sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"source": source, "error_count": count} 
            for source, count in sorted_sources[:10]
        ]
    
    def _generate_dlq_recommendations(self) -> List[str]:
        """Generate recommendations based on DLQ analysis."""
        recommendations = []
        
        # Check for high rate limit errors
        rate_limit_count = self.error_patterns.get(ErrorCategory.RATE_LIMIT, 0)
        if rate_limit_count > 10:
            recommendations.append(
                f"High rate limit errors ({rate_limit_count}) - consider implementing "
                "more aggressive rate limiting"
            )
        
        # Check for authentication errors
        auth_error_count = self.error_patterns.get(ErrorCategory.AUTHENTICATION, 0)
        if auth_error_count > 5:
            recommendations.append(
                f"Authentication errors detected ({auth_error_count}) - "
                "review API credentials and access tokens"
            )
        
        # Check for validation errors
        validation_error_count = self.error_patterns.get(ErrorCategory.VALIDATION_ERROR, 0)
        if validation_error_count > 15:
            recommendations.append(
                f"High validation error rate ({validation_error_count}) - "
                "review data extraction and validation logic"
            )
        
        if not recommendations:
            recommendations.append("DLQ error rates are within acceptable limits")
        
        return recommendations
    
    def cleanup_old_messages(self, days_old: int = 7) -> int:
        """Clean up old DLQ messages."""
        cutoff_time = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0
        
        for category in list(self.message_categories.keys()):
            messages = self.message_categories[category]
            self.message_categories[category] = [
                msg for msg in messages if msg.timestamp > cutoff_time
            ]
            cleaned_count += len(messages) - len(self.message_categories[category])
        
        self.logger.info(f"Cleaned up {cleaned_count} old DLQ messages")
        return cleaned_count
    
    def shutdown(self):
        """Gracefully shutdown DLQ processing."""
        self._stop_processing.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5)
        
        if hasattr(self.producer, 'close'):
            self.producer.close()
        if hasattr(self.consumer, 'close'):
            self.consumer.close()


class StructuredLogger:
    """
    Structured logging implementation with correlation IDs and detailed context.
    Implements structured logging as specified in documentation.
    """
    
    def __init__(self, 
                 service_name: str = "cerverus",
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 enable_json: bool = True):
        
        self.service_name = service_name
        self.log_level = getattr(logging, log_level.upper())
        
        # Configure structlog
        self._configure_structlog(enable_json)
        
        # Setup file logging if specified
        if log_file:
            self._setup_file_logging(log_file)
        
        self.logger = structlog.get_logger(service_name)
    
    def _configure_structlog(self, enable_json: bool):
        """Configure structlog with appropriate processors."""
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder()
        ]
        
        if enable_json:
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=None,
            level=self.log_level,
        )
    
    def _setup_file_logging(self, log_file: str):
        """Setup file logging with rotation."""
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5
        )
        
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
    
    def log_extraction_start(self, 
                           source: str,
                           symbols: List[str],
                           correlation_id: str,
                           start_date: datetime,
                           end_date: datetime):
        """Log extraction start with correlation ID."""
        self.logger.info(
            "Data extraction started",
            event_type="extraction_start",
            source=source,
            symbols=symbols,
            symbol_count=len(symbols),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def log_extraction_success(self, 
                             result: DataExtractionResult,
                             correlation_id: str,
                             processing_time_ms: float):
        """Log successful extraction with detailed metrics."""
        self.logger.info(
            "Data extraction completed successfully",
            event_type="extraction_success",
            source=result.source,
            records_count=result.records_count,
            s3_path=result.s3_path,
            data_quality_score=result.data_quality_score,
            processing_time_ms=processing_time_ms,
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def log_extraction_failure(self,
                             source: str,
                             error_message: str,
                             correlation_id: str,
                             processing_time_ms: float,
                             error_category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR):
        """Log extraction failure with error details."""
        self.logger.error(
            "Data extraction failed",
            event_type="extraction_failure",
            source=source,
            error_message=error_message,
            error_category=error_category.value,
            processing_time_ms=processing_time_ms,
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def log_validation_result(self,
                            source: str,
                            total_validations: int,
                            passed_validations: int,
                            overall_score: float,
                            correlation_id: str):
        """Log validation results with quality metrics."""
        self.logger.info(
            "Data validation completed",
            event_type="validation_complete",
            source=source,
            total_validations=total_validations,
            passed_validations=passed_validations,
            failed_validations=total_validations - passed_validations,
            pass_rate=passed_validations / total_validations if total_validations > 0 else 0,
            overall_quality_score=overall_score,
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def log_circuit_breaker_state_change(self,
                                       source: str,
                                       from_state: str,
                                       to_state: str,
                                       reason: str,
                                       correlation_id: str):
        """Log circuit breaker state changes."""
        self.logger.warning(
            "Circuit breaker state changed",
            event_type="circuit_breaker_state_change",
            source=source,
            from_state=from_state,
            to_state=to_state,
            reason=reason,
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def log_cache_metrics(self,
                        cache_level: str,
                        hit_count: int,
                        miss_count: int,
                        hit_ratio: float,
                        correlation_id: str):
        """Log cache performance metrics."""
        self.logger.info(
            "Cache metrics update",
            event_type="cache_metrics",
            cache_level=cache_level,
            hit_count=hit_count,
            miss_count=miss_count,
            total_requests=hit_count + miss_count,
            hit_ratio=hit_ratio,
            miss_ratio=1.0 - hit_ratio,
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def log_dlq_message(self,
                       message_id: str,
                       source: str,
                       error_category: ErrorCategory,
                       error_severity: ErrorSeverity,
                       correlation_id: str):
        """Log DLQ message creation."""
        self.logger.warning(
            "Message sent to Dead Letter Queue",
            event_type="dlq_message",
            dlq_message_id=message_id,
            source=source,
            error_category=error_category.value,
            error_severity=error_severity.value,
            correlation_id=correlation_id,
            service=self.service_name
        )
    
    def create_correlation_id(self) -> str:
        """Create a new correlation ID for request tracking."""
        return str(uuid.uuid4())


class ErrorHandler:
    """
    Centralized error handling system integrating DLQ and structured logging.
    Provides unified interface for error management across the system.
    """
    
    def __init__(self, dlq: DeadLetterQueue, logger: StructuredLogger):
        self.dlq = dlq
        self.logger = logger
        self.error_stats = defaultdict(int)
    
    def handle_extraction_error(self,
                              source: str,
                              original_data: Any,
                              error: Exception,
                              correlation_id: str,
                              processing_context: Optional[Dict[str, Any]] = None) -> str:
        """Handle extraction errors with categorization and DLQ routing."""
        
        # Categorize the error
        error_category = self._categorize_error(error)
        error_severity = self._determine_severity(error_category, str(error))
        
        # Log the error
        self.logger.log_extraction_failure(
            source=source,
            error_message=str(error),
            correlation_id=correlation_id,
            processing_time_ms=0,  # Would be provided by caller
            error_category=error_category
        )
        
        # Send to DLQ
        dlq_message_id = self.dlq.send_to_dlq(
            source=source,
            original_data=original_data,
            error_message=str(error),
            error_category=error_category,
            error_severity=error_severity,
            correlation_id=correlation_id,
            processing_context=processing_context
        )
        
        # Update statistics
        self.error_stats[f"{source}_{error_category.value}"] += 1
        
        # Log DLQ routing
        if dlq_message_id:
            self.logger.log_dlq_message(
                message_id=dlq_message_id,
                source=source,
                error_category=error_category,
                error_severity=error_severity,
                correlation_id=correlation_id
            )
        
        return dlq_message_id
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Automatically categorize errors based on type and message."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Rate limiting errors
        if "rate limit" in error_str or "429" in error_str or "too many requests" in error_str:
            return ErrorCategory.RATE_LIMIT
        
        # Authentication errors
        if any(auth_term in error_str for auth_term in ["unauthorized", "forbidden", "401", "403", "api key", "authentication"]):
            return ErrorCategory.AUTHENTICATION
        
        # Network errors
        if any(net_term in error_type for net_term in ["connection", "timeout", "network"]) or \
           any(net_term in error_str for net_term in ["connection", "timeout", "network", "dns"]):
            return ErrorCategory.NETWORK_ERROR
        
        # Validation errors
        if any(val_term in error_str for val_term in ["validation", "schema", "invalid", "malformed"]):
            return ErrorCategory.VALIDATION_ERROR
        
        # Temporary failures
        if any(temp_term in error_str for temp_term in ["temporary", "retry", "service unavailable", "502", "503", "504"]):
            return ErrorCategory.TEMPORARY_FAILURE
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _determine_severity(self, category: ErrorCategory, error_message: str) -> ErrorSeverity:
        """Determine error severity based on category and message."""
        error_lower = error_message.lower()
        
        # Critical severity
        if category == ErrorCategory.AUTHENTICATION:
            return ErrorSeverity.CRITICAL
        
        if any(critical_term in error_lower for critical_term in ["critical", "fatal", "corrupted"]):
            return ErrorSeverity.CRITICAL
        
        # High severity
        if category == ErrorCategory.DATA_QUALITY:
            return ErrorSeverity.HIGH
        
        if any(high_term in error_lower for high_term in ["failed", "error", "exception"]):
            return ErrorSeverity.HIGH
        
        # Medium severity
        if category in [ErrorCategory.RATE_LIMIT, ErrorCategory.NETWORK_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity
        return ErrorSeverity.LOW
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        total_errors = sum(self.error_stats.values())
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_errors": total_errors,
            "error_breakdown": dict(self.error_stats),
            "dlq_analytics": self.dlq.get_dlq_analytics(),
            "top_error_patterns": self._get_top_error_patterns()
        }
    
    def _get_top_error_patterns(self) -> List[Dict[str, Any]]:
        """Get top error patterns for analysis."""
        sorted_errors = sorted(self.error_stats.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {
                "pattern": pattern,
                "count": count,
                "percentage": round((count / total_errors) * 100, 2) if total_errors > 0 else 0
            }
            for pattern, count in sorted_errors[:10]
        ]


class LogAggregator:
    """
    Log aggregation system with ELK Stack integration.
    Implements log aggregation as specified in documentation.
    """
    
    def __init__(self, 
                 elasticsearch_host: str = "localhost",
                 elasticsearch_port: int = 9200,
                 index_prefix: str = "cerverus-logs"):
        
        self.elasticsearch_host = elasticsearch_host
        self.elasticsearch_port = elasticsearch_port
        self.index_prefix = index_prefix
        
        # Initialize Elasticsearch client
        try:
            from elasticsearch import Elasticsearch
            self.es_client = Elasticsearch([
                f"http://{elasticsearch_host}:{elasticsearch_port}"
            ])
            
            # Test connection
            self.es_client.ping()
            self.elasticsearch_available = True
            
        except Exception:
            self.elasticsearch_available = False
            self.es_client = None
        
        self.logger = structlog.get_logger("cerverus.log_aggregator")
        
        # Setup index templates
        if self.elasticsearch_available:
            self._setup_index_templates()
    
    def _setup_index_templates(self):
        """Setup Elasticsearch index templates for structured logging."""
        template = {
            "index_patterns": [f"{self.index_prefix}-*"],
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 1,
                "refresh_interval": "5s"
            },
            "mappings": {
                "properties": {
                    "@timestamp": {"type": "date"},
                    "level": {"type": "keyword"},
                    "logger": {"type": "keyword"},
                    "message": {"type": "text"},
                    "event_type": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "correlation_id": {"type": "keyword"},
                    "service": {"type": "keyword"},
                    "processing_time_ms": {"type": "float"},
                    "error_category": {"type": "keyword"},
                    "error_severity": {"type": "keyword"},
                    "records_count": {"type": "integer"},
                    "data_quality_score": {"type": "float"}
                }
            }
        }
        
        try:
            self.es_client.indices.put_template(
                name=f"{self.index_prefix}-template",
                body=template
            )
            self.logger.info("Elasticsearch index template created")
        except Exception as e:
            self.logger.error("Failed to create index template", error=str(e))
    
    def send_log_to_elasticsearch(self, log_record: Dict[str, Any]):
        """Send structured log record to Elasticsearch."""
        if not self.elasticsearch_available:
            return
        
        try:
            # Generate index name with date
            index_date = datetime.now().strftime("%Y.%m.%d")
            index_name = f"{self.index_prefix}-{index_date}"
            
            # Add timestamp if not present
            if "@timestamp" not in log_record:
                log_record["@timestamp"] = datetime.now().isoformat()
            
            # Index the document
            self.es_client.index(
                index=index_name,
                body=log_record
            )
            
        except Exception as e:
            # Don't let logging errors break the main application
            pass
    
    def search_logs(self, 
                   query: str,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   source: Optional[str] = None,
                   event_type: Optional[str] = None,
                   size: int = 100) -> List[Dict[str, Any]]:
        """Search logs with filtering capabilities."""
        if not self.elasticsearch_available:
            return []
        
        try:
            # Build Elasticsearch query
            must_clauses = []
            
            if query:
                must_clauses.append({
                    "multi_match": {
                        "query": query,
                        "fields": ["message", "error_message", "source"]
                    }
                })
            
            if source:
                must_clauses.append({"term": {"source": source}})
            
            if event_type:
                must_clauses.append({"term": {"event_type": event_type}})
            
            if start_time or end_time:
                time_range = {}
                if start_time:
                    time_range["gte"] = start_time.isoformat()
                if end_time:
                    time_range["lte"] = end_time.isoformat()
                
                must_clauses.append({
                    "range": {"@timestamp": time_range}
                })
            
            search_body = {
                "query": {
                    "bool": {"must": must_clauses}
                } if must_clauses else {"match_all": {}},
                "sort": [{"@timestamp": {"order": "desc"}}],
                "size": size
            }
            
            # Execute search
            response = self.es_client.search(
                index=f"{self.index_prefix}-*",
                body=search_body
            )
            
            return [hit["_source"] for hit in response["hits"]["hits"]]
            
        except Exception as e:
            self.logger.error("Failed to search logs", error=str(e))
            return []


class ComprehensiveErrorHandler:
    """
    Comprehensive error handling system orchestrating all components.
    Provides unified interface for error management and forensic analysis.
    """
    
    def __init__(self,
                 kafka_servers: str = "localhost:9092",
                 elasticsearch_host: str = "localhost",
                 log_file: Optional[str] = None):
        
        # Initialize components
        self.dlq = DeadLetterQueue(kafka_servers)
        self.structured_logger = StructuredLogger(
            service_name="cerverus-stage1",
            log_file=log_file
        )
        self.error_handler = ErrorHandler(self.dlq, self.structured_logger)
        self.log_aggregator = LogAggregator(elasticsearch_host)
        
        # Hook structured logging to send to Elasticsearch
        self._setup_elasticsearch_logging()
        
        self.logger = self.structured_logger.logger
    
    def _setup_elasticsearch_logging(self):
        """Setup automatic forwarding of structured logs to Elasticsearch."""
        if not self.log_aggregator.elasticsearch_available:
            return
        
        # Create custom handler that forwards to Elasticsearch
        class ElasticsearchHandler(logging.Handler):
            def __init__(self, log_aggregator):
                super().__init__()
                self.log_aggregator = log_aggregator
            
            def emit(self, record):
                try:
                    if hasattr(record, 'msg') and isinstance(record.msg, dict):
                        self.log_aggregator.send_log_to_elasticsearch(record.msg)
                except Exception:
                    pass  # Don't let logging errors break the application
        
        # Add the handler to the root logger
        es_handler = ElasticsearchHandler(self.log_aggregator)
        logging.getLogger().addHandler(es_handler)
    
    def handle_extraction_pipeline_error(self,
                                       source: str,
                                       extraction_data: Any,
                                       error: Exception,
                                       processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors in the extraction pipeline with comprehensive logging and recovery.
        Returns information about error handling actions taken.
        """
        
        correlation_id = self.structured_logger.create_correlation_id()
        start_time = time.time()
        
        try:
            # Handle the error through centralized handler
            dlq_message_id = self.error_handler.handle_extraction_error(
                source=source,
                original_data=extraction_data,
                error=error,
                correlation_id=correlation_id,
                processing_context=processing_context
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create comprehensive error response
            error_response = {
                "correlation_id": correlation_id,
                "dlq_message_id": dlq_message_id,
                "error_handled": dlq_message_id is not None,
                "processing_time_ms": processing_time_ms,
                "error_category": self.error_handler._categorize_error(error).value,
                "error_severity": self.error_handler._determine_severity(
                    self.error_handler._categorize_error(error), 
                    str(error)
                ).value,
                "retry_recommended": dlq_message_id is not None,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log successful error handling
            self.logger.info(
                "Error handling completed",
                event_type="error_handling_complete",
                correlation_id=correlation_id,
                dlq_message_id=dlq_message_id,
                source=source,
                processing_time_ms=processing_time_ms
            )
            
            return error_response
            
        except Exception as handling_error:
            # Error in error handling - this is critical
            self.logger.critical(
                "Failed to handle extraction error",
                event_type="error_handling_failed",
                correlation_id=correlation_id,
                source=source,
                original_error=str(error),
                handling_error=str(handling_error)
            )
            
            return {
                "correlation_id": correlation_id,
                "error_handled": False,
                "critical_failure": True,
                "original_error": str(error),
                "handling_error": str(handling_error),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of error handling system."""
        return {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "dead_letter_queue": {
                    "status": "healthy" if hasattr(self.dlq.producer, 'send') else "degraded",
                    "analytics": self.dlq.get_dlq_analytics()
                },
                "structured_logging": {
                    "status": "healthy",
                    "service_name": self.structured_logger.service_name
                },
                "log_aggregation": {
                    "status": "healthy" if self.log_aggregator.elasticsearch_available else "degraded",
                    "elasticsearch_available": self.log_aggregator.elasticsearch_available
                }
            },
            "error_statistics": self.error_handler.get_error_statistics(),
            "overall_health": self._calculate_overall_health()
        }
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall health status of error handling system."""
        dlq_healthy = hasattr(self.dlq.producer, 'send')
        logging_healthy = True  # Structured logging is always available
        es_healthy = self.log_aggregator.elasticsearch_available
        
        if dlq_healthy and logging_healthy and es_healthy:
            return "healthy"
        elif dlq_healthy and logging_healthy:
            return "degraded"  # ES optional for core functionality
        else:
            return "critical"
    
    def search_error_logs(self,
                         error_pattern: str,
                         hours_back: int = 24,
                         source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search error logs for forensic analysis."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        return self.log_aggregator.search_logs(
            query=error_pattern,
            start_time=start_time,
            end_time=end_time,
            source=source,
            event_type="extraction_failure"
        )
    
    def generate_error_report(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate comprehensive error report for the specified period."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Get error statistics
        error_stats = self.error_handler.get_error_statistics()
        dlq_analytics = self.dlq.get_dlq_analytics()
        
        # Search for recent errors
        recent_errors = self.log_aggregator.search_logs(
            query="",
            start_time=start_time,
            end_time=end_time,
            event_type="extraction_failure",
            size=1000
        )
        
        # Analyze error trends
        error_trends = self._analyze_error_trends(recent_errors)
        
        report = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "days": days_back
            },
            "summary": {
                "total_errors": error_stats["total_errors"],
                "dlq_messages": dlq_analytics["total_messages"],
                "unique_error_patterns": len(error_stats["error_breakdown"]),
                "most_problematic_source": self._get_most_problematic_source(error_stats)
            },
            "error_breakdown": error_stats["error_breakdown"],
            "dlq_analytics": dlq_analytics,
            "error_trends": error_trends,
            "recommendations": self._generate_comprehensive_recommendations(
                error_stats, dlq_analytics, error_trends
            )
        }
        
        return report
    
    def _analyze_error_trends(self, recent_errors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in recent error data."""
        if not recent_errors:
            return {"trend": "no_data", "hourly_distribution": {}}
        
        # Group errors by hour
        hourly_counts = defaultdict(int)
        error_categories = defaultdict(int)
        
        for error in recent_errors:
            try:
                timestamp = datetime.fromisoformat(error["@timestamp"])
                hour_key = timestamp.strftime("%Y-%m-%d %H:00")
                hourly_counts[hour_key] += 1
                
                category = error.get("error_category", "unknown")
                error_categories[category] += 1
                
            except Exception:
                continue
        
        # Determine trend
        hours = sorted(hourly_counts.keys())
        if len(hours) >= 2:
            recent_avg = sum(hourly_counts[h] for h in hours[-6:]) / min(6, len(hours))
            earlier_avg = sum(hourly_counts[h] for h in hours[:-6]) / max(1, len(hours) - 6)
            
            if recent_avg > earlier_avg * 1.2:
                trend = "increasing"
            elif recent_avg < earlier_avg * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "trend": trend,
            "hourly_distribution": dict(hourly_counts),
            "category_distribution": dict(error_categories),
            "peak_error_hour": max(hourly_counts.items(), key=lambda x: x[1])[0] if hourly_counts else None
        }
    
    def _get_most_problematic_source(self, error_stats: Dict[str, Any]) -> Optional[str]:
        """Identify the source generating the most errors."""
        if not error_stats["error_breakdown"]:
            return None
        
        # Extract source from error pattern (format: "source_category")
        source_counts = defaultdict(int)
        for pattern, count in error_stats["error_breakdown"].items():
            if "_" in pattern:
                source = pattern.split("_")[0]
                source_counts[source] += count
        
        if source_counts:
            return max(source_counts.items(), key=lambda x: x[1])[0]
        
        return None
    
    def _generate_comprehensive_recommendations(self,
                                              error_stats: Dict[str, Any],
                                              dlq_analytics: Dict[str, Any],
                                              error_trends: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations based on all error data."""
        recommendations = []
        
        # Add DLQ recommendations
        recommendations.extend(dlq_analytics.get("recommendations", []))
        
        # Trend-based recommendations
        if error_trends["trend"] == "increasing":
            recommendations.append(
                "Error rate is increasing - investigate recent changes and implement "
                "additional monitoring"
            )
        
        # Category-specific recommendations
        categories = error_trends.get("category_distribution", {})
        
        if categories.get("rate_limit", 0) > categories.get("network_error", 0):
            recommendations.append(
                "Rate limiting is the primary issue - implement more conservative "
                "request patterns and consider upgrading API limits"
            )
        
        if categories.get("authentication", 0) > 0:
            recommendations.append(
                "Authentication errors detected - review API credentials and "
                "implement credential rotation procedures"
            )
        
        # Performance recommendations
        total_errors = error_stats["total_errors"]
        if total_errors > 100:
            recommendations.append(
                f"High error volume ({total_errors}) - consider implementing "
                "additional error prevention measures"
            )
        
        if not recommendations:
            recommendations.append("Error handling system is operating within normal parameters")
        
        return recommendations
    
    def cleanup_old_data(self, days_old: int = 7) -> Dict[str, int]:
        """Clean up old error handling data."""
        return {
            "dlq_messages_cleaned": self.dlq.cleanup_old_messages(days_old),
            "elasticsearch_cleanup": 0  # Would implement ES index cleanup here
        }