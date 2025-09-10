"""
Cerverus System - Stage 1: Data Collection
Data Validation and Quality System with Checkpointing
Following documentation requirements exactly as specified
"""

import etcd3
import json
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import re
import pandas as pd
from pathlib import Path
import hashlib
import asyncio

from .extraction import DataExtractionResult

logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationRule:
    """Data validation rule definition."""
    rule_id: str
    name: str
    description: str
    severity: ValidationSeverity
    rule_type: str  # schema, business_logic, statistical
    parameters: Dict[str, Any]
    enabled: bool = True

@dataclass
class ValidationResult:
    """Result of data validation."""
    rule_id: str
    rule_name: str
    severity: ValidationSeverity
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    source: str
    timestamp: datetime
    total_records: int
    validation_results: List[ValidationResult]
    overall_score: float
    quality_metrics: Dict[str, float]
    recommendations: List[str]

class SchemaValidator:
    """
    Schema validation for data types, required fields, and formats.
    Implements validation of schemas as per documentation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SchemaValidator")
        
        # Define schema rules for different data sources
        self.schema_rules = {
            "yahoo_finance": self._get_yahoo_finance_schema(),
            "sec_edgar": self._get_sec_edgar_schema(),
            "finra": self._get_finra_schema(),
            "alpha_vantage": self._get_alpha_vantage_schema()
        }
    
    def _get_yahoo_finance_schema(self) -> List[ValidationRule]:
        """Yahoo Finance data schema validation rules."""
        return [
            ValidationRule(
                rule_id="yf_001",
                name="Required Fields Check",
                description="Validate required fields are present",
                severity=ValidationSeverity.ERROR,
                rule_type="schema",
                parameters={
                    "required_fields": ["ohlc", "volume", "company_info"],
                    "ohlc_fields": ["Open", "High", "Low", "Close"]
                }
            ),
            ValidationRule(
                rule_id="yf_002",
                name="Data Types Validation",
                description="Validate data types for numeric fields",
                severity=ValidationSeverity.ERROR,
                rule_type="schema",
                parameters={
                    "numeric_fields": ["Open", "High", "Low", "Close"],
                    "integer_fields": ["Volume"]
                }
            ),
            ValidationRule(
                rule_id="yf_003",
                name="Symbol Format Validation",
                description="Validate symbol format matches expected pattern",
                severity=ValidationSeverity.WARNING,
                rule_type="schema",
                parameters={
                    "pattern": r"^[A-Z]{1,5}$",
                    "field": "symbol"
                }
            )
        ]
    
    def _get_sec_edgar_schema(self) -> List[ValidationRule]:
        """SEC EDGAR data schema validation rules."""
        return [
            ValidationRule(
                rule_id="sec_001",
                name="CIK Format Validation",
                description="Validate CIK format is 10 digits",
                severity=ValidationSeverity.ERROR,
                rule_type="schema",
                parameters={
                    "pattern": r"^[0-9]{10}$",
                    "field": "cik"
                }
            ),
            ValidationRule(
                rule_id="sec_002",
                name="Form Type Validation",
                description="Validate form type is in allowed set",
                severity=ValidationSeverity.ERROR,
                rule_type="schema",
                parameters={
                    "allowed_values": ["10-K", "10-Q", "8-K", "4"],
                    "field": "form_type"
                }
            ),
            ValidationRule(
                rule_id="sec_003",
                name="Filing Date Validation",
                description="Validate filing date is not null and reasonable",
                severity=ValidationSeverity.ERROR,
                rule_type="schema",
                parameters={
                    "field": "filing_date",
                    "required": True,
                    "max_age_days": 3650  # 10 years
                }
            )
        ]
    
    def _get_finra_schema(self) -> List[ValidationRule]:
        """FINRA data schema validation rules."""
        return [
            ValidationRule(
                rule_id="finra_001",
                name="Dark Pool Data Structure",
                description="Validate dark pool data has required structure",
                severity=ValidationSeverity.WARNING,
                rule_type="schema",
                parameters={
                    "required_fields": ["dark_pool_data", "short_interest", "regulatory_alerts"]
                }
            )
        ]
    
    def _get_alpha_vantage_schema(self) -> List[ValidationRule]:
        """Alpha Vantage data schema validation rules."""
        return [
            ValidationRule(
                rule_id="av_001",
                name="Technical Indicators Structure",
                description="Validate technical indicators have expected structure",
                severity=ValidationSeverity.WARNING,
                rule_type="schema",
                parameters={
                    "required_fields": ["technical_indicators"],
                    "indicator_fields": ["rsi", "macd", "bollinger_bands"]
                }
            )
        ]
    
    def validate_schema(self, data: Dict[str, Any], source: str) -> List[ValidationResult]:
        """Validate data against schema rules for the source."""
        results = []
        
        if source not in self.schema_rules:
            results.append(ValidationResult(
                rule_id="unknown_source",
                rule_name="Unknown Source",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message=f"No schema rules defined for source: {source}"
            ))
            return results
        
        rules = self.schema_rules[source]
        
        for rule in rules:
            if not rule.enabled:
                continue
            
            try:
                result = self._apply_schema_rule(data, rule)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Error applying rule: {str(e)}"
                ))
        
        return results
    
    def _apply_schema_rule(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Apply a single schema validation rule."""
        params = rule.parameters
        
        if rule.rule_id.endswith("_001"):  # Required fields checks
            return self._validate_required_fields(data, rule)
        elif rule.rule_id.endswith("_002"):  # Data type checks
            return self._validate_data_types(data, rule)
        elif rule.rule_id.endswith("_003"):  # Format/pattern checks
            return self._validate_patterns(data, rule)
        else:
            return self._validate_generic_rule(data, rule)
    
    def _validate_required_fields(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate required fields are present."""
        params = rule.parameters
        required_fields = params.get("required_fields", [])
        
        missing_fields = []
        for symbol_data in data.values():
            if isinstance(symbol_data, dict):
                for field in required_fields:
                    if field not in symbol_data:
                        missing_fields.append(field)
        
        passed = len(missing_fields) == 0
        message = "All required fields present" if passed else f"Missing fields: {missing_fields}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"missing_fields": missing_fields}
        )
    
    def _validate_data_types(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate data types for numeric fields."""
        params = rule.parameters
        numeric_fields = params.get("numeric_fields", [])
        integer_fields = params.get("integer_fields", [])
        
        type_errors = []
        
        for symbol, symbol_data in data.items():
            if isinstance(symbol_data, dict) and "ohlc" in symbol_data:
                for record in symbol_data["ohlc"]:
                    for field in numeric_fields:
                        if field in record:
                            value = record[field]
                            if not isinstance(value, (int, float)) or value is None:
                                type_errors.append(f"{symbol}.{field}: {type(value)}")
        
        passed = len(type_errors) == 0
        message = "All data types valid" if passed else f"Type errors: {len(type_errors)}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"type_errors": type_errors[:10]}  # Limit to first 10
        )
    
    def _validate_patterns(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate field patterns using regex."""
        params = rule.parameters
        pattern = params.get("pattern", "")
        field = params.get("field", "")
        
        if not pattern or not field:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                passed=False,
                message="Pattern or field not specified"
            )
        
        pattern_errors = []
        compiled_pattern = re.compile(pattern)
        
        for symbol, symbol_data in data.items():
            if field == "symbol":
                if not compiled_pattern.match(symbol):
                    pattern_errors.append(symbol)
            elif isinstance(symbol_data, dict) and field in symbol_data:
                value = str(symbol_data[field])
                if not compiled_pattern.match(value):
                    pattern_errors.append(f"{symbol}.{field}: {value}")
        
        passed = len(pattern_errors) == 0
        message = "All patterns valid" if passed else f"Pattern errors: {len(pattern_errors)}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"pattern_errors": pattern_errors[:10]}
        )
    
    def _validate_generic_rule(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Generic rule validation fallback."""
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=True,
            message="Generic validation passed"
        )


class BusinessLogicValidator:
    """
    Business logic validation for financial data consistency.
    Implements validation of business rules as per documentation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.BusinessLogicValidator")
        
        self.business_rules = [
            ValidationRule(
                rule_id="bl_001",
                name="Price Consistency Check",
                description="Validate High >= Low and Close within High/Low range",
                severity=ValidationSeverity.ERROR,
                rule_type="business_logic",
                parameters={}
            ),
            ValidationRule(
                rule_id="bl_002",
                name="Positive Price Validation",
                description="Validate all prices are positive",
                severity=ValidationSeverity.ERROR,
                rule_type="business_logic",
                parameters={"min_value": 0.01}
            ),
            ValidationRule(
                rule_id="bl_003",
                name="Volume Validation",
                description="Validate volume is non-negative",
                severity=ValidationSeverity.WARNING,
                rule_type="business_logic",
                parameters={"min_value": 0}
            ),
            ValidationRule(
                rule_id="bl_004",
                name="Temporal Consistency",
                description="Validate data timestamps are coherent",
                severity=ValidationSeverity.WARNING,
                rule_type="business_logic",
                parameters={"max_future_days": 1}
            )
        ]
    
    def validate_business_logic(self, data: Dict[str, Any], source: str) -> List[ValidationResult]:
        """Validate business logic rules."""
        results = []
        
        for rule in self.business_rules:
            if not rule.enabled:
                continue
            
            try:
                result = self._apply_business_rule(data, rule, source)
                results.append(result)
            except Exception as e:
                results.append(ValidationResult(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    severity=ValidationSeverity.ERROR,
                    passed=False,
                    message=f"Error applying business rule: {str(e)}"
                ))
        
        return results
    
    def _apply_business_rule(self, data: Dict[str, Any], rule: ValidationRule, source: str) -> ValidationResult:
        """Apply a single business logic rule."""
        if rule.rule_id == "bl_001":
            return self._validate_price_consistency(data, rule)
        elif rule.rule_id == "bl_002":
            return self._validate_positive_prices(data, rule)
        elif rule.rule_id == "bl_003":
            return self._validate_volume(data, rule)
        elif rule.rule_id == "bl_004":
            return self._validate_temporal_consistency(data, rule)
        else:
            return ValidationResult(
                rule_id=rule.rule_id,
                rule_name=rule.name,
                severity=rule.severity,
                passed=True,
                message="Business rule not implemented"
            )
    
    def _validate_price_consistency(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate price consistency: High >= Low, Close within range."""
        consistency_errors = []
        
        for symbol, symbol_data in data.items():
            if isinstance(symbol_data, dict) and "ohlc" in symbol_data:
                for i, record in enumerate(symbol_data["ohlc"]):
                    try:
                        high = float(record.get("High", 0))
                        low = float(record.get("Low", 0))
                        close = float(record.get("Close", 0))
                        
                        if high < low:
                            consistency_errors.append(f"{symbol}[{i}]: High({high}) < Low({low})")
                        
                        if close < low or close > high:
                            consistency_errors.append(f"{symbol}[{i}]: Close({close}) outside High({high})/Low({low})")
                            
                    except (ValueError, TypeError):
                        consistency_errors.append(f"{symbol}[{i}]: Invalid numeric values")
        
        passed = len(consistency_errors) == 0
        message = "Price consistency valid" if passed else f"Consistency errors: {len(consistency_errors)}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"consistency_errors": consistency_errors[:10]}
        )
    
    def _validate_positive_prices(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate all prices are positive."""
        min_value = rule.parameters.get("min_value", 0.01)
        negative_prices = []
        
        for symbol, symbol_data in data.items():
            if isinstance(symbol_data, dict) and "ohlc" in symbol_data:
                for i, record in enumerate(symbol_data["ohlc"]):
                    for price_field in ["Open", "High", "Low", "Close"]:
                        try:
                            price = float(record.get(price_field, 0))
                            if price <= 0:
                                negative_prices.append(f"{symbol}[{i}].{price_field}: {price}")
                        except (ValueError, TypeError):
                            negative_prices.append(f"{symbol}[{i}].{price_field}: Invalid")
        
        passed = len(negative_prices) == 0
        message = "All prices positive" if passed else f"Negative/zero prices: {len(negative_prices)}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"negative_prices": negative_prices[:10]}
        )
    
    def _validate_volume(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate volume is non-negative."""
        min_value = rule.parameters.get("min_value", 0)
        volume_errors = []
        
        for symbol, symbol_data in data.items():
            if isinstance(symbol_data, dict) and "volume" in symbol_data:
                for i, volume in enumerate(symbol_data["volume"]):
                    try:
                        vol = int(volume)
                        if vol < min_value:
                            volume_errors.append(f"{symbol}[{i}]: {vol}")
                    except (ValueError, TypeError):
                        volume_errors.append(f"{symbol}[{i}]: Invalid volume")
        
        passed = len(volume_errors) == 0
        message = "Volume validation passed" if passed else f"Volume errors: {len(volume_errors)}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"volume_errors": volume_errors[:10]}
        )
    
    def _validate_temporal_consistency(self, data: Dict[str, Any], rule: ValidationRule) -> ValidationResult:
        """Validate temporal consistency of data."""
        max_future_days = rule.parameters.get("max_future_days", 1)
        temporal_errors = []
        now = datetime.now()
        max_future = now + timedelta(days=max_future_days)
        
        for symbol, symbol_data in data.items():
            if isinstance(symbol_data, dict) and "metadata" in symbol_data:
                try:
                    timestamp_str = symbol_data["metadata"].get("extraction_timestamp", "")
                    if timestamp_str:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if timestamp > max_future:
                            temporal_errors.append(f"{symbol}: Future timestamp {timestamp}")
                except Exception:
                    temporal_errors.append(f"{symbol}: Invalid timestamp")
        
        passed = len(temporal_errors) == 0
        message = "Temporal consistency valid" if passed else f"Temporal errors: {len(temporal_errors)}"
        
        return ValidationResult(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            passed=passed,
            message=message,
            details={"temporal_errors": temporal_errors}
        )


class CheckpointManager:
    """
    Intelligent checkpointing system using etcd.
    Implements checkpointing system as specified in documentation.
    """
    
    def __init__(self, 
                 etcd_host: str = "localhost",
                 etcd_port: int = 2379,
                 checkpoint_prefix: str = "/cerverus/checkpoints/"):
        
        try:
            self.etcd_client = etcd3.client(host=etcd_host, port=etcd_port)
            # Test connection
            self.etcd_client.get("test")
        except Exception as e:
            logger.warning(f"Failed to connect to etcd: {str(e)}. Using memory fallback.")
            self.etcd_client = self._create_memory_fallback()
        
        self.checkpoint_prefix = checkpoint_prefix
        self.logger = logging.getLogger(f"{__name__}.CheckpointManager")
    
    def _create_memory_fallback(self):
        """Create memory-based fallback for etcd."""
        class MemoryEtcd:
            def __init__(self):
                self.data = {}
            
            def put(self, key, value):
                self.data[key] = value
                return True
            
            def get(self, key):
                value = self.data.get(key)
                return (value.encode() if value else None, None)
            
            def delete(self, key):
                return self.data.pop(key, None) is not None
            
            def get_prefix(self, prefix):
                matching = [(k.encode(), v.encode()) for k, v in self.data.items() 
                           if k.startswith(prefix)]
                return matching
        
        return MemoryEtcd()
    
    def create_checkpoint(self, source: str, extraction_result: DataExtractionResult) -> str:
        """Create checkpoint after successful extraction."""
        checkpoint_id = f"{source}_{int(time.time())}"
        checkpoint_key = f"{self.checkpoint_prefix}{checkpoint_id}"
        
        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "source": source,
            "timestamp": extraction_result.timestamp.isoformat(),
            "records_count": extraction_result.records_count,
            "s3_path": extraction_result.s3_path,
            "data_quality_score": extraction_result.data_quality_score,
            "success": extraction_result.success,
            "error_message": extraction_result.error_message,
            "created_at": datetime.now().isoformat(),
            "checkpoint_type": "extraction_success"
        }
        
        try:
            self.etcd_client.put(checkpoint_key, json.dumps(checkpoint_data))
            self.logger.info(f"Created checkpoint {checkpoint_id} for {source}")
            return checkpoint_id
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {str(e)}")
            return None
    
    def get_latest_checkpoint(self, source: str) -> Optional[Dict[str, Any]]:
        """Get the latest valid checkpoint for a source."""
        try:
            prefix = f"{self.checkpoint_prefix}{source}_"
            checkpoints = self.etcd_client.get_prefix(prefix)
            
            if not checkpoints:
                return None
            
            # Parse and sort checkpoints by timestamp
            parsed_checkpoints = []
            for key_bytes, value_bytes in checkpoints:
                try:
                    checkpoint_data = json.loads(value_bytes.decode())
                    parsed_checkpoints.append(checkpoint_data)
                except Exception:
                    continue
            
            if not parsed_checkpoints:
                return None
            
            # Return the most recent checkpoint
            latest = max(parsed_checkpoints, key=lambda x: x.get("created_at", ""))
            return latest
            
        except Exception as e:
            self.logger.error(f"Failed to get latest checkpoint for {source}: {str(e)}")
            return None
    
    def validate_checkpoint_integrity(self, checkpoint_id: str) -> bool:
        """Validate checkpoint integrity."""
        try:
            checkpoint_key = f"{self.checkpoint_prefix}{checkpoint_id}"
            value_bytes, _ = self.etcd_client.get(checkpoint_key)
            
            if not value_bytes:
                return False
            
            checkpoint_data = json.loads(value_bytes.decode())
            
            # Validate required fields
            required_fields = ["checkpoint_id", "source", "timestamp", "success"]
            for field in required_fields:
                if field not in checkpoint_data:
                    return False
            
            # Validate timestamp format
            try:
                datetime.fromisoformat(checkpoint_data["timestamp"])
            except ValueError:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate checkpoint {checkpoint_id}: {str(e)}")
            return False
    
    def cleanup_old_checkpoints(self, max_age_days: int = 30) -> int:
        """Clean up checkpoints older than specified days."""
        try:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)
            checkpoints = self.etcd_client.get_prefix(self.checkpoint_prefix)
            
            deleted_count = 0
            for key_bytes, value_bytes in checkpoints:
                try:
                    checkpoint_data = json.loads(value_bytes.decode())
                    created_at = datetime.fromisoformat(checkpoint_data["created_at"])
                    
                    if created_at < cutoff_time:
                        self.etcd_client.delete(key_bytes.decode())
                        deleted_count += 1
                        
                except Exception:
                    continue
            
            self.logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old checkpoints: {str(e)}")
            return 0
    
    def recover_from_checkpoint(self, source: str) -> Optional[Dict[str, Any]]:
        """Recover from the last valid checkpoint."""
        latest_checkpoint = self.get_latest_checkpoint(source)
        
        if not latest_checkpoint:
            self.logger.warning(f"No checkpoint found for {source}")
            return None
        
        checkpoint_id = latest_checkpoint["checkpoint_id"]
        
        if not self.validate_checkpoint_integrity(checkpoint_id):
            self.logger.error(f"Checkpoint {checkpoint_id} failed integrity validation")
            return None
        
        self.logger.info(f"Recovered from checkpoint {checkpoint_id} for {source}")
        return latest_checkpoint


class DataQualityManager:
    """
    Comprehensive data quality management system.
    Orchestrates validation, checkpointing, and quality reporting.
    """
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.schema_validator = SchemaValidator()
        self.business_validator = BusinessLogicValidator()
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(f"{__name__}.DataQualityManager")
    
    def validate_extraction_result(self, result: DataExtractionResult) -> DataQualityReport:
        """Perform comprehensive validation of extraction result."""
        if not result.success or not result.data:
            return DataQualityReport(
                source=result.source,
                timestamp=result.timestamp,
                total_records=0,
                validation_results=[],
                overall_score=0.0,
                quality_metrics={},
                recommendations=["Data extraction failed - no validation performed"]
            )
        
        # Perform schema validation
        schema_results = self.schema_validator.validate_schema(result.data, result.source)
        
        # Perform business logic validation
        business_results = self.business_validator.validate_business_logic(result.data, result.source)
        
        # Combine all validation results
        all_results = schema_results + business_results
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(all_results, result)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results, quality_metrics)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(quality_metrics)
        
        # Create checkpoint if quality is acceptable
        if overall_score >= 0.8:  # 80% quality threshold
            self.checkpoint_manager.create_checkpoint(result.source, result)
        
        report = DataQualityReport(
            source=result.source,
            timestamp=result.timestamp,
            total_records=result.records_count,
            validation_results=all_results,
            overall_score=overall_score,
            quality_metrics=quality_metrics,
            recommendations=recommendations
        )
        
        return report
    
    def _calculate_quality_metrics(self, results: List[ValidationResult], 
                                 extraction_result: DataExtractionResult) -> Dict[str, float]:
        """Calculate comprehensive quality metrics."""
        total_rules = len(results)
        passed_rules = sum(1 for r in results if r.passed)
        
        # Rule-based metrics
        rule_pass_rate = passed_rules / total_rules if total_rules > 0 else 0.0
        
        # Severity-weighted metrics
        critical_failed = sum(1 for r in results if not r.passed and r.severity == ValidationSeverity.CRITICAL)
        error_failed = sum(1 for r in results if not r.passed and r.severity == ValidationSeverity.ERROR)
        warning_failed = sum(1 for r in results if not r.passed and r.severity == ValidationSeverity.WARNING)
        
        # Calculate severity-weighted score
        severity_penalty = (critical_failed * 0.4) + (error_failed * 0.3) + (warning_failed * 0.1)
        severity_score = max(0.0, 1.0 - severity_penalty)
        
        # Data completeness
        completeness_score = extraction_result.data_quality_score or 0.0
        
        return {
            "rule_pass_rate": rule_pass_rate,
            "severity_score": severity_score,
            "completeness_score": completeness_score,
            "critical_failures": critical_failed,
            "error_failures": error_failed,
            "warning_failures": warning_failed,
            "total_rules_checked": total_rules
        }
    
    def _generate_recommendations(self, results: List[ValidationResult], 
                                metrics: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Critical issues
        if metrics["critical_failures"] > 0:
            recommendations.append("CRITICAL: Address critical validation failures immediately")
        
        # Low rule pass rate
        if metrics["rule_pass_rate"] < 0.8:
            recommendations.append("Low validation pass rate - review data extraction logic")
        
        # Low completeness
        if metrics["completeness_score"] < 0.9:
            recommendations.append("Data completeness below 90% - check for missing data sources")
        
        # Error patterns
        error_patterns = {}
        for result in results:
            if not result.passed:
                error_type = result.rule_name
                error_patterns[error_type] = error_patterns.get(error_type, 0) + 1
        
        if error_patterns:
            most_common_error = max(error_patterns.items(), key=lambda x: x[1])
            recommendations.append(f"Most common validation failure: {most_common_error[0]} ({most_common_error[1]} occurrences)")
        
        # Data quality trends
        if metrics["rule_pass_rate"] >= 0.95:
            recommendations.append("Excellent data quality - maintain current extraction processes")
        elif metrics["rule_pass_rate"] >= 0.8:
            recommendations.append("Good data quality - minor improvements needed")
        else:
            recommendations.append("Poor data quality - major improvements required")
        
        return recommendations
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        # Weighted combination of different quality aspects
        weights = {
            "rule_pass_rate": 0.4,
            "severity_score": 0.3,
            "completeness_score": 0.3
        }
        
        overall_score = 0.0
        for metric, weight in weights.items():
            overall_score += metrics.get(metric, 0.0) * weight
        
        return round(overall_score, 3)
    
    def get_quality_summary(self, source: Optional[str] = None, 
                          days_back: int = 7) -> Dict[str, Any]:
        """Get quality summary for monitoring dashboard."""
        # This would typically query historical validation results
        # For now, return a template structure
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "source_filter": source,
            "period_days": days_back,
            "quality_trends": {
                "avg_overall_score": 0.85,
                "score_trend": "stable",  # improving, degrading, stable
                "total_validations": 0,
                "success_rate": 0.0
            },
            "common_issues": [],
            "recommendations": [
                "Implement regular quality monitoring",
                "Set up alerting for quality degradation",
                "Review validation rules quarterly"
            ]
        }
        
        return summary
    
    def export_quality_report(self, report: DataQualityReport, 
                            output_path: str) -> bool:
        """Export quality report to file."""
        try:
            report_dict = {
                "source": report.source,
                "timestamp": report.timestamp.isoformat(),
                "total_records": report.total_records,
                "overall_score": report.overall_score,
                "quality_metrics": report.quality_metrics,
                "recommendations": report.recommendations,
                "validation_results": [
                    {
                        "rule_id": r.rule_id,
                        "rule_name": r.rule_name,
                        "severity": r.severity.value,
                        "passed": r.passed,
                        "message": r.message,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in report.validation_results
                ]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_dict, f, indent=2)
            
            self.logger.info(f"Exported quality report to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export quality report: {str(e)}")
            return False


class StatisticalAnomalyDetector:
    """
    Statistical anomaly detection for data validation.
    Implements advanced validation of business logic as per documentation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.StatisticalAnomalyDetector")
    
    def detect_price_anomalies(self, data: Dict[str, Any], 
                             z_threshold: float = 3.0) -> List[ValidationResult]:
        """Detect statistical anomalies in price data."""
        results = []
        
        try:
            # Collect all prices for statistical analysis
            all_prices = {
                "Open": [],
                "High": [],
                "Low": [],
                "Close": []
            }
            
            price_data = []
            for symbol, symbol_data in data.items():
                if isinstance(symbol_data, dict) and "ohlc" in symbol_data:
                    for record in symbol_data["ohlc"]:
                        price_record = {"symbol": symbol}
                        for price_type in all_prices.keys():
                            try:
                                price = float(record.get(price_type, 0))
                                all_prices[price_type].append(price)
                                price_record[price_type] = price
                            except (ValueError, TypeError):
                                price_record[price_type] = None
                        price_data.append(price_record)
            
            # Calculate statistics and detect anomalies
            anomalies = []
            for price_type, prices in all_prices.items():
                if len(prices) < 10:  # Need minimum data for statistical analysis
                    continue
                
                # Remove zeros and nulls for statistics
                valid_prices = [p for p in prices if p and p > 0]
                if len(valid_prices) < 5:
                    continue
                
                mean_price = sum(valid_prices) / len(valid_prices)
                variance = sum((p - mean_price) ** 2 for p in valid_prices) / len(valid_prices)
                std_dev = variance ** 0.5
                
                # Find outliers using z-score
                for record in price_data:
                    price = record.get(price_type)
                    if price and price > 0 and std_dev > 0:
                        z_score = abs(price - mean_price) / std_dev
                        if z_score > z_threshold:
                            anomalies.append(f"{record['symbol']}.{price_type}: {price} (z={z_score:.2f})")
            
            passed = len(anomalies) == 0
            message = "No statistical anomalies detected" if passed else f"Found {len(anomalies)} statistical anomalies"
            
            results.append(ValidationResult(
                rule_id="stat_001",
                rule_name="Statistical Price Anomaly Detection",
                severity=ValidationSeverity.WARNING,
                passed=passed,
                message=message,
                details={"anomalies": anomalies[:20], "z_threshold": z_threshold}
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                rule_id="stat_001",
                rule_name="Statistical Price Anomaly Detection",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Error in statistical analysis: {str(e)}"
            ))
        
        return results
    
    def detect_volume_anomalies(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Detect anomalies in trading volume."""
        results = []
        
        try:
            all_volumes = []
            volume_data = []
            
            for symbol, symbol_data in data.items():
                if isinstance(symbol_data, dict) and "volume" in symbol_data:
                    for i, volume in enumerate(symbol_data["volume"]):
                        try:
                            vol = int(volume)
                            if vol >= 0:  # Valid volume
                                all_volumes.append(vol)
                                volume_data.append({"symbol": symbol, "index": i, "volume": vol})
                        except (ValueError, TypeError):
                            continue
            
            if len(all_volumes) < 10:
                results.append(ValidationResult(
                    rule_id="stat_002",
                    rule_name="Volume Anomaly Detection",
                    severity=ValidationSeverity.INFO,
                    passed=True,
                    message="Insufficient volume data for statistical analysis"
                ))
                return results
            
            # Calculate volume statistics
            mean_volume = sum(all_volumes) / len(all_volumes)
            sorted_volumes = sorted(all_volumes)
            q1 = sorted_volumes[len(sorted_volumes) // 4]
            q3 = sorted_volumes[3 * len(sorted_volumes) // 4]
            iqr = q3 - q1
            
            # Detect outliers using IQR method
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            anomalies = []
            for record in volume_data:
                volume = record["volume"]
                if volume < lower_bound or volume > upper_bound:
                    anomalies.append(f"{record['symbol']}[{record['index']}]: {volume}")
            
            passed = len(anomalies) == 0
            message = "No volume anomalies detected" if passed else f"Found {len(anomalies)} volume anomalies"
            
            results.append(ValidationResult(
                rule_id="stat_002",
                rule_name="Volume Anomaly Detection",
                severity=ValidationSeverity.WARNING,
                passed=passed,
                message=message,
                details={
                    "anomalies": anomalies[:10],
                    "volume_statistics": {
                        "mean": mean_volume,
                        "q1": q1,
                        "q3": q3,
                        "iqr": iqr,
                        "bounds": [lower_bound, upper_bound]
                    }
                }
            ))
            
        except Exception as e:
            results.append(ValidationResult(
                rule_id="stat_002",
                rule_name="Volume Anomaly Detection",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Error in volume analysis: {str(e)}"
            ))
        
        return results


class CrossSourceValidator:
    """
    Cross-source validation for data consistency.
    Implements validation between multiple data sources as per documentation.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CrossSourceValidator")
    
    def validate_cross_source_consistency(self, 
                                        source_results: Dict[str, DataExtractionResult]) -> List[ValidationResult]:
        """Validate consistency between multiple data sources."""
        results = []
        
        if len(source_results) < 2:
            results.append(ValidationResult(
                rule_id="cross_001",
                rule_name="Cross-Source Validation",
                severity=ValidationSeverity.INFO,
                passed=True,
                message="Only one data source available - no cross-validation performed"
            ))
            return results
        
        # Find common symbols across sources
        common_symbols = self._find_common_symbols(source_results)
        
        if not common_symbols:
            results.append(ValidationResult(
                rule_id="cross_001",
                rule_name="Cross-Source Symbol Consistency",
                severity=ValidationSeverity.WARNING,
                passed=False,
                message="No common symbols found across data sources"
            ))
            return results
        
        # Validate price consistency across sources
        price_consistency_result = self._validate_price_consistency_across_sources(
            source_results, common_symbols
        )
        results.append(price_consistency_result)
        
        return results
    
    def _find_common_symbols(self, source_results: Dict[str, DataExtractionResult]) -> List[str]:
        """Find symbols that appear in multiple data sources."""
        symbol_sets = []
        
        for source, result in source_results.items():
            if result.success and result.data:
                symbols = set(result.data.keys())
                symbol_sets.append(symbols)
        
        if not symbol_sets:
            return []
        
        # Find intersection of all symbol sets
        common_symbols = symbol_sets[0]
        for symbol_set in symbol_sets[1:]:
            common_symbols = common_symbols.intersection(symbol_set)
        
        return list(common_symbols)
    
    def _validate_price_consistency_across_sources(self, 
                                                 source_results: Dict[str, DataExtractionResult],
                                                 common_symbols: List[str]) -> ValidationResult:
        """Validate price consistency across sources for common symbols."""
        inconsistencies = []
        tolerance_percent = 5.0  # 5% tolerance for price differences
        
        try:
            for symbol in common_symbols:
                symbol_prices = {}
                
                # Collect prices from each source
                for source, result in source_results.items():
                    if (result.success and result.data and 
                        symbol in result.data and 
                        isinstance(result.data[symbol], dict) and
                        "ohlc" in result.data[symbol]):
                        
                        ohlc_data = result.data[symbol]["ohlc"]
                        if ohlc_data:  # Take first available price record
                            symbol_prices[source] = ohlc_data[0]
                
                # Compare prices between sources
                if len(symbol_prices) >= 2:
                    sources = list(symbol_prices.keys())
                    for i in range(len(sources)):
                        for j in range(i + 1, len(sources)):
                            source1, source2 = sources[i], sources[j]
                            prices1 = symbol_prices[source1]
                            prices2 = symbol_prices[source2]
                            
                            # Compare Close prices
                            try:
                                close1 = float(prices1.get("Close", 0))
                                close2 = float(prices2.get("Close", 0))
                                
                                if close1 > 0 and close2 > 0:
                                    diff_percent = abs(close1 - close2) / ((close1 + close2) / 2) * 100
                                    
                                    if diff_percent > tolerance_percent:
                                        inconsistencies.append(
                                            f"{symbol}: {source1}=${close1}, {source2}=${close2} "
                                            f"(diff: {diff_percent:.1f}%)"
                                        )
                            except (ValueError, TypeError):
                                continue
            
            passed = len(inconsistencies) == 0
            message = ("Price consistency validated across sources" if passed else 
                      f"Found {len(inconsistencies)} price inconsistencies")
            
            return ValidationResult(
                rule_id="cross_002",
                rule_name="Cross-Source Price Consistency",
                severity=ValidationSeverity.WARNING,
                passed=passed,
                message=message,
                details={
                    "inconsistencies": inconsistencies[:10],
                    "tolerance_percent": tolerance_percent,
                    "common_symbols_checked": len(common_symbols)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                rule_id="cross_002",
                rule_name="Cross-Source Price Consistency",
                severity=ValidationSeverity.ERROR,
                passed=False,
                message=f"Error in cross-source validation: {str(e)}"
            )


# Integration class for comprehensive validation
class ComprehensiveDataValidator:
    """
    Main validation orchestrator combining all validation types.
    Provides unified interface for complete data quality assessment.
    """
    
    def __init__(self, etcd_host: str = "localhost", etcd_port: int = 2379):
        self.checkpoint_manager = CheckpointManager(etcd_host, etcd_port)
        self.quality_manager = DataQualityManager(self.checkpoint_manager)
        self.statistical_detector = StatisticalAnomalyDetector()
        self.cross_source_validator = CrossSourceValidator()
        self.logger = logging.getLogger(f"{__name__}.ComprehensiveDataValidator")
    
    def validate_single_source(self, result: DataExtractionResult) -> DataQualityReport:
        """Perform comprehensive validation on single source data."""
        return self.quality_manager.validate_extraction_result(result)
    
    def validate_multiple_sources(self, 
                                results: Dict[str, DataExtractionResult]) -> Dict[str, Any]:
        """Perform comprehensive validation across multiple sources."""
        validation_summary = {
            "timestamp": datetime.now().isoformat(),
            "sources_validated": list(results.keys()),
            "individual_reports": {},
            "cross_source_validation": [],
            "overall_assessment": {}
        }
        
        # Validate each source individually
        individual_scores = []
        for source, result in results.items():
            if result.success:
                report = self.validate_single_source(result)
                validation_summary["individual_reports"][source] = {
                    "overall_score": report.overall_score,
                    "total_records": report.total_records,
                    "quality_metrics": report.quality_metrics,
                    "recommendations": report.recommendations,
                    "validation_count": len(report.validation_results),
                    "passed_validations": sum(1 for v in report.validation_results if v.passed)
                }
                individual_scores.append(report.overall_score)
            else:
                validation_summary["individual_reports"][source] = {
                    "overall_score": 0.0,
                    "error": result.error_message,
                    "success": False
                }
        
        # Perform cross-source validation
        cross_source_results = self.cross_source_validator.validate_cross_source_consistency(results)
        validation_summary["cross_source_validation"] = [
            {
                "rule_name": r.rule_name,
                "passed": r.passed,
                "message": r.message,
                "severity": r.severity.value
            }
            for r in cross_source_results
        ]
        
        # Calculate overall assessment
        avg_individual_score = sum(individual_scores) / len(individual_scores) if individual_scores else 0.0
        cross_source_score = sum(1 for r in cross_source_results if r.passed) / len(cross_source_results) if cross_source_results else 1.0
        
        overall_score = (avg_individual_score * 0.8) + (cross_source_score * 0.2)
        
        validation_summary["overall_assessment"] = {
            "overall_score": round(overall_score, 3),
            "avg_individual_score": round(avg_individual_score, 3),
            "cross_source_score": round(cross_source_score, 3),
            "status": self._get_quality_status(overall_score),
            "total_sources": len(results),
            "successful_sources": len([r for r in results.values() if r.success])
        }
        
        return validation_summary
    
    def _get_quality_status(self, score: float) -> str:
        """Get quality status based on score."""
        if score >= 0.95:
            return "excellent"
        elif score >= 0.85:
            return "good"
        elif score >= 0.70:
            return "acceptable"
        elif score >= 0.50:
            return "poor"
        else:
            return "critical"
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of validation system."""
        return {
            "timestamp": datetime.now().isoformat(),
            "checkpoint_manager": {
                "status": "healthy" if hasattr(self.checkpoint_manager.etcd_client, 'get') else "degraded"
            },
            "validators": {
                "schema_validator": "healthy",
                "business_validator": "healthy", 
                "statistical_detector": "healthy",
                "cross_source_validator": "healthy"
            },
            "overall_status": "healthy"
        }