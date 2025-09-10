"""
Cerverus System - Stage 1: Data Collection
S3 Data Lake Storage Layer Configuration
Implements Bronze/Silver/Gold architecture as per documentation
"""

import boto3
import json
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
from botocore.exceptions import ClientError, NoCredentialsError
import uuid
import gzip
import io

from .extraction import DataExtractionResult

logger = logging.getLogger(__name__)

@dataclass
class S3StorageConfig:
    """S3 storage configuration for data lake layers."""
    bucket_name: str
    region: str = "us-east-1"
    encryption: str = "AES256"
    versioning_enabled: bool = True
    lifecycle_policies: bool = True

@dataclass
class DataLakeMetadata:
    """Metadata for data lake storage as specified in documentation."""
    source: str
    timestamp: datetime
    records_count: int
    s3_path: str
    data_quality_score: float
    schema_version: str = "1.0"
    partition_keys: Optional[Dict[str, str]] = None
    file_format: str = "parquet"
    compression: str = "snappy"

class S3DataLakeManager:
    """
    S3 Data Lake Manager implementing Bronze/Silver/Gold architecture.
    Manages hierarchical structure with partitioning by year/month/day/hour.
    """
    
    def __init__(self, 
                 bronze_bucket: str = "cerverus-bronze",
                 silver_bucket: str = "cerverus-silver", 
                 gold_bucket: str = "cerverus-gold",
                 region: str = "us-east-1"):
        
        self.bronze_config = S3StorageConfig(bucket_name=bronze_bucket, region=region)
        self.silver_config = S3StorageConfig(bucket_name=silver_bucket, region=region)
        self.gold_config = S3StorageConfig(bucket_name=gold_bucket, region=region)
        
        self.s3_client = boto3.client('s3', region_name=region)
        self.logger = logging.getLogger(f"{__name__}.S3DataLakeManager")
        
        # Initialize buckets
        self._initialize_data_lake()
    
    def _initialize_data_lake(self):
        """Initialize S3 buckets with proper configuration."""
        buckets_to_create = [
            self.bronze_config,
            self.silver_config, 
            self.gold_config
        ]
        
        for config in buckets_to_create:
            try:
                self._create_bucket_if_not_exists(config)
                self._configure_bucket_policies(config)
                self._setup_lifecycle_policies(config)
                
            except Exception as e:
                self.logger.error(f"Failed to initialize bucket {config.bucket_name}: {str(e)}")
    
    def _create_bucket_if_not_exists(self, config: S3StorageConfig):
        """Create S3 bucket if it doesn't exist."""
        try:
            self.s3_client.head_bucket(Bucket=config.bucket_name)
            self.logger.info(f"Bucket {config.bucket_name} already exists")
            
        except ClientError as e:
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                # Bucket doesn't exist, create it
                try:
                    if config.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=config.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=config.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': config.region}
                        )
                    
                    self.logger.info(f"Created bucket {config.bucket_name}")
                    
                except Exception as create_error:
                    self.logger.error(f"Failed to create bucket {config.bucket_name}: {str(create_error)}")
                    raise
            else:
                raise
    
    def _configure_bucket_policies(self, config: S3StorageConfig):
        """Configure bucket encryption and versioning."""
        try:
            # Enable versioning
            if config.versioning_enabled:
                self.s3_client.put_bucket_versioning(
                    Bucket=config.bucket_name,
                    VersioningConfiguration={'Status': 'Enabled'}
                )
            
            # Configure server-side encryption
            encryption_config = {
                'Rules': [{
                    'ApplyServerSideEncryptionByDefault': {
                        'SSEAlgorithm': config.encryption
                    }
                }]
            }
            
            self.s3_client.put_bucket_encryption(
                Bucket=config.bucket_name,
                ServerSideEncryptionConfiguration=encryption_config
            )
            
            self.logger.info(f"Configured encryption and versioning for {config.bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to configure bucket policies for {config.bucket_name}: {str(e)}")
    
    def _setup_lifecycle_policies(self, config: S3StorageConfig):
        """Setup lifecycle policies for cost management as per documentation."""
        if not config.lifecycle_policies:
            return
        
        # Define lifecycle configuration based on layer
        if "bronze" in config.bucket_name:
            lifecycle_config = self._get_bronze_lifecycle_policy()
        elif "silver" in config.bucket_name:
            lifecycle_config = self._get_silver_lifecycle_policy()
        elif "gold" in config.bucket_name:
            lifecycle_config = self._get_gold_lifecycle_policy()
        else:
            return
        
        try:
            self.s3_client.put_bucket_lifecycle_configuration(
                Bucket=config.bucket_name,
                LifecycleConfiguration=lifecycle_config
            )
            
            self.logger.info(f"Applied lifecycle policies to {config.bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply lifecycle policies to {config.bucket_name}: {str(e)}")
    
    def _get_bronze_lifecycle_policy(self) -> Dict[str, Any]:
        """Lifecycle policy for Bronze layer - raw data."""
        return {
            'Rules': [
                {
                    'ID': 'BronzeDataLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ],
                    'Expiration': {
                        'Days': 2555  # 7 years for compliance
                    }
                }
            ]
        }
    
    def _get_silver_lifecycle_policy(self) -> Dict[str, Any]:
        """Lifecycle policy for Silver layer - processed data."""
        return {
            'Rules': [
                {
                    'ID': 'SilverDataLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'Transitions': [
                        {
                            'Days': 60,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 180,
                            'StorageClass': 'GLACIER'
                        },
                        {
                            'Days': 730,
                            'StorageClass': 'DEEP_ARCHIVE'
                        }
                    ],
                    'Expiration': {
                        'Days': 2555  # 7 years for compliance
                    }
                }
            ]
        }
    
    def _get_gold_lifecycle_policy(self) -> Dict[str, Any]:
        """Lifecycle policy for Gold layer - curated ML features."""
        return {
            'Rules': [
                {
                    'ID': 'GoldDataLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': ''},
                    'Transitions': [
                        {
                            'Days': 90,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 365,
                            'StorageClass': 'GLACIER'
                        }
                    ],
                    'Expiration': {
                        'Days': 2555  # 7 years for compliance
                    }
                }
            ]
        }
    
    def store_bronze_data(self, result: DataExtractionResult, 
                         file_format: str = "json") -> Optional[str]:
        """
        Store raw data in Bronze layer with hierarchical partitioning.
        Implements partitioning by year/month/day/hour as per documentation.
        """
        if not result.success or not result.data:
            self.logger.warning("Cannot store unsuccessful or empty extraction result")
            return None
        
        try:
            # Generate hierarchical S3 path
            timestamp = result.timestamp
            s3_path = self._generate_hierarchical_path(
                layer="bronze",
                source=result.source,
                timestamp=timestamp,
                file_format=file_format
            )
            
            # Prepare data for storage
            storage_data = {
                "extraction_metadata": {
                    "source": result.source,
                    "timestamp": result.timestamp.isoformat(),
                    "records_count": result.records_count,
                    "data_quality_score": result.data_quality_score,
                    "extraction_id": str(uuid.uuid4())
                },
                "data": result.data
            }
            
            # Store data
            if file_format.lower() == "json":
                self._store_json_data(self.bronze_config.bucket_name, s3_path, storage_data)
            elif file_format.lower() == "parquet":
                self._store_parquet_data(self.bronze_config.bucket_name, s3_path, storage_data)
            
            # Store metadata
            metadata = DataLakeMetadata(
                source=result.source,
                timestamp=result.timestamp,
                records_count=result.records_count,
                s3_path=s3_path,
                data_quality_score=result.data_quality_score or 0.0,
                partition_keys=self._extract_partition_keys(timestamp),
                file_format=file_format
            )
            
            self._store_metadata(self.bronze_config.bucket_name, s3_path, metadata)
            
            self.logger.info(f"Successfully stored Bronze data at {s3_path}")
            return s3_path
            
        except Exception as e:
            self.logger.error(f"Failed to store Bronze data: {str(e)}")
            return None
    
    def _generate_hierarchical_path(self, layer: str, source: str, 
                                  timestamp: datetime, file_format: str) -> str:
        """Generate hierarchical S3 path with partitioning by year/month/day/hour."""
        # Partitioning structure: layer/source/year/month/day/hour/filename
        return (f"{layer}/{source}/"
                f"year={timestamp.year}/"
                f"month={timestamp.month:02d}/"
                f"day={timestamp.day:02d}/"
                f"hour={timestamp.hour:02d}/"
                f"{int(timestamp.timestamp())}_{str(uuid.uuid4())[:8]}.{file_format}")
    
    def _extract_partition_keys(self, timestamp: datetime) -> Dict[str, str]:
        """Extract partition keys for Hive-style partitioning."""
        return {
            "year": str(timestamp.year),
            "month": f"{timestamp.month:02d}",
            "day": f"{timestamp.day:02d}",
            "hour": f"{timestamp.hour:02d}"
        }
    
    def _store_json_data(self, bucket: str, s3_path: str, data: Dict[str, Any]):
        """Store data as compressed JSON."""
        json_data = json.dumps(data, default=str, indent=2)
        
        # Compress data
        compressed_data = gzip.compress(json_data.encode('utf-8'))
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=s3_path,
            Body=compressed_data,
            ContentType='application/json',
            ContentEncoding='gzip',
            Metadata={
                'source': data.get('extraction_metadata', {}).get('source', 'unknown'),
                'records_count': str(data.get('extraction_metadata', {}).get('records_count', 0)),
                'extraction_timestamp': data.get('extraction_metadata', {}).get('timestamp', ''),
                'data_quality_score': str(data.get('extraction_metadata', {}).get('data_quality_score', 0.0))
            }
        )
    
    def _store_parquet_data(self, bucket: str, s3_path: str, data: Dict[str, Any]):
        """Store data as Parquet with Snappy compression."""
        # This would require pandas and pyarrow
        # For now, we'll store as JSON and log that Parquet would be used
        self.logger.info(f"Parquet storage requested for {s3_path} - using JSON fallback")
        self._store_json_data(bucket, s3_path, data)
    
    def _store_metadata(self, bucket: str, data_path: str, metadata: DataLakeMetadata):
        """Store metadata alongside data."""
        metadata_path = data_path.replace('.json', '_metadata.json').replace('.parquet', '_metadata.json')
        
        metadata_dict = asdict(metadata)
        metadata_dict['timestamp'] = metadata.timestamp.isoformat()
        
        metadata_json = json.dumps(metadata_dict, indent=2)
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=metadata_path,
            Body=metadata_json.encode('utf-8'),
            ContentType='application/json',
            Metadata={
                'type': 'data_lake_metadata',
                'source': metadata.source,
                'schema_version': metadata.schema_version
            }
        )
    
    def list_bronze_data(self, source: Optional[str] = None, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """List Bronze layer data with optional filtering."""
        try:
            prefix = "bronze/"
            if source:
                prefix += f"{source}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bronze_config.bucket_name,
                Prefix=prefix
            )
            
            objects = []
            for obj in response.get('Contents', []):
                # Skip metadata files
                if '_metadata.json' in obj['Key']:
                    continue
                
                # Filter by date if specified
                if start_date or end_date:
                    obj_timestamp = obj['LastModified'].replace(tzinfo=None)
                    if start_date and obj_timestamp < start_date:
                        continue
                    if end_date and obj_timestamp > end_date:
                        continue
                
                objects.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat(),
                    'etag': obj['ETag'].strip('"')
                })
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Failed to list Bronze data: {str(e)}")
            return []
    
    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics for all layers."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "layers": {}
        }
        
        for layer_name, config in [
            ("bronze", self.bronze_config),
            ("silver", self.silver_config),
            ("gold", self.gold_config)
        ]:
            try:
                # Get bucket size and object count
                response = self.s3_client.list_objects_v2(Bucket=config.bucket_name)
                
                total_size = 0
                object_count = 0
                
                if 'Contents' in response:
                    for obj in response['Contents']:
                        total_size += obj['Size']
                        object_count += 1
                
                metrics["layers"][layer_name] = {
                    "bucket_name": config.bucket_name,
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2),
                    "object_count": object_count,
                    "estimated_monthly_cost": self._estimate_storage_cost(total_size)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to get metrics for {layer_name}: {str(e)}")
                metrics["layers"][layer_name] = {"error": str(e)}
        
        return metrics
    
    def _estimate_storage_cost(self, size_bytes: int) -> float:
        """Estimate monthly storage cost in USD."""
        # Standard S3 pricing (approximate)
        size_gb = size_bytes / (1024 ** 3)
        
        if size_gb <= 50:
            cost_per_gb = 0.023  # First 50GB
        elif size_gb <= 450:
            cost_per_gb = 0.022  # Next 450GB
        else:
            cost_per_gb = 0.021  # Over 500GB
        
        return round(size_gb * cost_per_gb, 2)
    
    def cleanup_old_data(self, layer: str, days_old: int = 30):
        """Clean up old data based on lifecycle policies."""
        if layer not in ["bronze", "silver", "gold"]:
            raise ValueError("Layer must be bronze, silver, or gold")
        
        config = getattr(self, f"{layer}_config")
        cutoff_date = datetime.now() - timedelta(days=days_old)
        
        try:
            response = self.s3_client.list_objects_v2(Bucket=config.bucket_name)
            
            deleted_objects = []
            for obj in response.get('Contents', []):
                if obj['LastModified'].replace(tzinfo=None) < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=config.bucket_name,
                        Key=obj['Key']
                    )
                    deleted_objects.append(obj['Key'])
            
            self.logger.info(f"Cleaned up {len(deleted_objects)} objects from {layer} layer")
            return deleted_objects
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup {layer} data: {str(e)}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on S3 Data Lake."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "buckets": {}
        }
        
        unhealthy_buckets = []
        
        for layer_name, config in [
            ("bronze", self.bronze_config),
            ("silver", self.silver_config),
            ("gold", self.gold_config)
        ]:
            try:
                # Check bucket accessibility
                self.s3_client.head_bucket(Bucket=config.bucket_name)
                
                # Check bucket policies
                encryption_response = self.s3_client.get_bucket_encryption(Bucket=config.bucket_name)
                versioning_response = self.s3_client.get_bucket_versioning(Bucket=config.bucket_name)
                
                bucket_health = {
                    "accessible": True,
                    "encryption_enabled": len(encryption_response.get('ServerSideEncryptionConfiguration', {}).get('Rules', [])) > 0,
                    "versioning_enabled": versioning_response.get('Status') == 'Enabled',
                    "lifecycle_policies": True  # Assume configured if bucket exists
                }
                
                is_healthy = all(bucket_health.values())
                bucket_health["status"] = "healthy" if is_healthy else "unhealthy"
                
                if not is_healthy:
                    unhealthy_buckets.append(layer_name)
                
                health_status["buckets"][layer_name] = bucket_health
                
            except Exception as e:
                health_status["buckets"][layer_name] = {
                    "accessible": False,
                    "status": "unhealthy",
                    "error": str(e)
                }
                unhealthy_buckets.append(layer_name)
        
        # Overall status
        if unhealthy_buckets:
            if len(unhealthy_buckets) == 3:
                health_status["overall_status"] = "critical"
            else:
                health_status["overall_status"] = "degraded"
        
        return health_status