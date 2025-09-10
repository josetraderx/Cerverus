"""
Cerverus System - Stage 1: Data Collection
Multilevel Cache System Implementation
L1 (Redis) - Hot Data, L2 (Memory) - Frequently Accessed, L3 (S3) - Cold Storage
Following documentation requirements exactly as specified
"""

import redis
import json
import time
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from collections import OrderedDict
import threading
import asyncio
import pickle
import gzip
from pathlib import Path
import weakref

from .storage import S3DataLakeManager

logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Cache metrics for monitoring as per documentation."""
    hit_count: int = 0
    miss_count: int = 0
    eviction_count: int = 0
    total_requests: int = 0
    
    @property
    def hit_ratio(self) -> float:
        """Calculate hit/miss ratio."""
        if self.total_requests == 0:
            return 0.0
        return self.hit_count / self.total_requests
    
    @property
    def miss_ratio(self) -> float:
        """Calculate miss ratio."""
        return 1.0 - self.hit_ratio

class CacheL1Redis:
    """
    Cache L1 (Redis) - Hot Data
    For real-time market data with high volatility
    As specified in Stage 1 documentation
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 password: Optional[str] = None,
                 cluster_enabled: bool = False):
        
        self.host = host
        self.port = port
        self.db = db
        self.cluster_enabled = cluster_enabled
        
        try:
            if cluster_enabled:
                from rediscluster import RedisCluster
                self.redis_client = RedisCluster(
                    startup_nodes=[{"host": host, "port": port}],
                    password=password,
                    decode_responses=True
                )
            else:
                self.redis_client = redis.Redis(
                    host=host,
                    port=port,
                    db=db,
                    password=password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
            
            # Test connection
            self.redis_client.ping()
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            # Fallback to mock Redis for development
            self.redis_client = self._create_mock_redis()
        
        self.metrics = CacheMetrics()
        self.logger = logging.getLogger(f"{__name__}.CacheL1Redis")
        
        # TTL configuration based on data volatility
        self.default_ttl = {
            "market_data": 300,      # 5 minutes for market data
            "company_info": 3600,    # 1 hour for company information
            "regulatory": 86400,     # 24 hours for regulatory data
            "technical": 1800        # 30 minutes for technical indicators
        }
    
    def _create_mock_redis(self):
        """Create mock Redis client for development/testing."""
        class MockRedis:
            def __init__(self):
                self.data = {}
                self.expiry = {}
            
            def ping(self):
                return True
            
            def set(self, key, value, ex=None):
                self.data[key] = value
                if ex:
                    self.expiry[key] = time.time() + ex
                return True
            
            def get(self, key):
                if key in self.expiry and time.time() > self.expiry[key]:
                    self.delete(key)
                    return None
                return self.data.get(key)
            
            def delete(self, key):
                self.data.pop(key, None)
                self.expiry.pop(key, None)
                return True
            
            def exists(self, key):
                if key in self.expiry and time.time() > self.expiry[key]:
                    self.delete(key)
                    return False
                return key in self.data
            
            def keys(self, pattern="*"):
                # Simple pattern matching
                if pattern == "*":
                    return list(self.data.keys())
                return [k for k in self.data.keys() if pattern.replace("*", "") in k]
        
        self.logger.warning("Using mock Redis client - configure Redis for production")
        return MockRedis()
    
    def put(self, key: str, value: Any, data_type: str = "market_data", 
            custom_ttl: Optional[int] = None) -> bool:
        """
        Store data in L1 cache with TTL based on volatility.
        Implements dynamic TTL based on data volatility as per documentation.
        """
        try:
            # Determine TTL based on data type
            ttl = custom_ttl or self.default_ttl.get(data_type, 300)
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value, default=str)
            else:
                serialized_value = str(value)
            
            # Store with TTL
            success = self.redis_client.set(key, serialized_value, ex=ttl)
            
            if success:
                self.logger.debug(f"Stored key {key} in L1 cache with TTL {ttl}s")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to store key {key} in L1 cache: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from L1 cache."""
        try:
            self.metrics.total_requests += 1
            
            value = self.redis_client.get(key)
            
            if value is not None:
                self.metrics.hit_count += 1
                
                # Try to deserialize JSON
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                self.metrics.miss_count += 1
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve key {key} from L1 cache: {str(e)}")
            self.metrics.miss_count += 1
            return None
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry."""
        try:
            result = self.redis_client.delete(key)
            if result:
                self.logger.debug(f"Invalidated key {key} from L1 cache")
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate key {key} from L1 cache: {str(e)}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate multiple keys matching pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(f"Invalidated {deleted} keys matching pattern {pattern}")
                return deleted
            return 0
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate pattern {pattern}: {str(e)}")
            return 0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get L1 cache metrics."""
        return {
            "hit_count": self.metrics.hit_count,
            "miss_count": self.metrics.miss_count,
            "total_requests": self.metrics.total_requests,
            "hit_ratio": self.metrics.hit_ratio,
            "miss_ratio": self.metrics.miss_ratio,
            "connection_status": self._check_connection()
        }
    
    def _check_connection(self) -> bool:
        """Check Redis connection status."""
        try:
            self.redis_client.ping()
            return True
        except:
            return False


class CacheL2Memory:
    """
    Cache L2 (Memory Local) - Frequently Accessed Data
    LRU cache for data accessed frequently but not real-time critical
    As specified in Stage 1 documentation
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.access_times = {}
        self.memory_usage = 0
        self.lock = threading.RLock()
        self.metrics = CacheMetrics()
        self.logger = logging.getLogger(f"{__name__}.CacheL2Memory")
    
    def put(self, key: str, value: Any) -> bool:
        """Store data in L2 memory cache with LRU eviction."""
        try:
            with self.lock:
                # Calculate memory usage of new item
                item_size = self._calculate_memory_usage(value)
                
                # Check if item is too large
                if item_size > self.max_memory_bytes:
                    self.logger.warning(f"Item {key} too large for L2 cache: {item_size} bytes")
                    return False
                
                # Remove existing item if updating
                if key in self.cache:
                    old_size = self._calculate_memory_usage(self.cache[key])
                    self.memory_usage -= old_size
                    del self.cache[key]
                
                # Evict items if necessary
                while (len(self.cache) >= self.max_size or 
                       self.memory_usage + item_size > self.max_memory_bytes):
                    if not self.cache:
                        break
                    self._evict_lru_item()
                
                # Store new item
                self.cache[key] = value
                self.access_times[key] = time.time()
                self.memory_usage += item_size
                
                self.logger.debug(f"Stored key {key} in L2 cache ({item_size} bytes)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store key {key} in L2 cache: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from L2 cache with LRU update."""
        try:
            with self.lock:
                self.metrics.total_requests += 1
                
                if key in self.cache:
                    # Update access time and move to end (most recently used)
                    value = self.cache.pop(key)
                    self.cache[key] = value
                    self.access_times[key] = time.time()
                    
                    self.metrics.hit_count += 1
                    return value
                else:
                    self.metrics.miss_count += 1
                    return None
                    
        except Exception as e:
            self.logger.error(f"Failed to retrieve key {key} from L2 cache: {str(e)}")
            self.metrics.miss_count += 1
            return None
    
    def _evict_lru_item(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Remove oldest item (first in OrderedDict)
        key, value = self.cache.popitem(last=False)
        item_size = self._calculate_memory_usage(value)
        self.memory_usage -= item_size
        self.access_times.pop(key, None)
        self.metrics.eviction_count += 1
        
        self.logger.debug(f"Evicted key {key} from L2 cache ({item_size} bytes)")
    
    def _calculate_memory_usage(self, obj: Any) -> int:
        """Estimate memory usage of object."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, (list, tuple)):
                return sum(self._calculate_memory_usage(item) for item in obj)
            elif isinstance(obj, dict):
                return sum(self._calculate_memory_usage(k) + self._calculate_memory_usage(v) 
                          for k, v in obj.items())
            else:
                return 1024  # Default estimate
    
    def invalidate(self, key: str) -> bool:
        """Remove item from L2 cache."""
        try:
            with self.lock:
                if key in self.cache:
                    value = self.cache.pop(key)
                    item_size = self._calculate_memory_usage(value)
                    self.memory_usage -= item_size
                    self.access_times.pop(key, None)
                    
                    self.logger.debug(f"Invalidated key {key} from L2 cache")
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to invalidate key {key} from L2 cache: {str(e)}")
            return False
    
    def clear(self):
        """Clear all items from L2 cache."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.memory_usage = 0
            self.logger.info("Cleared all items from L2 cache")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get L2 cache metrics."""
        with self.lock:
            return {
                "hit_count": self.metrics.hit_count,
                "miss_count": self.metrics.miss_count,
                "eviction_count": self.metrics.eviction_count,
                "total_requests": self.metrics.total_requests,
                "hit_ratio": self.metrics.hit_ratio,
                "miss_ratio": self.metrics.miss_ratio,
                "current_size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": round(self.memory_usage / (1024 * 1024), 2),
                "max_memory_mb": round(self.max_memory_bytes / (1024 * 1024), 2)
            }


class CacheL3S3:
    """
    Cache L3 (S3) - Cold Storage
    Long-term cache with intelligent tiering and compression
    As specified in Stage 1 documentation
    """
    
    def __init__(self, s3_data_lake: S3DataLakeManager, 
                 cache_bucket: str = "cerverus-cache-l3"):
        self.s3_data_lake = s3_data_lake
        self.cache_bucket = cache_bucket
        self.s3_client = s3_data_lake.s3_client
        self.metrics = CacheMetrics()
        self.logger = logging.getLogger(f"{__name__}.CacheL3S3")
        
        # Initialize cache bucket
        self._initialize_cache_bucket()
    
    def _initialize_cache_bucket(self):
        """Initialize S3 cache bucket with intelligent tiering."""
        try:
            # Create bucket if not exists
            try:
                self.s3_client.head_bucket(Bucket=self.cache_bucket)
            except:
                self.s3_client.create_bucket(Bucket=self.cache_bucket)
                self.logger.info(f"Created L3 cache bucket: {self.cache_bucket}")
            
            # Configure intelligent tiering
            intelligent_tiering_config = {
                'Id': 'EntireBucketIntelligentTiering',
                'Status': 'Enabled',
                'Filter': {'Prefix': ''},
                'Tierings': [
                    {
                        'Days': 1,
                        'AccessTier': 'ARCHIVE_ACCESS'
                    },
                    {
                        'Days': 90,
                        'AccessTier': 'DEEP_ARCHIVE_ACCESS'
                    }
                ]
            }
            
            self.s3_client.put_bucket_intelligent_tiering_configuration(
                Bucket=self.cache_bucket,
                Id='EntireBucketIntelligentTiering',
                IntelligentTieringConfiguration=intelligent_tiering_config
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize L3 cache bucket: {str(e)}")
    
    def put(self, key: str, value: Any, compress: bool = True) -> bool:
        """Store data in L3 S3 cache with compression."""
        try:
            # Serialize data
            if isinstance(value, (dict, list)):
                data = json.dumps(value, default=str).encode('utf-8')
            else:
                data = str(value).encode('utf-8')
            
            # Compress data if requested
            if compress:
                data = gzip.compress(data)
                content_encoding = 'gzip'
            else:
                content_encoding = None
            
            # Generate S3 key with timestamp
            s3_key = f"l3_cache/{datetime.now().strftime('%Y/%m/%d')}/{key}"
            
            # Store in S3
            put_kwargs = {
                'Bucket': self.cache_bucket,
                'Key': s3_key,
                'Body': data,
                'Metadata': {
                    'cache_timestamp': str(int(time.time())),
                    'original_key': key,
                    'compressed': str(compress).lower()
                },
                'StorageClass': 'INTELLIGENT_TIERING'
            }
            
            if content_encoding:
                put_kwargs['ContentEncoding'] = content_encoding
            
            self.s3_client.put_object(**put_kwargs)
            
            self.logger.debug(f"Stored key {key} in L3 cache at {s3_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store key {key} in L3 cache: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from L3 S3 cache."""
        try:
            self.metrics.total_requests += 1
            
            # Find the most recent version of the key
            s3_key = self._find_latest_key(key)
            if not s3_key:
                self.metrics.miss_count += 1
                return None
            
            # Get object from S3
            response = self.s3_client.get_object(Bucket=self.cache_bucket, Key=s3_key)
            data = response['Body'].read()
            
            # Decompress if necessary
            metadata = response.get('Metadata', {})
            if metadata.get('compressed', 'false').lower() == 'true':
                data = gzip.decompress(data)
            
            # Deserialize
            try:
                result = json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                result = data.decode('utf-8')
            
            self.metrics.hit_count += 1
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve key {key} from L3 cache: {str(e)}")
            self.metrics.miss_count += 1
            return None
    
    def _find_latest_key(self, key: str) -> Optional[str]:
        """Find the most recent version of a cached key."""
        try:
            # List objects with the key prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.cache_bucket,
                Prefix=f"l3_cache/",
                Delimiter='/'
            )
            
            matching_keys = []
            for obj in response.get('Contents', []):
                if obj['Key'].endswith(f"/{key}"):
                    matching_keys.append((obj['Key'], obj['LastModified']))
            
            if matching_keys:
                # Return the most recently modified key
                latest_key = max(matching_keys, key=lambda x: x[1])[0]
                return latest_key
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find latest key for {key}: {str(e)}")
            return None
    
    def invalidate(self, key: str) -> bool:
        """Remove data from L3 S3 cache."""
        try:
            s3_key = self._find_latest_key(key)
            if s3_key:
                self.s3_client.delete_object(Bucket=self.cache_bucket, Key=s3_key)
                self.logger.debug(f"Invalidated key {key} from L3 cache")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to invalidate key {key} from L3 cache: {str(e)}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get L3 cache metrics."""
        try:
            # Get bucket size
            response = self.s3_client.list_objects_v2(Bucket=self.cache_bucket)
            total_size = sum(obj['Size'] for obj in response.get('Contents', []))
            object_count = len(response.get('Contents', []))
            
            return {
                "hit_count": self.metrics.hit_count,
                "miss_count": self.metrics.miss_count,
                "total_requests": self.metrics.total_requests,
                "hit_ratio": self.metrics.hit_ratio,
                "miss_ratio": self.metrics.miss_ratio,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "object_count": object_count,
                "bucket_name": self.cache_bucket
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get L3 cache metrics: {str(e)}")
            return {"error": str(e)}


class MultilevelCacheManager:
    """
    Multilevel Cache Manager orchestrating L1/L2/L3 caches.
    Implements intelligent cache hierarchy with automatic promotion/demotion.
    """
    
    def __init__(self, 
                 l1_cache: CacheL1Redis,
                 l2_cache: CacheL2Memory,
                 l3_cache: CacheL3S3):
        self.l1_cache = l1_cache
        self.l2_cache = l2_cache
        self.l3_cache = l3_cache
        self.logger = logging.getLogger(f"{__name__}.MultilevelCacheManager")
        
        # Cache hierarchy configuration
        self.promotion_threshold = 3  # Promote to upper level after 3 hits
        self.access_tracking = {}
    
    def get(self, key: str, data_type: str = "market_data") -> Optional[Any]:
        """
        Retrieve data from cache hierarchy (L1 -> L2 -> L3).
        Implements intelligent cache invalidation as per documentation.
        """
        # Try L1 first (Redis - hot data)
        value = self.l1_cache.get(key)
        if value is not None:
            self._track_access(key)
            return value
        
        # Try L2 (Memory - frequently accessed)
        value = self.l2_cache.get(key)
        if value is not None:
            self._track_access(key)
            # Promote to L1 if frequently accessed
            if self._should_promote_to_l1(key):
                self.l1_cache.put(key, value, data_type)
            return value
        
        # Try L3 (S3 - cold storage)
        value = self.l3_cache.get(key)
        if value is not None:
            self._track_access(key)
            # Promote to L2
            self.l2_cache.put(key, value)
            # Promote to L1 if frequently accessed
            if self._should_promote_to_l1(key):
                self.l1_cache.put(key, value, data_type)
            return value
        
        return None
    
    def put(self, key: str, value: Any, data_type: str = "market_data",
            force_level: Optional[int] = None) -> bool:
        """
        Store data in appropriate cache level based on access patterns.
        """
        success = True
        
        if force_level == 1 or force_level is None:
            # Store in L1 (Redis) for hot data
            success &= self.l1_cache.put(key, value, data_type)
        
        if force_level == 2 or (force_level is None and data_type in ["company_info", "regulatory"]):
            # Store in L2 (Memory) for frequently accessed data
            success &= self.l2_cache.put(key, value)
        
        if force_level == 3 or force_level is None:
            # Always store in L3 (S3) for persistence
            success &= self.l3_cache.put(key, value)
        
        return success
    
    def invalidate(self, key: str) -> Dict[str, bool]:
        """Invalidate key from all cache levels."""
        results = {
            "l1": self.l1_cache.invalidate(key),
            "l2": self.l2_cache.invalidate(key),
            "l3": self.l3_cache.invalidate(key)
        }
        
        # Remove from access tracking
        self.access_tracking.pop(key, None)
        
        return results
    
    def invalidate_pattern(self, pattern: str) -> Dict[str, int]:
        """Invalidate keys matching pattern from L1 cache."""
        return {
            "l1": self.l1_cache.invalidate_pattern(pattern),
            "l2": 0,  # L2 doesn't support pattern invalidation
            "l3": 0   # L3 doesn't support pattern invalidation
        }
    
    def _track_access(self, key: str):
        """Track key access for promotion decisions."""
        if key not in self.access_tracking:
            self.access_tracking[key] = {
                "count": 0,
                "last_access": time.time()
            }
        
        self.access_tracking[key]["count"] += 1
        self.access_tracking[key]["last_access"] = time.time()
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if key should be promoted to L1 cache."""
        if key not in self.access_tracking:
            return False
        
        access_info = self.access_tracking[key]
        
        # Promote if accessed frequently in recent time
        recent_threshold = time.time() - 300  # 5 minutes
        return (access_info["count"] >= self.promotion_threshold and
                access_info["last_access"] > recent_threshold)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get metrics from all cache levels."""
        return {
            "timestamp": datetime.now().isoformat(),
            "l1_redis": self.l1_cache.get_metrics(),
            "l2_memory": self.l2_cache.get_metrics(),
            "l3_s3": self.l3_cache.get_metrics(),
            "access_tracking": {
                "tracked_keys": len(self.access_tracking),
                "promotion_threshold": self.promotion_threshold
            },
            "overall_performance": self._calculate_overall_performance()
        }
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall cache system performance."""
        l1_metrics = self.l1_cache.get_metrics()
        l2_metrics = self.l2_cache.get_metrics()
        l3_metrics = self.l3_cache.get_metrics()
        
        total_requests = (l1_metrics["total_requests"] + 
                         l2_metrics["total_requests"] + 
                         l3_metrics["total_requests"])
        
        total_hits = (l1_metrics["hit_count"] + 
                     l2_metrics["hit_count"] + 
                     l3_metrics["hit_count"])
        
        overall_hit_ratio = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "overall_hit_ratio": overall_hit_ratio,
            "overall_miss_ratio": 1.0 - overall_hit_ratio,
            "total_requests": total_requests,
            "total_hits": total_hits
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check of multilevel cache system."""
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "cache_levels": {}
        }
        
        # Check L1 Redis
        l1_healthy = self.l1_cache._check_connection()
        health_status["cache_levels"]["l1_redis"] = {
            "status": "healthy" if l1_healthy else "unhealthy",
            "connection": l1_healthy
        }
        
        # Check L2 Memory (always healthy if process is running)
        health_status["cache_levels"]["l2_memory"] = {
            "status": "healthy",
            "memory_usage_ok": self.l2_cache.memory_usage < self.l2_cache.max_memory_bytes
        }
        
        # Check L3 S3
        try:
            self.l3_cache.s3_client.head_bucket(Bucket=self.l3_cache.cache_bucket)
            l3_healthy = True
        except:
            l3_healthy = False
        
        health_status["cache_levels"]["l3_s3"] = {
            "status": "healthy" if l3_healthy else "unhealthy",
            "bucket_accessible": l3_healthy
        }
        
        # Overall status
        unhealthy_levels = [k for k, v in health_status["cache_levels"].items() 
                           if v["status"] == "unhealthy"]
        
        if len(unhealthy_levels) >= 2:
            health_status["overall_status"] = "critical"
        elif len(unhealthy_levels) == 1:
            health_status["overall_status"] = "degraded"
        
        return health_status