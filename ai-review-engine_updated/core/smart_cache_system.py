"""
Smart Cache System for AI Phone Review Engine
Implements Redis-like caching with TTL management, invalidation triggers, and cache warming
"""

import asyncio
import json
import logging
import time
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict, OrderedDict
import weakref
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    FIFO = "fifo"         # First In First Out
    RANDOM = "random"     # Random eviction

class CacheEventType(Enum):
    """Cache event types for monitoring"""
    HIT = "hit"
    MISS = "miss"
    SET = "set"
    DELETE = "delete"
    EXPIRE = "expire"
    EVICT = "evict"
    CLEAR = "clear"

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    last_accessed: float
    access_count: int
    size_bytes: int
    tags: List[str]
    dependencies: List[str]
    metadata: Dict[str, Any]

@dataclass
class CacheStats:
    """Cache statistics"""
    total_requests: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    total_size_bytes: int
    entry_count: int
    expired_entries: int
    evicted_entries: int
    last_reset: str

@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size_mb: int = 500
    default_ttl_seconds: int = 3600
    max_entries: int = 10000
    eviction_strategy: CacheStrategy = CacheStrategy.LRU
    enable_persistence: bool = True
    persistence_interval: int = 300  # 5 minutes
    enable_compression: bool = True
    enable_monitoring: bool = True
    warm_cache_on_startup: bool = True

class CacheInvalidationTrigger:
    """Handles cache invalidation based on various triggers"""
    
    def __init__(self):
        self.triggers = defaultdict(list)  # key pattern -> list of callbacks
        self.time_triggers = []  # scheduled invalidations
        
    def add_pattern_trigger(self, pattern: str, callback: Callable):
        """Add invalidation trigger for key pattern"""
        self.triggers[pattern].append(callback)
    
    def add_time_trigger(self, trigger_time: datetime, keys: List[str]):
        """Schedule invalidation at specific time"""
        self.time_triggers.append({
            'time': trigger_time,
            'keys': keys,
            'executed': False
        })
    
    def add_dependency_trigger(self, dependency_key: str, dependent_keys: List[str]):
        """Add dependency-based invalidation"""
        def invalidate_dependents():
            return dependent_keys
        
        self.add_pattern_trigger(dependency_key, invalidate_dependents)
    
    def check_triggers(self, key: str) -> List[str]:
        """Check if key change should trigger invalidations"""
        keys_to_invalidate = []
        
        # Check pattern triggers
        for pattern, callbacks in self.triggers.items():
            if self._matches_pattern(key, pattern):
                for callback in callbacks:
                    try:
                        keys = callback()
                        if keys:
                            keys_to_invalidate.extend(keys)
                    except Exception as e:
                        logger.error(f"Error executing invalidation trigger: {e}")
        
        # Check time triggers
        current_time = datetime.now()
        for trigger in self.time_triggers:
            if not trigger['executed'] and current_time >= trigger['time']:
                keys_to_invalidate.extend(trigger['keys'])
                trigger['executed'] = True
        
        return list(set(keys_to_invalidate))  # Remove duplicates
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard matching)"""
        if '*' not in pattern:
            return key == pattern
        
        # Convert pattern to regex-like matching
        import re
        regex_pattern = pattern.replace('*', '.*')
        return bool(re.match(f'^{regex_pattern}$', key))

class SmartCacheSystem:
    """Advanced caching system with intelligent features"""
    
    def __init__(self, config: CacheConfig = None):
        """Initialize the smart cache system"""
        
        self.config = config or CacheConfig()
        self.cache_store = OrderedDict()  # Main cache storage
        self.stats = CacheStats(0, 0, 0, 0.0, 0, 0, 0, 0, datetime.now().isoformat())
        self.lock = threading.RLock()  # Thread-safe operations
        
        # Invalidation system
        self.invalidation_triggers = CacheInvalidationTrigger()
        
        # Monitoring and events
        self.event_listeners = defaultdict(list)
        self.access_patterns = defaultdict(list)  # For smart prefetching
        
        # Background tasks
        self.cleanup_task = None
        self.persistence_task = None
        self.running = False
        
        # Persistence
        self.persistence_path = Path("cache/smart_cache.pkl")
        
        # Compression
        if self.config.enable_compression:
            try:
                import lz4.frame
                self.compressor = lz4.frame
            except ImportError:
                logger.warning("lz4 not available, disabling compression")
                self.config.enable_compression = False
        
        # Initialize
        self.start()
    
    def start(self):
        """Start background tasks"""
        if self.running:
            return
            
        self.running = True
        
        # Load persisted cache if enabled
        if self.config.enable_persistence and self.config.warm_cache_on_startup:
            self._load_cache_from_disk()
        
        # Start background cleanup task
        self.cleanup_task = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_task.start()
        
        # Start persistence task
        if self.config.enable_persistence:
            self.persistence_task = threading.Thread(target=self._persistence_worker, daemon=True)
            self.persistence_task.start()
    
    def stop(self):
        """Stop background tasks and save cache"""
        self.running = False
        
        if self.config.enable_persistence:
            self._save_cache_to_disk()
        
        if self.cleanup_task:
            self.cleanup_task.join(timeout=5)
        
        if self.persistence_task:
            self.persistence_task.join(timeout=5)
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        with self.lock:
            self.stats.total_requests += 1
            
            if key not in self.cache_store:
                self.stats.cache_misses += 1
                self._emit_event(CacheEventType.MISS, key)
                self._record_access_pattern(key, False)
                return default
            
            entry = self.cache_store[key]
            
            # Check expiration
            if self._is_expired(entry):
                del self.cache_store[key]
                self.stats.expired_entries += 1
                self.stats.cache_misses += 1
                self._emit_event(CacheEventType.EXPIRE, key)
                return default
            
            # Update access metadata
            entry.last_accessed = time.time()
            entry.access_count += 1
            
            # Move to end for LRU
            if self.config.eviction_strategy == CacheStrategy.LRU:
                self.cache_store.move_to_end(key)
            
            self.stats.cache_hits += 1
            self._emit_event(CacheEventType.HIT, key)
            self._record_access_pattern(key, True)
            self._update_hit_rate()
            
            # Decompress if needed
            value = self._decompress_value(entry.value)
            
            return value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, 
                 tags: List[str] = None, dependencies: List[str] = None,
                 metadata: Dict[str, Any] = None) -> bool:
        """Set value in cache"""
        with self.lock:
            try:
                # Compress value if enabled
                compressed_value = self._compress_value(value)
                
                # Calculate size
                size_bytes = len(pickle.dumps(compressed_value))
                
                # Check if we need to make space
                if not self._ensure_space(size_bytes):
                    logger.warning(f"Could not make space for key: {key}")
                    return False
                
                # Calculate expiration
                expires_at = None
                if ttl or self.config.default_ttl_seconds:
                    ttl_seconds = ttl or self.config.default_ttl_seconds
                    expires_at = time.time() + ttl_seconds
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    created_at=time.time(),
                    expires_at=expires_at,
                    last_accessed=time.time(),
                    access_count=1,
                    size_bytes=size_bytes,
                    tags=tags or [],
                    dependencies=dependencies or [],
                    metadata=metadata or {}
                )
                
                # Store entry
                self.cache_store[key] = entry
                self.stats.total_size_bytes += size_bytes
                self.stats.entry_count += 1
                
                # Check for invalidation triggers
                keys_to_invalidate = self.invalidation_triggers.check_triggers(key)
                for invalid_key in keys_to_invalidate:
                    await self.delete(invalid_key)
                
                self._emit_event(CacheEventType.SET, key, {'size': size_bytes, 'ttl': ttl})
                
                return True
                
            except Exception as e:
                logger.error(f"Error setting cache key {key}: {e}")
                return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        with self.lock:
            if key in self.cache_store:
                entry = self.cache_store[key]
                self.stats.total_size_bytes -= entry.size_bytes
                self.stats.entry_count -= 1
                del self.cache_store[key]
                self._emit_event(CacheEventType.DELETE, key)
                return True
            return False
    
    async def clear(self, pattern: Optional[str] = None, tags: Optional[List[str]] = None):
        """Clear cache entries matching pattern or tags"""
        with self.lock:
            keys_to_delete = []
            
            for key, entry in self.cache_store.items():
                should_delete = False
                
                if pattern and self.invalidation_triggers._matches_pattern(key, pattern):
                    should_delete = True
                elif tags and any(tag in entry.tags for tag in tags):
                    should_delete = True
                elif not pattern and not tags:  # Clear all
                    should_delete = True
                
                if should_delete:
                    keys_to_delete.append(key)
            
            # Delete matched keys
            for key in keys_to_delete:
                await self.delete(key)
            
            if not pattern and not tags:
                # Full clear
                self.cache_store.clear()
                self.stats.total_size_bytes = 0
                self.stats.entry_count = 0
            
            self._emit_event(CacheEventType.CLEAR, None, {'keys_cleared': len(keys_to_delete)})
    
    async def get_multi(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache"""
        results = {}
        for key in keys:
            value = await self.get(key)
            if value is not None:
                results[key] = value
        return results
    
    async def set_multi(self, key_value_pairs: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """Set multiple values in cache"""
        results = {}
        for key, value in key_value_pairs.items():
            results[key] = await self.set(key, value, ttl)
        return results
    
    async def increment(self, key: str, delta: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment numeric value in cache"""
        with self.lock:
            current = await self.get(key, 0)
            if not isinstance(current, (int, float)):
                return None
            
            new_value = int(current) + delta
            success = await self.set(key, new_value, ttl)
            return new_value if success else None
    
    async def expire_at(self, key: str, expire_time: datetime) -> bool:
        """Set expiration time for key"""
        with self.lock:
            if key in self.cache_store:
                entry = self.cache_store[key]
                entry.expires_at = expire_time.timestamp()
                return True
            return False
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            self.stats.hit_rate = (
                self.stats.cache_hits / max(1, self.stats.total_requests) * 100
            )
            return self.stats
    
    def reset_stats(self):
        """Reset cache statistics"""
        with self.lock:
            self.stats = CacheStats(0, 0, 0, 0.0, self.stats.total_size_bytes, 
                                  self.stats.entry_count, 0, 0, datetime.now().isoformat())
    
    # Smart features
    async def warm_cache(self, keys_and_loaders: Dict[str, Callable]) -> Dict[str, bool]:
        """Warm cache with data from loaders"""
        results = {}
        
        for key, loader in keys_and_loaders.items():
            try:
                # Check if already cached and fresh
                existing = await self.get(key)
                if existing is not None:
                    results[key] = True
                    continue
                
                # Load data
                value = await loader() if asyncio.iscoroutinefunction(loader) else loader()
                results[key] = await self.set(key, value)
                
            except Exception as e:
                logger.error(f"Error warming cache for key {key}: {e}")
                results[key] = False
        
        return results
    
    def add_invalidation_trigger(self, trigger_pattern: str, dependent_keys: List[str]):
        """Add cache invalidation trigger"""
        def invalidate():
            return dependent_keys
        
        self.invalidation_triggers.add_pattern_trigger(trigger_pattern, invalidate)
    
    def add_event_listener(self, event_type: CacheEventType, callback: Callable):
        """Add event listener for cache operations"""
        self.event_listeners[event_type].append(callback)
    
    def get_access_patterns(self) -> Dict[str, List[Dict]]:
        """Get access patterns for analysis"""
        with self.lock:
            return dict(self.access_patterns)
    
    def suggest_prefetch_keys(self, limit: int = 10) -> List[str]:
        """Suggest keys for prefetching based on access patterns"""
        suggestions = []
        
        with self.lock:
            # Analyze access patterns to find frequently accessed keys that might be missing
            for key, accesses in self.access_patterns.items():
                if len(accesses) < 3:  # Need some history
                    continue
                
                recent_misses = sum(1 for access in accesses[-10:] if not access['hit'])
                if recent_misses > 3 and key not in self.cache_store:
                    suggestions.append(key)
        
        return suggestions[:limit]
    
    # Internal methods
    def _ensure_space(self, needed_bytes: int) -> bool:
        """Ensure enough space in cache"""
        max_bytes = self.config.max_size_mb * 1024 * 1024
        
        # Check if we're within limits
        if (self.stats.total_size_bytes + needed_bytes <= max_bytes and 
            self.stats.entry_count < self.config.max_entries):
            return True
        
        # Need to evict entries
        return self._evict_entries(needed_bytes)
    
    def _evict_entries(self, needed_bytes: int) -> bool:
        """Evict entries based on strategy"""
        evicted_bytes = 0
        evicted_count = 0
        max_bytes = self.config.max_size_mb * 1024 * 1024
        
        if self.config.eviction_strategy == CacheStrategy.LRU:
            # Remove least recently used entries
            while (self.stats.total_size_bytes + needed_bytes > max_bytes or 
                   self.stats.entry_count >= self.config.max_entries):
                if not self.cache_store:
                    break
                
                # Get oldest entry (first in OrderedDict for LRU)
                key, entry = next(iter(self.cache_store.items()))
                evicted_bytes += entry.size_bytes
                evicted_count += 1
                del self.cache_store[key]
                
                self._emit_event(CacheEventType.EVICT, key, {'strategy': 'lru'})
                
                if evicted_count >= 100:  # Prevent infinite loop
                    break
        
        elif self.config.eviction_strategy == CacheStrategy.TTL:
            # Remove expired entries first
            expired_keys = []
            for key, entry in self.cache_store.items():
                if self._is_expired(entry):
                    expired_keys.append(key)
            
            for key in expired_keys:
                entry = self.cache_store[key]
                evicted_bytes += entry.size_bytes
                evicted_count += 1
                del self.cache_store[key]
                
                self._emit_event(CacheEventType.EXPIRE, key)
        
        elif self.config.eviction_strategy == CacheStrategy.LFU:
            # Remove least frequently used entries
            entries_by_frequency = sorted(
                self.cache_store.items(),
                key=lambda x: x[1].access_count
            )
            
            for key, entry in entries_by_frequency:
                if (self.stats.total_size_bytes + needed_bytes <= max_bytes and 
                    self.stats.entry_count < self.config.max_entries):
                    break
                
                evicted_bytes += entry.size_bytes
                evicted_count += 1
                del self.cache_store[key]
                
                self._emit_event(CacheEventType.EVICT, key, {'strategy': 'lfu'})
        
        # Update stats
        self.stats.total_size_bytes -= evicted_bytes
        self.stats.entry_count -= evicted_count
        self.stats.evicted_entries += evicted_count
        
        return evicted_bytes >= needed_bytes
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        if entry.expires_at is None:
            return False
        return time.time() > entry.expires_at
    
    def _compress_value(self, value: Any) -> Any:
        """Compress value if compression is enabled"""
        if not self.config.enable_compression:
            return value
        
        try:
            serialized = pickle.dumps(value)
            if len(serialized) > 1024:  # Only compress larger values
                compressed = self.compressor.compress(serialized)
                return {'_compressed': True, 'data': compressed}
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        
        return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Decompress value if it was compressed"""
        if (isinstance(value, dict) and 
            value.get('_compressed') and 
            self.config.enable_compression):
            
            try:
                decompressed = self.compressor.decompress(value['data'])
                return pickle.loads(decompressed)
            except Exception as e:
                logger.error(f"Decompression failed: {e}")
        
        return value
    
    def _emit_event(self, event_type: CacheEventType, key: Optional[str], 
                   data: Optional[Dict] = None):
        """Emit cache event to listeners"""
        if not self.config.enable_monitoring:
            return
        
        event_data = {
            'timestamp': time.time(),
            'key': key,
            'data': data or {}
        }
        
        for callback in self.event_listeners.get(event_type, []):
            try:
                callback(event_data)
            except Exception as e:
                logger.error(f"Error in event listener: {e}")
    
    def _record_access_pattern(self, key: str, hit: bool):
        """Record access pattern for analysis"""
        if not self.config.enable_monitoring:
            return
        
        pattern = {
            'timestamp': time.time(),
            'hit': hit
        }
        
        self.access_patterns[key].append(pattern)
        
        # Keep only recent patterns (last 100 accesses per key)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def _update_hit_rate(self):
        """Update hit rate calculation"""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = (self.stats.cache_hits / self.stats.total_requests) * 100
    
    def _cleanup_worker(self):
        """Background worker to clean up expired entries"""
        while self.running:
            try:
                with self.lock:
                    expired_keys = []
                    for key, entry in self.cache_store.items():
                        if self._is_expired(entry):
                            expired_keys.append(key)
                    
                    # Remove expired entries
                    for key in expired_keys:
                        entry = self.cache_store[key]
                        self.stats.total_size_bytes -= entry.size_bytes
                        self.stats.entry_count -= 1
                        self.stats.expired_entries += 1
                        del self.cache_store[key]
                        
                        self._emit_event(CacheEventType.EXPIRE, key)
                
                # Sleep for 30 seconds before next cleanup
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in cleanup worker: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _persistence_worker(self):
        """Background worker to persist cache to disk"""
        while self.running:
            try:
                time.sleep(self.config.persistence_interval)
                self._save_cache_to_disk()
                
            except Exception as e:
                logger.error(f"Error in persistence worker: {e}")
    
    def _save_cache_to_disk(self):
        """Save cache to disk"""
        try:
            self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save only non-expired entries
            cache_data = {}
            with self.lock:
                for key, entry in self.cache_store.items():
                    if not self._is_expired(entry):
                        cache_data[key] = {
                            'value': entry.value,
                            'created_at': entry.created_at,
                            'expires_at': entry.expires_at,
                            'tags': entry.tags,
                            'dependencies': entry.dependencies,
                            'metadata': entry.metadata
                        }
            
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Saved {len(cache_data)} cache entries to disk")
            
        except Exception as e:
            logger.error(f"Error saving cache to disk: {e}")
    
    def _load_cache_from_disk(self):
        """Load cache from disk"""
        try:
            if not self.persistence_path.exists():
                return
            
            with open(self.persistence_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            loaded_count = 0
            current_time = time.time()
            
            with self.lock:
                for key, data in cache_data.items():
                    # Skip expired entries
                    if data.get('expires_at') and current_time > data['expires_at']:
                        continue
                    
                    # Create cache entry
                    size_bytes = len(pickle.dumps(data['value']))
                    entry = CacheEntry(
                        key=key,
                        value=data['value'],
                        created_at=data.get('created_at', current_time),
                        expires_at=data.get('expires_at'),
                        last_accessed=current_time,
                        access_count=1,
                        size_bytes=size_bytes,
                        tags=data.get('tags', []),
                        dependencies=data.get('dependencies', []),
                        metadata=data.get('metadata', {})
                    )
                    
                    self.cache_store[key] = entry
                    self.stats.total_size_bytes += size_bytes
                    self.stats.entry_count += 1
                    loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} cache entries from disk")
            
        except Exception as e:
            logger.error(f"Error loading cache from disk: {e}")

# Factory functions and utilities
def create_smart_cache_system(config: CacheConfig = None) -> SmartCacheSystem:
    """Create configured smart cache system"""
    return SmartCacheSystem(config or CacheConfig())

def create_phone_review_cache() -> SmartCacheSystem:
    """Create cache system optimized for phone review data"""
    config = CacheConfig(
        max_size_mb=1000,           # 1GB for phone data
        default_ttl_seconds=3600,   # 1 hour default
        max_entries=50000,          # Large number of entries
        eviction_strategy=CacheStrategy.LRU,
        enable_persistence=True,
        persistence_interval=600,   # 10 minutes
        enable_compression=True,
        enable_monitoring=True,
        warm_cache_on_startup=True
    )
    
    cache = SmartCacheSystem(config)
    
    # Add phone-specific invalidation triggers
    cache.add_invalidation_trigger("phone:*:price", ["phone:*:summary", "pricing:*"])
    cache.add_invalidation_trigger("phone:*:reviews", ["phone:*:summary", "reviews:*"])
    cache.add_invalidation_trigger("phone:*:specs", ["phone:*:summary", "specifications:*"])
    
    return cache

# Cache decorators
def cached(cache_system: SmartCacheSystem, ttl: int = 3600, key_prefix: str = ""):
    """Decorator to cache function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__] if key_prefix else [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = await cache_system.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            await cache_system.set(cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions, we need to use asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            finally:
                loop.close()
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Context manager for cache transactions
class CacheTransaction:
    """Context manager for atomic cache operations"""
    
    def __init__(self, cache_system: SmartCacheSystem):
        self.cache_system = cache_system
        self.operations = []
        self.committed = False
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if not self.committed and exc_type is None:
            await self.commit()
        elif exc_type is not None:
            await self.rollback()
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Queue set operation"""
        self.operations.append(('set', key, value, ttl))
    
    async def delete(self, key: str):
        """Queue delete operation"""
        self.operations.append(('delete', key))
    
    async def commit(self):
        """Execute all queued operations"""
        for op in self.operations:
            if op[0] == 'set':
                await self.cache_system.set(op[1], op[2], op[3])
            elif op[0] == 'delete':
                await self.cache_system.delete(op[1])
        
        self.committed = True
        self.operations.clear()
    
    async def rollback(self):
        """Cancel all queued operations"""
        self.operations.clear()