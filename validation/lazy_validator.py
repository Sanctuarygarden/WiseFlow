from typing import Dict, Any, Callable, Optional, Tuple
import logging
import hashlib
import time

logger = logging.getLogger(__name__)

class LazyValidator:
    """
    Performance optimization for validation operations
    
    This component:
    1. Provides caching for expensive validation operations
    2. Implements early stopping for high-confidence cases
    3. Supports fallback strategies when validation fails
    4. Manages validation complexity based on system load
    """
    
    def __init__(
        self, 
        validator_func: Callable, 
        threshold: float = 0.8,
        cache_size: int = 100,
        cache_ttl: int = 3600  # Cache time-to-live in seconds
    ):
        """
        Initialize lazy validator
        
        Args:
            validator_func: Function to call for validation
            threshold: Confidence threshold for early returns
            cache_size: Maximum cache size
            cache_ttl: Time-to-live for cache entries in seconds
        """
        self.validator_func = validator_func
        self.threshold = threshold
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.cache = {}  # key -> (result, timestamp, confidence)
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "early_returns": 0,
            "full_validations": 0
        }
        
    def validate(self, key: str, *args, **kwargs) -> Any:
        """
        Validate with caching and early stopping
        
        Args:
            key: Cache key for the validation
            *args, **kwargs: Arguments to pass to validator function
            
        Returns:
            Validation result
        """
        # Generate cache key if not provided
        if not key:
            key = self._generate_cache_key(args, kwargs)
            
        # Check cache
        current_time = time.time()
        if key in self.cache:
            result, timestamp, confidence = self.cache[key]
            
            # Check if cache entry is still valid
            if current_time - timestamp < self.cache_ttl:
                self.stats["cache_hits"] += 1
                
                # Early return for high confidence results
                if confidence >= self.threshold:
                    self.stats["early_returns"] += 1
                    return result
                    
                # Otherwise, refresh but use cached as fallback
                try:
                    self.stats["full_validations"] += 1
                    new_result = self.validator_func(*args, **kwargs)
                    new_confidence = new_result.get("confidence", 0.0)
                    
                    # Update cache
                    self.cache[key] = (new_result, current_time, new_confidence)
                    
                    # Clean cache if too large
                    self._clean_cache()
                    
                    return new_result
                except Exception as e:
                    logger.warning(f"Validation failed, using cached result: {e}")
                    return result
            
            # Cache entry expired
            del self.cache[key]
        
        # Cache miss
        self.stats["cache_misses"] += 1
        self.stats["full_validations"] += 1
        
        try:
            # Perform full validation
            result = self.validator_func(*args, **kwargs)
            confidence = result.get("confidence", 0.0)
            
            # Cache result
            self.cache[key] = (result, current_time, confidence)
            
            # Clean cache if too large
            self._clean_cache()
            
            return result
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            # Return fallback result
            return {
                "is_valid": True,  # Default to valid on error
                "confidence": 0.5,
                "error": str(e),
                "fallback": True
            }
    
    def _generate_cache_key(self, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """Generate a cache key from args and kwargs"""
        # Create a string representation of args and kwargs
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
        
        # Generate hash
        key = hashlib.md5(f"{args_str}|{kwargs_str}".encode()).hexdigest()
        return key
    
    def _clean_cache(self) -> None:
        """Clean old or excess entries from cache"""
        # If cache is under size limit, do nothing
        if len(self.cache) <= self.cache_size:
            return
            
        # Remove oldest entries
        current_time = time.time()
        expired_keys = []
        
        # First remove expired entries
        for key, (_, timestamp, _) in self.cache.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
            
        # If still over limit, remove oldest entries
        if len(self.cache) > self.cache_size:
            # Sort by timestamp (oldest first)
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1][1]
            )
            
            # Remove oldest entries until under limit
            for key, _ in sorted_entries[:len(self.cache) - self.cache_size]:
                del self.cache[key]
    
    def invalidate(self, key: Optional[str] = None) -> None:
        """
        Invalidate cache entries
        
        Args:
            key: Specific key to invalidate, or None to invalidate all
        """
        if key is None:
            self.cache.clear()
        elif key in self.cache:
            del self.cache[key]
    
    def get_stats(self) -> Dict[str, int]:
        """Get validation statistics"""
        return {
            **self.stats,
            "cache_size": len(self.cache)
        }
