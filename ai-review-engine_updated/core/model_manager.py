"""
Centralized Model Manager for efficient model loading and caching
Implements singleton pattern with lazy loading
"""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import pickle
import hashlib
import json
from datetime import datetime, timedelta
import threading
import gc
import psutil
import os

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton Model Manager for centralized model loading and caching
    Features:
    - Lazy loading of models
    - Memory-efficient caching
    - Model versioning
    - Automatic cleanup of unused models
    """
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _model_metadata: Dict[str, Dict] = {}
    _cache_dir = Path("cache/models")
    
    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model manager"""
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
        self._setup_monitoring()
        logger.info("ModelManager initialized")
    
    def _load_metadata(self):
        """Load model metadata from cache"""
        metadata_file = self._cache_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self._model_metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata: {e}")
                self._model_metadata = {}
    
    def _save_metadata(self):
        """Save model metadata to cache"""
        metadata_file = self._cache_dir / "metadata.json"
        try:
            with open(metadata_file, 'w') as f:
                json.dump(self._model_metadata, f, default=str)
        except Exception as e:
            logger.error(f"Could not save metadata: {e}")
    
    def _setup_monitoring(self):
        """Setup memory monitoring for automatic cleanup"""
        self._memory_threshold = 0.8  # 80% memory usage triggers cleanup
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=5)
    
    def get_model(self, model_name: str, model_config: Optional[Dict] = None) -> Any:
        """
        Get a model by name, loading it if necessary
        
        Args:
            model_name: Name/identifier of the model
            model_config: Optional configuration for loading the model
            
        Returns:
            The requested model instance
        """
        # Check memory and cleanup if needed
        self._check_memory_usage()
        
        # Return cached model if available
        if model_name in self._models:
            self._update_usage_stats(model_name)
            logger.debug(f"Returning cached model: {model_name}")
            return self._models[model_name]
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        model = self._load_model(model_name, model_config)
        
        if model:
            self._models[model_name] = model
            self._update_metadata(model_name, model_config)
            self._update_usage_stats(model_name)
        
        return model
    
    def _load_model(self, model_name: str, config: Optional[Dict] = None) -> Any:
        """
        Load a specific model based on its name
        
        Args:
            model_name: Name of the model to load
            config: Optional configuration
            
        Returns:
            Loaded model instance
        """
        try:
            # Check for cached model file
            cache_file = self._cache_dir / f"{model_name}.pkl"
            if cache_file.exists() and self._is_cache_valid(model_name):
                logger.info(f"Loading {model_name} from cache")
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            
            # Load based on model type
            if 'transformer' in model_name.lower():
                return self._load_transformer_model(model_name, config)
            elif 'vader' in model_name.lower():
                return self._load_vader_model()
            elif 'spacy' in model_name.lower():
                return self._load_spacy_model(config)
            elif 'sklearn' in model_name.lower():
                return self._load_sklearn_model(model_name, config)
            else:
                logger.warning(f"Unknown model type: {model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def _load_transformer_model(self, model_name: str, config: Optional[Dict] = None) -> Any:
        """Load a transformer model"""
        try:
            from transformers import AutoModel, AutoTokenizer, pipeline
            
            config = config or {}
            model_path = config.get('model_path', model_name)
            
            # Determine model type and load accordingly
            if 'sentiment' in model_name.lower():
                model = pipeline(
                    "sentiment-analysis",
                    model=model_path,
                    device=-1  # CPU by default
                )
            elif 'ner' in model_name.lower():
                model = pipeline(
                    "ner",
                    model=model_path,
                    aggregation_strategy="simple",
                    device=-1
                )
            elif 'zero-shot' in model_name.lower():
                model = pipeline(
                    "zero-shot-classification",
                    model=model_path,
                    device=-1
                )
            else:
                # Generic transformer model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
                model = {'tokenizer': tokenizer, 'model': model}
            
            # Cache the model
            self._cache_model(model_name, model)
            return model
            
        except ImportError:
            logger.warning(f"Transformers not available for {model_name}")
            return None
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            return None
    
    def _load_vader_model(self) -> Any:
        """Load VADER sentiment analyzer"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("VADER not available")
            return None
    
    def _load_spacy_model(self, config: Optional[Dict] = None) -> Any:
        """Load spaCy model"""
        try:
            import spacy
            model_name = config.get('spacy_model', 'en_core_web_sm') if config else 'en_core_web_sm'
            return spacy.load(model_name)
        except ImportError:
            logger.warning("spaCy not available")
            return None
        except OSError:
            logger.warning(f"spaCy model not found, downloading...")
            try:
                import subprocess
                subprocess.run(['python', '-m', 'spacy', 'download', model_name])
                return spacy.load(model_name)
            except:
                return None
    
    def _load_sklearn_model(self, model_name: str, config: Optional[Dict] = None) -> Any:
        """Load scikit-learn model"""
        try:
            if 'random_forest' in model_name.lower():
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(**(config or {}))
            elif 'gradient_boost' in model_name.lower():
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(**(config or {}))
            elif 'mlp' in model_name.lower():
                from sklearn.neural_network import MLPClassifier
                return MLPClassifier(**(config or {}))
            else:
                logger.warning(f"Unknown sklearn model: {model_name}")
                return None
        except ImportError:
            logger.warning("scikit-learn not available")
            return None
    
    def _cache_model(self, model_name: str, model: Any):
        """Cache a model to disk"""
        try:
            cache_file = self._cache_dir / f"{model_name}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Cached model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not cache model {model_name}: {e}")
    
    def _is_cache_valid(self, model_name: str) -> bool:
        """Check if cached model is still valid"""
        if model_name not in self._model_metadata:
            return False
        
        metadata = self._model_metadata[model_name]
        cache_time = datetime.fromisoformat(metadata.get('cached_at', '2000-01-01'))
        cache_duration = datetime.now() - cache_time
        
        # Cache is valid for 7 days
        return cache_duration < timedelta(days=7)
    
    def _update_metadata(self, model_name: str, config: Optional[Dict] = None):
        """Update model metadata"""
        self._model_metadata[model_name] = {
            'loaded_at': datetime.now().isoformat(),
            'cached_at': datetime.now().isoformat(),
            'config': config,
            'usage_count': 0,
            'last_used': datetime.now().isoformat()
        }
        self._save_metadata()
    
    def _update_usage_stats(self, model_name: str):
        """Update usage statistics for a model"""
        if model_name in self._model_metadata:
            self._model_metadata[model_name]['usage_count'] += 1
            self._model_metadata[model_name]['last_used'] = datetime.now().isoformat()
    
    def _check_memory_usage(self):
        """Check memory usage and cleanup if necessary"""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            
            if memory_percent > self._memory_threshold:
                logger.warning(f"High memory usage: {memory_percent:.1%}")
                self._cleanup_unused_models()
            
            # Periodic cleanup
            if datetime.now() - self._last_cleanup > self._cleanup_interval:
                self._cleanup_unused_models()
                self._last_cleanup = datetime.now()
                
        except Exception as e:
            logger.error(f"Error checking memory: {e}")
    
    def _cleanup_unused_models(self):
        """Remove least recently used models from memory"""
        if len(self._models) <= 1:
            return
        
        # Sort models by last usage
        model_usage = []
        for name, metadata in self._model_metadata.items():
            if name in self._models:
                last_used = datetime.fromisoformat(metadata.get('last_used', '2000-01-01'))
                model_usage.append((name, last_used))
        
        model_usage.sort(key=lambda x: x[1])
        
        # Remove oldest 25% of models
        num_to_remove = max(1, len(model_usage) // 4)
        for name, _ in model_usage[:num_to_remove]:
            logger.info(f"Removing model from memory: {name}")
            del self._models[name]
        
        # Force garbage collection
        gc.collect()
    
    def preload_models(self, model_names: List[str]):
        """
        Preload a list of models for better performance
        
        Args:
            model_names: List of model names to preload
        """
        logger.info(f"Preloading {len(model_names)} models...")
        for name in model_names:
            self.get_model(name)
        logger.info("Preloading complete")
    
    def clear_cache(self):
        """Clear all cached models"""
        logger.info("Clearing model cache...")
        self._models.clear()
        self._model_metadata.clear()
        
        # Clear cache files
        for cache_file in self._cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Could not delete {cache_file}: {e}")
        
        self._save_metadata()
        gc.collect()
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get statistics about loaded models"""
        return {
            'loaded_models': list(self._models.keys()),
            'total_models': len(self._models),
            'cache_size': sum(f.stat().st_size for f in self._cache_dir.glob("*.pkl")),
            'memory_usage': psutil.virtual_memory().percent,
            'metadata': self._model_metadata
        }


# Singleton instance getter
def get_model_manager() -> ModelManager:
    """Get the singleton ModelManager instance"""
    return ModelManager()
