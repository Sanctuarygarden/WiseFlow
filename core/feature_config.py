import json
import os
from typing import Dict, Any, Optional

class FeatureConfig:
    """
    Feature flags for incremental deployment and A/B testing
    
    This component:
    1. Manages feature flags for gradual component activation
    2. Supports loading configuration from file or environment
    3. Enables A/B testing of different components
    4. Provides centralized control of system features
    """
    
    DEFAULT_CONFIG = {
        "context_consolidation": True,
        "proactive_evolution": True,
        "logical_coherence": True,
        "feedback_execution": True,
        "concept_recall": True,
        "opro_optimization": True,
        "tree_of_thought": True,
        "graph_of_thought": True,
        "pal_generation": True
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature configuration
        
        Args:
            config_path: Optional path to JSON configuration file
        """
        self.features = self.DEFAULT_CONFIG.copy()
        
        # Try to load from file if provided
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
            
        # Override with environment variables if present
        self._load_from_environment()
        
    def is_enabled(self, feature_name: str) -> bool:
        """
        Check if a feature is enabled
        
        Args:
            feature_name: Name of the feature to check
            
        Returns:
            True if feature is enabled, False otherwise
        """
        return self.features.get(feature_name, False)
        
    def enable_feature(self, feature_name: str) -> None:
        """Enable a specific feature"""
        if feature_name in self.features:
            self.features[feature_name] = True
            
    def disable_feature(self, feature_name: str) -> None:
        """Disable a specific feature"""
        if feature_name in self.features:
            self.features[feature_name] = False
            
    def get_all_features(self) -> Dict[str, bool]:
        """Get the status of all features"""
        return self.features.copy()
        
    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Update only keys that exist in DEFAULT_CONFIG
            for key in self.DEFAULT_CONFIG:
                if key in config:
                    self.features[key] = bool(config[key])
        except Exception as e:
            print(f"Error loading feature configuration from {config_path}: {e}")
            
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables"""
        prefix = "FEATURE_"
        
        for key in self.DEFAULT_CONFIG:
            env_key = f"{prefix}{key.upper()}"
            if env_key in os.environ:
                self.features[key] = os.environ[env_key].lower() in ('true', '1', 't', 'yes')
