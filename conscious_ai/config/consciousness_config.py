"""
Consciousness Configuration Management
====================================
Centralized configuration for consciousness thresholds and evaluation strategies.
Integrates fixes from temporary fix files into a proper configuration system.
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class EvaluationMode(Enum):
    """Evaluation strategy modes"""
    ML_FIRST = "ml_first"
    SEMANTIC_ONLY = "semantic_only"
    HEURISTIC_ONLY = "heuristic_only"
    HYBRID = "hybrid"


@dataclass
class ConsciousnessThresholds:
    """Consciousness detection thresholds"""
    sensory_activation: float = 0.55
    memory_items_min: int = 3
    metacognitive_thoughts_min: int = 1
    confidence_level: float = 0.55
    contextual_relevance: float = 0.45
    consciousness_metric_f: float = 1.3


@dataclass
class Phase34Config:
    """Phase 3.4 Critical Evaluation Configuration"""
    evaluation_mode: EvaluationMode = EvaluationMode.HYBRID
    max_correction_attempts: int = 3
    temperature_decay: float = 0.3
    use_ml_classifier: bool = True
    fallback_to_semantic: bool = True
    classifier_path: Optional[str] = "./models/coherence_classifier"
    

@dataclass
class ConsciousnessConfig:
    """Main consciousness system configuration"""
    thresholds: ConsciousnessThresholds
    phase34: Phase34Config
    enable_debug_logging: bool = False
    enable_metrics_plotting: bool = True
    max_memory_history: int = 50


class ConfigManager:
    """Manages consciousness configuration with file persistence"""
    
    def __init__(self, config_path: str = "./config/consciousness_config.json"):
        self.config_path = config_path
        self.config = self._load_or_create_default()
    
    def _load_or_create_default(self) -> ConsciousnessConfig:
        """Load config from file or create default"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return self._dict_to_config(data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        return self._create_default_config()
    
    def _create_default_config(self) -> ConsciousnessConfig:
        """Create default configuration with optimized settings"""
        # Apply fixes from quick_fix_phase34.py - use semantic evaluation by default
        phase34_config = Phase34Config(
            evaluation_mode=EvaluationMode.SEMANTIC_ONLY,
            use_ml_classifier=False,
            fallback_to_semantic=True
        )
        
        # Apply consciousness threshold optimizations
        thresholds = ConsciousnessThresholds(
            sensory_activation=0.45,  # Slightly lower for better detection
            consciousness_metric_f=1.2  # More lenient threshold
        )
        
        return ConsciousnessConfig(
            thresholds=thresholds,
            phase34=phase34_config
        )
    
    def _dict_to_config(self, data: Dict[str, Any]) -> ConsciousnessConfig:
        """Convert dictionary to configuration object"""
        thresholds = ConsciousnessThresholds(**data.get('thresholds', {}))
        
        phase34_data = data.get('phase34', {})
        if 'evaluation_mode' in phase34_data:
            phase34_data['evaluation_mode'] = EvaluationMode(phase34_data['evaluation_mode'])
        phase34 = Phase34Config(**phase34_data)
        
        return ConsciousnessConfig(
            thresholds=thresholds,
            phase34=phase34,
            enable_debug_logging=data.get('enable_debug_logging', False),
            enable_metrics_plotting=data.get('enable_metrics_plotting', True),
            max_memory_history=data.get('max_memory_history', 50)
        )
    
    def save_config(self):
        """Save current configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        config_dict = asdict(self.config)
        config_dict['phase34']['evaluation_mode'] = self.config.phase34.evaluation_mode.value
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def update_phase34_mode(self, mode: EvaluationMode):
        """Update Phase 3.4 evaluation mode"""
        self.config.phase34.evaluation_mode = mode
        self.config.phase34.use_ml_classifier = mode != EvaluationMode.SEMANTIC_ONLY
        self.save_config()
    
    def set_semantic_only_mode(self):
        """Set semantic-only evaluation mode (from quick_fix_phase34.py)"""
        self.update_phase34_mode(EvaluationMode.SEMANTIC_ONLY)
        print("Phase 3.4 set to semantic-only evaluation mode")
    
    def enable_ml_classifier(self, classifier_path: str = "./models/coherence_classifier"):
        """Enable ML classifier if available"""
        if os.path.exists(classifier_path):
            self.config.phase34.use_ml_classifier = True
            self.config.phase34.classifier_path = classifier_path
            self.config.phase34.evaluation_mode = EvaluationMode.HYBRID
            self.save_config()
            print(f"ML classifier enabled at {classifier_path}")
            return True
        else:
            print(f"ML classifier not found at {classifier_path}, keeping semantic-only mode")
            return False


# Global configuration instance
_config_manager = None

def get_consciousness_config() -> ConsciousnessConfig:
    """Get global consciousness configuration"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config

def get_config_manager() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager