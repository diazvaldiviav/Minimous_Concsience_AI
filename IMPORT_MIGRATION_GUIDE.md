# Import Migration Guide

## Quick Reference for Updated Import Paths

### Core Components (Simplified)
```python
# NEW: Simplified core imports
from conscious_ai.core import (
    # Phase 1: Perception
    SensoryModule,
    
    # Phase 2: Cognitive Context
    ConsciousState, ConsciousStateHistory, 
    GoalGenerator, AutomaticThoughtGenerator, ActiveMemory,
    
    # Phase 3: Coherent Generation
    StateEvolutionEngine, ConsciousResponseGenerator,
    
    # Phase 3.4: Critical Evaluation
    CriticalStateEvaluator, CoherenceEvaluator,
    
    # Phase 3.5: Narrative Generation
    NarrativeGenerator,
    
    # Autonomous Thinking (FIXED NAMING)
    AutonomousThoughtGenerator, AutonomousConsciousAI, EnhancedAutonomousConsciousAI,
    
    # Shared utilities
    ConsciousnessMetrics, CentralIntegrator, SelfModel, ReentranceModule
)
```

### Configuration System (NEW)
```python
from conscious_ai.config.consciousness_config import (
    get_consciousness_config, get_config_manager, EvaluationMode
)

# Get current configuration
config = get_consciousness_config()

# Use configuration manager for dynamic changes
manager = get_config_manager()
manager.set_semantic_only_mode()  # Addresses Phase 3.4 ML issues
```

### Migration Map

#### Before → After
```python
# OLD PATHS (UPDATE THESE)
from conscious_ai.modules.sensory import SensoryModule
from conscious_ai.autonomus_thinking.autonomous_thinking import AutonomousThoughtGenerator
from conscious_ai.modules.sensibilityAnalisys import sensibilidad_metricas
from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator

# NEW PATHS (USE THESE)
from conscious_ai.core import SensoryModule, AutonomousThoughtGenerator, CriticalStateEvaluator
from conscious_ai.modules.sensitivity_analysis import sensibilidad_metricas
```

## Quick Start with Refactored Structure

```python
# 1. Initialize with simplified imports
from conscious_ai.core import (
    SensoryModule, ActiveMemory, SelfModel, ConsciousState,
    AutonomousThoughtGenerator, CriticalStateEvaluator
)
from conscious_ai.config.consciousness_config import get_consciousness_config

# 2. Get configuration
config = get_consciousness_config()

# 3. Initialize components
sensory = SensoryModule()
memory = ActiveMemory()
autonomous = AutonomousThoughtGenerator()

# 4. Use configuration for thresholds
consciousness_threshold = config.thresholds.consciousness_metric_f
```

## Configuration Examples

### Set Semantic-Only Mode (Recommended for Development)
```python
from conscious_ai.config.consciousness_config import get_config_manager

manager = get_config_manager()
manager.set_semantic_only_mode()
print("Phase 3.4 configured for semantic-only evaluation")
```

### Access Thresholds
```python
from conscious_ai.config.consciousness_config import get_consciousness_config

config = get_consciousness_config()
print(f"Consciousness threshold: {config.thresholds.consciousness_metric_f}")
print(f"Sensory activation: {config.thresholds.sensory_activation}")
```

## Breaking Changes Summary

### Renamed Directories/Files
- `autonomus_thinking/` → `autonomous_thinking/`
- `sensibilityAnalisys.py` → `sensitivity_analysis.py`

### Removed Files
- `modules/sensory.py` (duplicate)
- `phases/p34_critical_evaluation/coherence_evaluator.py` (duplicate)
- All `fix_*.py` files (replaced with configuration system)

### New Structure
- `conscious_ai/core/` - Centralized imports
- `conscious_ai/config/` - Configuration management
- `tests/test_refactored_structure.py` - Validation tests