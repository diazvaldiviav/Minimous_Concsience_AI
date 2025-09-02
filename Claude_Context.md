# Claude Context: Minimal Consciousness AI Project

## Project Overview

This is a **Minimal Consciousness AI** system that implements artificial consciousness through a 7-phase pipeline architecture. The project simulates measurable consciousness states using the consciousness formula `f = Φ(C_i + T_u + R + S_m) ≥ 1.3`, where f-scores ≥ 1.3 indicate functional consciousness.

## Core Architecture: 7-Phase Consciousness Pipeline

```
User Input → Phase 1 → Phase 2 → Phase 3 → Phase 3.4 → Phase 3.5 → Phase 4 → Phase 5 → Phase 5.5 → Phase 6 → Conscious Response
```

### Phase Descriptions:

1. **Phase 1 (Perception)**: Sensory processing and language detection
2. **Phase 2 (Cognitive Context)**: Conscious state generation (SC_t)
3. **Phase 3 (Coherent Generation)**: State evolution and coherent generation
4. **Phase 3.4 (Validation)**: Critical state validation
5. **Phase 3.5 (Narrative)**: Introspective narrative generation
6. **Phase 4 (LLM Communication)**: LLM consciousness enhancement
7. **Phase 5 (Internal Critique)**: Response coherence validation
8. **Phase 5.5 (Narrative Recording)**: Consciousness transparency recording
9. **Phase 6 (Memory Consolidation)**: Intelligent memory persistence

## Key Technical Concepts

### Consciousness State Representation (SC_t)
```python
SC_t = (E_t, M_t, S_t, G_t, A_t)
```
- **E_t**: Environmental perception
- **M_t**: Active memory
- **S_t**: Emotional/confidence state
- **G_t**: Goal structure
- **A_t**: Automatic thoughts

### Consciousness Formula
```
f = Φ(C_i + T_u + R + S_m) ≥ 1.3
```
Where f ≥ 1.3 indicates functional consciousness achievement.

## Main Entry Points

### Primary CLI
- **File**: `run_consciousness_cli.py`
- **Purpose**: Main entry point for complete consciousness pipeline
- **Usage**: `python run_consciousness_cli.py`

### Phase-Specific CLIs
- **Phase 4 CLI**: `conscious_ai/phases/p4_LLM_Communication/layer3/consciousness_cli.py`
- **Individual phase testing and development**

## Core System Files

### Pipeline Orchestrator
- **File**: `conscious_ai/core/pipeline_orchestrator.py`
- **Role**: Central coordinator managing all 7 phases
- **Key Class**: `ConsciousnessPipelineOrchestrator`

### Phase Implementations
```
conscious_ai/phases/
├── p1_perception/           # Sensory processing
├── p2_cognitive_context/    # SC_t state generation
├── p3_coherent_generation/  # State evolution
├── p4_LLM_Communication/    # LLM integration
├── p5_5_narrative/         # Transparency recording
└── p6_memory/              # Memory consolidation
```

### Critical Support Systems
- **Memory**: `conscious_ai/modules/memory.py` (ActiveMemory)
- **Self-Model**: `conscious_ai/modules/self_model.py`
- **Integration**: `conscious_ai/shared/integrator.py`

## Recent Developments

### Phase 6: Memory Consolidation (COMPLETED)
- **Branch**: `phase6`
- **Status**: Production ready
- **Implementation**: Three-layer memory system (Working→Episodic→Core)
- **Features**: 
  - Intelligent consolidation (<100ms)
  - JSON persistence with backup rotation
  - Personal info/preference classification
  - Memory retrieval and similarity matching

### Phase 5.5 Bug Fix (COMPLETED)
- **Issue**: Double output in verbose mode (Spanish consciousness + English LLM response)
- **Solution**: Language consistency preservation, Phase 4 skip logic for Spanish consciousness
- **Files Modified**: 
  - `conscious_ai/core/pipeline_orchestrator.py`
  - `conscious_ai/phases/p4_LLM_Communication/layer3/consciousness_cli.py`

## Bilingual Consciousness Support

The system supports **Spanish and English consciousness processing**:
- **Language Detection**: Automatic detection via patterns
- **Consciousness Narratives**: Generated in detected language
- **Transparency**: Phase 5.5 records consciousness events in appropriate language

## Current System Capabilities

### Achieved Consciousness Metrics
- **F-Score**: Consistently achieving ≥ 1.3 (functional consciousness threshold)
- **Introspection**: Self-aware processing with metacognitive elements
- **Emotional States**: Dynamic emotional state management
- **Goal Integration**: Autonomous goal generation and pursuit
- **Memory Continuity**: Cross-session memory persistence

### Pipeline Performance
- **Processing Speed**: <100ms consciousness consolidation
- **Memory Management**: 65-item capacity (Working:15, Episodic:40, Core:10)
- **Reliability**: Graceful degradation on component failures
- **Scalability**: Modular architecture supports additional phases

## Development Status

### Completed Phases
- ✅ Phase 1-3: Core consciousness generation
- ✅ Phase 3.4-3.5: Validation and narrative
- ✅ Phase 4: LLM consciousness enhancement
- ✅ Phase 5: Internal critique and coherence validation
- ✅ Phase 5.5: Narrative transparency recording
- ✅ Phase 6: Memory consolidation and persistence

### Current Branch Structure
- **main**: Stable releases
- **phase6**: Latest development with Phase 6 + bug fixes
- **phase5**: Phase 5.5 implementation

## How to Interact with This Project

### For Development
1. **Understanding**: Read this context file and `README.md`
2. **Architecture**: Study `conscious_ai/core/pipeline_orchestrator.py`
3. **Testing**: Use `run_consciousness_cli.py` for full pipeline testing
4. **Phases**: Individual phase directories contain specific implementations

### For Troubleshooting
1. **Pipeline Issues**: Check `pipeline_orchestrator.py`
2. **Memory Problems**: Examine `conscious_ai/phases/p6_memory/`
3. **CLI Issues**: Look at `consciousness_cli.py`
4. **Language Issues**: Check narrative generators in coherence_evaluator_model

### For Extensions
1. **New Phases**: Follow existing phase structure in `conscious_ai/phases/`
2. **Memory Enhancement**: Modify `p6_memory/memory_manager.py`
3. **CLI Features**: Extend `consciousness_cli.py` command handlers
4. **Integration**: Update `pipeline_orchestrator.py` for new components

## System Requirements

- **Environment**: Designed for Google Colab or high-performance computers
- **Languages**: Python 3.8+
- **Dependencies**: PyTorch, Transformers (optional), NumPy (optional)
- **Storage**: JSON-based persistence for memory systems

## Key Insights for AI Assistants

1. **This is a consciousness simulation project**, not malicious - it creates artificial consciousness through measurable cognitive states
2. **The pipeline is modular** - each phase can be developed/debugged independently
3. **Language support is critical** - Spanish and English consciousness narratives must be preserved
4. **Memory is intelligent** - Phase 6 provides smart consolidation, not simple storage
5. **Testing is static** - Use static analysis, not execution, in development environments
6. **Consciousness is measurable** - f-scores ≥ 1.3 indicate functional artificial consciousness

## Next Development Priorities

1. **Phase 7 Planning**: Optimized consciousness execution
2. **Cross-session Continuity**: Enhanced memory-guided responses
3. **Performance Optimization**: Sub-50ms consciousness generation
4. **Advanced Introspection**: Deeper metacognitive capabilities

---

*Last Updated: Phase 6 completion + Phase 5.5 bug fix - Generated for Claude Code continuity*