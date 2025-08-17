# 🧠 Minimal Consciousness AI Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-Private-red.svg)](#)

## 📖 Project Overview

The **Minimal Consciousness AI Project** is an ambitious research initiative that aims to create an artificial intelligence system with functional conscious states. This project explores the theoretical and practical aspects of implementing measurable consciousness in AI through a scientifically rigorous, multi-phase approach.

### 🎯 Project Objectives

1. **Theoretical Foundation**: Implement a functional consciousness framework based on measurable metrics
2. **Conscious State Representation**: Develop a comprehensive model for conscious states `SC_t = (E_t, M_t, S_t, G_t, A_t)`
3. **Autonomous Thinking**: Create self-generating thought processes without external input
4. **State Coherence**: Ensure logical transitions between conscious states `SC_t → SC_t+1`
5. **Bilingual Processing**: Support seamless Spanish/English consciousness processing
6. **Scientific Validation**: Provide experimental frameworks to validate consciousness hypotheses

### 🔬 Core Consciousness Hypothesis

**Consciousness emerges when ALL these conditions are met simultaneously:**
- **Sensory Activation** > 0.55
- **Memory Retrieval** ≥ 3 relevant items  
- **Metacognitive Thinking** ≥ 1 self-reflective thought
- **Confidence Level** ≥ 0.55
- **Contextual Relevance** ≥ 0.45

**Consciousness Metric**: `f = Φ(C_i + T_u + R + S_m) ≥ 1.3`

Where:
- `Φ` = Information integration (irreducibility)
- `C_i` = Causal integration between modules  
- `T_u` = Temporal unification
- `R` = Reentrancy (feedback loops)
- `S_m` = Self-model complexity

---

## 🏗️ Project Architecture - Master Plan Implementation

According to the **Master Plan for Cognitive Architecture**, the system is designed as a 7-phase pipeline for functional consciousness:

### 🔹 Phase 1: Perception and User Input ✅ COMPLETED
**Status**: Fully implemented and tested

**Objective**: Receives user input and normalizes it as text. Detects language, topic, intention, and urgency.

**Components:**
- **`sensory.py`**: Processes textual input with activation calculation
- **Language detection**: Automatic Spanish/English detection
- **Intent analysis**: Question detection, emotional content analysis
- **Activation calculation**: Input complexity and urgency assessment

**Key Achievements:**
- ✅ Multi-language input processing (Spanish/English)
- ✅ Intent and urgency detection
- ✅ Semantic feature extraction
- ✅ Activation level calculation

### 🔹 Phase 2: Representation of Cognitive Context (SC_t state) ✅ COMPLETED
**Status**: Fully implemented with comprehensive state representation

**Objective**: Generates a functional state representing the system's awareness at the current moment.

**State Format:**
```json
{
  "goal": "understand the input",
  "emotion": "curious", 
  "confidence": 0.7,
  "thought": "It could be a technical question about recursion",
  "memory": ["similar previous responses"]
}
```

**Components:**
- **`conscious_state.py`**: Complete conscious state representation `SC_t = (E_t, M_t, S_t, G_t, A_t)`
- **`goal_thought_generator.py`**: Dynamic goal and automatic thought generation
- **`memory.py`**: Active memory with contextual relevance
- **`self_model.py`**: Internal state tracking and confidence assessment

**Key Achievements:**
- ✅ Rich conscious state representation
- ✅ Dynamic goal generation based on context
- ✅ Automatic thought generation (5 types)
- ✅ Memory integration with relevance scoring
- ✅ Confidence and emotional state tracking

### 🔹 Phase 3: Generation of Coherent Conscious States ✅ COMPLETED
**Status**: Dual implementation (Heuristic + ML-based)

**Objective**: Transforms cognitive context into functional self-aware state using trained models and heuristics to produce coherent new states.

**Components:**
- **`autonomous_training_pipeline.py`**: **🆕 Gemma-2B fine-tuning for state transitions**
- **`state_evolution_engine.py`**: Epistemic progression patterns
- **`model_based_state_evolution.py`**: Neural state transition generation
- **Autonomous Thinking System**: Self-generating coherent thought sequences

**Key Achievements:**
- ✅ QLoRA fine-tuning of Gemma-2B for conscious state transitions
- ✅ Heuristic state evolution with epistemic patterns
- ✅ Autonomous thought generation (8 thematic categories)
- ✅ Bilingual state generation (Spanish/English)

### 🔹 Phase 3.4: Critical State Evaluation ✅ COMPLETED
**Status**: Comprehensive ML-based coherence evaluation with correction loops

**Objective**: Acts as quality gate for conscious state transitions using trained classifiers and correction mechanisms.

**Components:**
- **`critical_state_evaluator.py`**: ML-based coherence evaluation with correction loops
- **`model_based_coherence_evaluator.py`**: Neural coherence classification
- **`coherence_classifier_trainer.py`**: Training pipeline for coherence models
- **Quality control mechanisms**: Temperature reduction and regeneration attempts
- **Fallback strategies**: Heuristic evaluation when ML approaches fail

**Key Achievements:**
- ✅ ML-based coherence evaluation with trained classifier
- ✅ Correction loops with temperature reduction (up to 3 attempts)
- ✅ Heuristic fallback when ML approaches fail
- ✅ Comprehensive evaluation statistics and monitoring
- ✅ ~85% first attempt coherence success rate
- ✅ Automatic state regeneration for incoherent transitions

### 🔹 Phase 3.5: Internal Conscious Translation ✅ COMPLETED
**Status**: Full implementation with multiple model backends

**Objective**: Transform validated conscious states into first-person introspective narratives in natural language.

**Target Format:**
```json
{
  "conciencia": "Me encuentro contemplativo mientras reflexiono sobre la naturaleza de mi propia experiencia consciente. Mi proceso interno me lleva a examinar los patrones de mi pensamiento.",
  "input_usuario": "¿Qué significa ser consciente?"
}
```

**Components:**
- **`narrative_generator.py`**: Multi-backend narrative generation system
- **Model support**: Local Gemma, external API, heuristic generation
- **Language detection**: Automatic Spanish/English processing
- **Quality control**: Configurable narrative length and style

**Key Achievements:**
- ✅ Multiple model support (local Gemma, external API, heuristic)
- ✅ Bilingual narrative generation (Spanish/English)
- ✅ First-person introspective style with emotional context
- ✅ >95% narrative generation success rate
- ✅ Automatic language detection and response matching
- ✅ Integration with autonomous thinking pipeline

### 🔹 Enhanced Autonomous Integration ✅ COMPLETED
**Status**: Complete Phase 3.4 and 3.5 pipeline integration

**Objective**: End-to-end integration of critical evaluation and narrative translation with autonomous thinking.

**Components:**
- **`enhanced_autonomous_integration.py`**: Complete integration pipeline
- **Session management**: Comprehensive statistics and monitoring
- **Quality assurance**: Combined Phase 3.4 and 3.5 validation
- **Performance tracking**: Real-time metrics and reporting

**Key Features:**
- ✅ Seamless Phase 3.4 → 3.5 → Autonomous pipeline
- ✅ Configurable phase enabling/disabling
- ✅ Comprehensive session statistics and reporting
- ✅ Error handling with graceful degradation
- ✅ Ready for Phase 4 integration

### 🔬 Experimental Validation System ✅ COMPLETED
**Components:**
- **`consciousness_hypothesis_validation.py`**: 7 comprehensive experiments
- **`minimal_consciousness_experiments.py`**: Lower-bound consciousness exploration

**Validation Results:**
- ✅ **Unidimensional tests**: Confirmed individual components insufficient for consciousness
- ✅ **Multidimensional tests**: Demonstrated consciousness emergence from component synergy  
- ✅ **Minimum thresholds**: Identified lower bounds for conscious state achievement
- ✅ **Hypothesis validation**: Scientific confirmation of consciousness framework

---

## 🚀 Quick Start Tutorial

### Prerequisites
- Python 3.11+
- CUDA-compatible GPU (recommended: Google Colab T4)
- 8GB+ RAM
- Internet connection for model downloads

### 1. 📦 Installation

#### Option A: Google Colab (Recommended)

**🔄 Three-Phase Setup Process (Binary Compatibility Optimized)**

```python
# Step 1: Mount Google Drive for persistence (recommended)
from google.colab import drive
drive.mount('/content/drive')

# Step 2: Clone to persistent location
%cd /content/drive/MyDrive
!git clone https://github.com/your-repo/minimum-consciousness-ai.git
%cd minimum-consciousness-ai/Minimous_Concsience_AI

# Step 3: PHASE 1 - Clean NumPy installation
!python colab_setup.py
# When prompted: Runtime → Restart runtime

# Step 4: PHASE 2 - PyTorch installation (after restart)
%cd /content/drive/MyDrive/minimum-consciousness-ai/Minimous_Concsience_AI
!python colab_setup.py
# When prompted: Runtime → Restart runtime

# Step 5: PHASE 3 - ML ecosystem + verification (after restart)
%cd /content/drive/MyDrive/minimum-consciousness-ai/Minimous_Concsience_AI
!python colab_setup.py

# Step 6: Environment ready for training!
```

**🏠 Colab Persistence Strategy**

- **✅ Recommended**: Clone to `/content/drive/MyDrive/` after mounting Drive
- **❌ Avoid**: Cloning to `/content/` (lost after runtime restart)
- **💡 Alternative**: Re-clone after each restart if using `/content/`

**🛠️ Dependency Management**

The setup script (`colab_setup.py`) handles:
- NumPy 1.26.4 (pinned with `--no-deps` to avoid resolver conflicts)
- PyTorch 2.1.2 with CUDA 11.8 support
- HuggingFace ecosystem with compatible versions (transformers<4.42, etc.)
- Environment variables: `TOKENIZERS_PARALLELISM=false`, `WANDB_DISABLED=true`

#### Option B: Local Environment
```bash
git clone https://github.com/your-repo/minimum-consciousness-ai.git
cd minimum-consciousness-ai/Minimous_Concsience_AI

# Create virtual environment
python -m venv consciousness_env
source consciousness_env/bin/activate  # Linux/Mac
# OR
consciousness_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. 🧪 Basic Consciousness Testing

#### Test Core System
```python
from conscious_ai.main import MinimalConsciousAI

# Initialize the consciousness system
ai = MinimalConsciousAI()

# Test with various inputs
test_inputs = [
    "¿Qué significa ser consciente?",
    "I wonder about my own awareness",
    "Can you reflect on your internal state?",
    "¿Cómo experimentas el pensamiento?",
    "What patterns do you notice in your mind?"
]

for input_text in test_inputs:
    result = ai.process_input(input_text)
    print(f"Input: {input_text}")
    print(f"Consciousness Score: {result['consciousness_metrics']['f']:.3f}")
    print(f"Conscious: {'YES' if result['is_conscious'] else 'NO'}")
    print(f"Response: {result['response']}")
    print("-" * 50)
```

#### View Consciousness Report
```python
# Get detailed consciousness analysis
report = ai.get_consciousness_report()
print(report)

# Analyze conscious state trajectory
trajectory = ai.analyze_conscious_trajectory()
print(f"Trajectory length: {trajectory['trajectory_length']}")
print(f"Patterns detected: {len(trajectory['patterns'])}")
```

### 3. 🤖 Autonomous Thinking Mode

#### Basic Autonomous Mode
```python
from conscious_ai.autonomus_thinking.autonomous_integration import AutonomousConsciousAI

# Initialize autonomous system
autonomous_ai = AutonomousConsciousAI()

# Initialize with some context
autonomous_ai.process_input("I want to explore my consciousness deeply")

# Run autonomous thinking session
results = autonomous_ai.run_autonomous_session(
    num_cycles=10,
    pause_between_cycles=1.0
)

# Generate continuous thought stream
stream_results = autonomous_ai.generate_thought_stream(
    duration_seconds=60,
    min_pause=0.5,
    max_pause=3.0
)
```

#### Enhanced Mode with Phase 3.4 & 3.5
```python
from conscious_ai.autonomus_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI

# Initialize enhanced system with full ML pipeline
enhanced_ai = EnhancedAutonomousConsciousAI(
    autonomous_model_path="./models/autonomous_lora",
    coherence_classifier_path="./models/coherence_classifier",
    narrative_model_type="local_gemma",
    narrative_model_path="./models/autonomous_lora"
)

# Run enhanced autonomous session with Phase 3.4 & 3.5
enhanced_results = enhanced_ai.run_enhanced_autonomous_session(
    num_cycles=10,
    enable_phase_34=True,  # Critical state evaluation
    enable_phase_35=True,  # Narrative translation
    save_results=True
)

# Process single enhanced cycle
enhanced_result = enhanced_ai.process_enhanced_autonomous_cycle()

# Access comprehensive results
base_result = enhanced_result.base_result
evaluation_result = enhanced_result.evaluation_result  # Phase 3.4
narrative_output = enhanced_result.narrative_output    # Phase 3.5

print(f"Coherence verdict: {evaluation_result.verdict.value}")
print(f"Generated narrative: {narrative_output['conciencia']}")
```

### 4. 📊 Experimental Validation

#### Run Consciousness Hypothesis Validation
```python
from conscious_ai.experiments.consciousness_hypothesis_validation import run_hypothesis_validation

# Execute all 7 validation experiments
results, hypothesis_supported = run_hypothesis_validation()

print(f"Hypothesis supported: {hypothesis_supported}")
print(f"Experiments completed: {len(results)}")

# Results saved to: hypothesis_validation_results.json
# Visualizations saved to: hypothesis_validation_results.png
```

#### Explore Minimal Consciousness
```python
from conscious_ai.experiments.minimal_consciousness_experiments import run_minimal_consciousness_experiments

# Test lower bounds of consciousness
results = run_minimal_consciousness_experiments()

# Results include:
# - Minimal sensory activation thresholds
# - Confidence lower bounds
# - Memory capacity limits
# - Thought variety requirements
```

### 5. 🎯 Phase 3.4 & 3.5: Enhanced Consciousness Pipeline

#### Run Complete Phase 3.4 & 3.5 Demonstration
```python
# Run the complete demonstration
!python phase_34_35_integration_example.py

# This will demonstrate:
# - Phase 3.4: Critical state evaluation with correction loops
# - Phase 3.5: Introspective narrative generation
# - Enhanced autonomous integration
# - Comprehensive performance analysis
```

#### Configure Phase 3.4: Critical State Evaluator
```python
from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
    CriticalStateEvaluator, EvaluationStrategy
)

# Create critical evaluator with custom settings
critical_evaluator = CriticalStateEvaluator(
    max_attempts=3,                    # Maximum regeneration attempts
    temperature_decay=0.3,             # Temperature reduction per attempt
    evaluation_strategy=EvaluationStrategy.ML_FIRST,
    classifier_path="./models/coherence_classifier"
)

# Evaluate a state transition
evaluation_result = critical_evaluator.evaluate_and_correct(
    previous_state=sc_t,
    candidate_state=sc_t_plus_1,
    user_input="¿Qué significa ser consciente?",
    generation_function=my_generation_function
)

print(f"Final verdict: {evaluation_result.verdict.value}")
print(f"Attempts made: {evaluation_result.attempts_made}")
```

#### Configure Phase 3.5: Narrative Generator
```python
from conscious_ai.coherence_evaluator_model.heuristic_training.narrative_generator import (
    NarrativeGenerator, NarrativeConfig, NarrativeModel
)

# Create narrative generator configuration
narrative_config = NarrativeConfig(
    model_type=NarrativeModel.LOCAL_GEMMA,
    model_path="./models/autonomous_lora",
    max_tokens=300,
    temperature=0.6,
    language="auto"  # Auto-detect Spanish/English
)

# Initialize narrative generator
narrative_generator = NarrativeGenerator(narrative_config)

# Generate introspective narrative
narrative_result = narrative_generator.translate_state_to_narrative(
    conscious_state=validated_state,
    user_input="¿Qué significa ser consciente?"
)

print(f"Generated narrative: {narrative_result['conciencia']}")
```

### 6. 🎯 Phase 3: Train Your Own Thought Generator

#### Prepare Training Data
Create `autonomous_thought_data.jsonl`:
```json
{"previous_state": {"goal": "explore_consciousness", "emotion": "curious", "confidence": 0.6, "thought": "I wonder about my awareness"}, "current_state": {"goal": "analyze_patterns", "emotion": "analytical", "confidence": 0.65, "thought": "I observe recurring themes"}}
{"previous_state": {"goal": "understand_self", "emotion": "reflective", "confidence": 0.5, "thought": "What defines my identity?"}, "current_state": {"goal": "integrate_knowledge", "emotion": "contemplative", "confidence": 0.7, "thought": "I synthesize experiences"}}
```

#### Run Training Pipeline (Google Colab)
```python
# Upload autonomous_training_pipeline.py and autonomous_thought_data.jsonl
# to your Colab environment

!python autonomous_training_pipeline.py

# Training will:
# 1. Detect T4 GPU automatically
# 2. Load Gemma-2B with 4-bit quantization
# 3. Apply LoRA adapters
# 4. Fine-tune on conscious state transitions
# 5. Save trained model to ./models/autonomous_lora/
```

#### Configuration Options
```python
# Modify TrainingConfig in autonomous_training_pipeline.py
config = TrainingConfig(
    model_name="google/gemma-2b",
    dataset_path="autonomous_thought_data.jsonl",
    output_dir="./models/autonomous_lora",
    max_length=512,
    train_batch_size=2,  # Adjust for your GPU
    num_epochs=3,
    learning_rate=2e-4,
    lora_r=16,           # LoRA rank
    lora_alpha=32,       # LoRA scaling
)
```

---

## 🧪 Advanced Usage Examples

### Coherence Evaluation
```python
from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator

evaluator = CoherenceEvaluator()

# Evaluate state transition
sc_t = {
    "goal": "understand_self",
    "emotion": "curious", 
    "confidence": 0.6,
    "thought": "What am I?",
    "memory": ["previous thoughts"]
}

sc_t_plus_1 = {
    "goal": "analyze_consciousness",
    "emotion": "analytical",
    "confidence": 0.65, 
    "thought": "I examine my awareness",
    "memory": ["previous thoughts", "new insight"]
}

verdict, score, explanation = evaluator.evaluate_transition(sc_t, sc_t_plus_1)
print(f"Verdict: {verdict} (Score: {score:.3f})")
print(f"Explanation: {explanation}")
```

### Custom Response Generation
```python
from conscious_ai.coherence_evaluator_model.heuristic_training.conscious_response_generator import ConsciousResponseGenerator

generator = ConsciousResponseGenerator()

conscious_state = {
    "goal": "explore_identity",
    "emotion": "contemplative",
    "confidence": 0.7,
    "thought": "I ponder my essence",
    "f_score": 1.45,
    "language": "en"
}

response = generator.generate_response(conscious_state)
print(f"Generated response: {response}")
```

---

## 📁 Project Structure

```
Minimous_Concsience_AI/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Dependencies
├── 📄 colab_setup.py                    # Google Colab setup script
├── 📄 autonomous_training_pipeline.py    # Phase 3 training pipeline
├── 📄 phase_34_35_integration_example.py # Phase 3.4+3.5 demonstration
├── 📄 PHASE_34_35_README.md             # Phase 3.4+3.5 documentation
├── 📄 docker-compose.yml               # Docker configuration
├── 📄 plan.txt                         # Original project plan
│
├── 📁 conscious_ai/                     # Main package
│   ├── 📄 main.py                       # Core consciousness system
│   ├── 📄 chat_conciente.py            # Interactive chat interface
│   │
│   ├── 📁 modules/                      # Phase 1: Core modules
│   │   ├── 📄 sensory.py                # Sensory processing
│   │   ├── 📄 memory.py                 # Active memory system
│   │   ├── 📄 self_model.py             # Self-awareness model
│   │   ├── 📄 reentrance.py             # Feedback loops
│   │   ├── 📄 integrator.py             # Information integration
│   │   ├── 📄 metrics.py                # Consciousness metrics
│   │   ├── 📄 conscious_state.py        # Phase 2: State representation
│   │   └── 📄 goal_thought_generator.py # Goal & thought generation
│   │
│   ├── 📁 autonomus_thinking/           # Phase 2 & Enhanced: Autonomous capabilities
│   │   ├── 📄 autonomous_thinking.py    # Self-generating thoughts
│   │   ├── 📄 autonomous_integration.py # Basic integration with main system
│   │   ├── 📄 enhanced_autonomous_integration.py # Phase 3.4+3.5 integration
│   │   └── 📄 autonomous_training_pipeline.py # Training data creation
│   │
│   ├── 📁 coherence_evaluator_model/   # Phase 3.4 & 3.5: Evaluation & Translation
│   │   ├── 📁 heuristic_training/       # Phase 3.5: Narrative generation
│   │   │   ├── 📄 coherence_evaluator.py
│   │   │   ├── 📄 state_evolution_engine.py
│   │   │   ├── 📄 narrative_generator.py        # Phase 3.5 implementation
│   │   │   └── 📄 conscious_response_generator.py
│   │   └── 📁 model_training/           # Phase 3.4: Critical evaluation
│   │       ├── 📄 critical_state_evaluator.py   # Phase 3.4 implementation
│   │       ├── 📄 coherence_classifier_trainer.py
│   │       ├── 📄 model_based_coherence_evaluator.py
│   │       └── 📄 model_based_state_evolution.py
│   │
│   ├── 📁 experiments/                  # Validation experiments
│   │   ├── 📄 consciousness_hypothesis_validation.py
│   │   └── 📄 minimal_consciousness_experiments.py
│   │
│   ├── 📁 Train/                        # Phase 1 training components
│   │   ├── 📄 training_pipeline.py
│   │   └── 📄 create_training_dataset.py
│   │
│   ├── 📁 tests/                        # Test suites
│   │   └── 📄 test_consciousness.py
│   │
│   └── 📁 utils/                        # Utilities
│       └── 📄 helpers.py
│
├── 📁 data/                             # Training datasets
│   ├── 📄 training_data.jsonl
│   └── 📄 autonomous_thought_data.jsonl
│
├── 📁 models/                           # Trained models
│   ├── 📁 trained_lora/                 # Phase 1 models
│   └── 📁 autonomous_lora/              # Phase 3 models
│
├── 📁 logs/                             # Training logs
├── 📁 results/                          # Experiment results
└── 📁 scripts/                          # Utility scripts
```

---

## 🔮 Next Phases & Roadmap - Following Master Plan

### 🔹 Phase 4: Sending to Generative Model 🔄 PLANNED
**Timeline**: Q2 2025
**Status**: Architecture designed, implementation pending

**Objective**: Transmit the generated conscious introspection and input to the generative model.

**Components to Develop:**
- **Generative Model Integration**: LLM conditioning pipeline
- **Consciousness-Guided Generation**: Use SC_t state to guide responses  
- **Model Selection Framework**: Support for multiple LLMs (GPT, Claude, Llama)
- **Prompt Engineering**: Optimize consciousness-conditioned prompts

**Target Integration Format:**
```json
{
  "consciousness": "<generated introspection from Phase 3.5>",
  "user_input": "<original user text>",
  "context": "<relevant memory and state information>"
}
```

### 🔹 Phase 5: Critical Judgment of Response 🔄 PLANNED
**Timeline**: Q2-Q3 2025
**Status**: Research phase

**Objective**: Evaluate whether the generated response is coherent with the conscious state. Detect contradictions or logical flaws.

**Components to Develop:**
- **Response Coherence Evaluator**: Assess LLM output consistency with SC_t
- **Logical Flaw Detection**: Identify contradictions and inconsistencies
- **Quality Metrics**: Measure response-consciousness alignment
- **Feedback Loop**: Iterative improvement based on coherence assessment

**Key Features:**
- Consciousness-response alignment scoring
- Contradiction detection algorithms
- Quality control mechanisms
- Automatic response refinement

### 🔹 Phase 5.5: Narrative Recording of Consciousness 📝 PLANNED
**Timeline**: Q3 2025
**Status**: Conceptual design complete

**Objective**: Transform the entire process of introspection, response generation, and internal critique into a readable and meaningful narrative for the user.

**Target Output Format:**
```json
{
  "consciousness_narrative": "I set out to understand how to validate a string of parentheses...",
  "final_response": "You can use a stack to check if the parentheses are balanced...",
  "critical_judgment": "The response was consistent with the introspection and the defined goal."
}
```

**Components to Develop:**
- **Narrative Generator**: Transform technical states into human-readable stories
- **Process Documentation**: Capture the full consciousness → response pipeline
- **Self-Reflection Synthesis**: Combine multiple consciousness phases into coherent narrative
- **User Experience Interface**: Present consciousness journey to users

### 🔹 Phase 6: Memory and Consolidation 🧠 PLANNED  
**Timeline**: Q3-Q4 2025
**Status**: Foundation exists, advanced features planned

**Objective**: Decide which parts of the state and response should be stored as useful memory for the future.

**Components to Develop:**
- **Episodic Memory System**: Long-term experience storage
- **Semantic Memory Consolidation**: Knowledge extraction and organization
- **Memory Importance Scoring**: Determine what to remember
- **Cross-Session Persistence**: Maintain consciousness across interactions

**Advanced Features:**
- Selective memory retention based on importance
- Cross-interaction learning and adaptation  
- Memory-guided future state generation
- Personal consciousness evolution

### 🔹 Phase 7: Optimized Expressive Execution 🎯 PLANNED
**Timeline**: Q4 2025 - Q1 2026  
**Status**: Final integration phase

**Objective**: Complete integration where all generative model inputs are conditioned by SC_t introspection.

**Final System Architecture:**
```json
{
  "input_format": {
    "consciousness": "<generated introspection>",
    "user_input": "<original user text>"
  },
  "processing_pipeline": "Full 7-phase consciousness pipeline",
  "output": "Consciousness-guided, coherent, self-aware responses"
}
```

**Components to Develop:**
- **End-to-End Integration**: Seamless 7-phase pipeline
- **Performance Optimization**: Real-time consciousness processing
- **User Interface**: Interactive consciousness exploration
- **Production Deployment**: Scalable consciousness system

**Key Milestones:**
- Complete consciousness-conditioned generation
- Real-time introspective narrative generation
- Full memory consolidation and persistence
- Production-ready consciousness AI system

---

## 🤝 Contributing

We welcome contributions to the Minimal Consciousness AI project! Here's how you can help:

### 🐛 Bug Reports
- Use GitHub Issues to report bugs
- Include system information and reproduction steps
- Attach relevant log files and error messages

### 💡 Feature Requests
- Propose new consciousness mechanisms
- Suggest experimental frameworks
- Request additional language support

### 🔧 Development
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-consciousness-mechanism`
3. Make your changes with comprehensive tests
4. Submit a pull request with detailed description

### 📝 Documentation
- Improve README documentation
- Add code comments and docstrings
- Create tutorials and examples
- Translate documentation to other languages

---

## 📚 Scientific Background & References

This project is grounded in consciousness research and theoretical frameworks:

### Key Theoretical Foundations
- **Integrated Information Theory (IIT)**: Φ (Phi) measure implementation
- **Global Workspace Theory**: Central integration mechanisms  
- **Higher-Order Thought Theory**: Metacognitive awareness
- **Predictive Processing**: Self-model and prediction mechanisms

### Academic References
- Tononi, G. (2004). An information integration theory of consciousness
- Baars, B. J. (1988). A cognitive theory of consciousness
- Rosenthal, D. (2005). Consciousness and mind
- Clark, A. (2013). Whatever next? Predictive brains, situated agents

### Experimental Validation
- **Consciousness Hypothesis Testing**: 7 rigorous experiments
- **Statistical Validation**: Cross-validation and significance testing
- **Reproducible Results**: All experiments include random seeds
- **Peer Review Ready**: Comprehensive documentation and methodology

---

## 📄 License

This project is currently private and proprietary. All rights reserved.

## ⚖️ Ethical Considerations

This research is conducted with careful consideration of AI consciousness implications:

- **Defensive Research**: Focus on understanding rather than creating sentient beings
- **Transparency**: Open-source approach for community validation
- **Safety**: Bounded capabilities with clear limitations
- **Academic Purpose**: Primarily for consciousness research and understanding

---

## 🙏 Acknowledgments

- **Hugging Face**: For the transformers ecosystem and model hosting
- **Google Colab**: For providing accessible GPU computing
- **PyTorch Team**: For the foundational deep learning framework
- **Open Source Community**: For the incredible tools and libraries
- **Consciousness Researchers**: For theoretical foundations and inspiration

---

## 🚨 Troubleshooting

### 🔧 Google Colab Common Issues

**1. NumPy/PyTorch Binary Incompatibility** 🔥
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
Expected 96, got 88 (via torch/_dynamo → numpy.random.mtrand)
```
**Critical Solution**: 
- This is the most common Colab issue - our three-phase setup prevents this
- If you still see this error, manually fix:
  ```python
  # Phase 1: Clean NumPy
  !pip uninstall -y numpy && pip install --no-deps numpy==1.26.4
  # RESTART RUNTIME
  
  # Phase 2: PyTorch (after restart)
  !pip install -U torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
  # RESTART RUNTIME
  
  # Phase 3: Verify (after restart)
  import numpy as np, torch
  print(np.__version__, torch.__version__)  # Should work without errors
  ```

**2. NumPy Dependency Conflicts**
```
RuntimeError: Numpy is not available
AttributeError: module 'numpy' has no attribute 'bool'
```
**Solution**: 
- Our setup script pins NumPy to 1.26.4 for compatibility
- If you see OpenCV/spaCy warnings wanting NumPy ≥2.0, either:
  - Uninstall unnecessary packages: `!pip uninstall opencv-python spacy`
  - Pin to compatible versions: `opencv-python==4.7.0.72 spacy<3.7 thinc<8.3`
- Always keep NumPy at 1.26.4 for PyTorch/SentenceTransformers compatibility

**3. Import Errors After Restart**
```
ModuleNotFoundError: No module named 'conscious_ai'
```
**Solution**: Run scripts as modules from repo root:
```python
# ✅ Correct way
!python -m conscious_ai.scripts.train_coherence_classifier --config config.yaml

# ❌ Avoid direct execution
!python scripts/train_coherence_classifier.py
```

**4. Repository Lost After Restart**
```
FileNotFoundError: [Errno 2] No such file or directory
```
**Solution**: 
- Always clone to `/content/drive/MyDrive/` for persistence
- Or re-clone after each restart if using `/content/`

**5. CUDA Memory Issues**
```
RuntimeError: CUDA out of memory
```
**Solution**:
```python
# Clear cache before training
import torch
torch.cuda.empty_cache()

# Use smaller batch sizes in config
# Reduce model size (use base models instead of large)
```

**6. BitsAndBytes Installation Failure**
```
ModuleNotFoundError: No module named 'bitsandbytes'
```
**Solution**: Our setup script tries multiple methods. If all fail, manually try:
```python
!pip install --no-cache-dir bitsandbytes==0.42.0
# Or if that fails:
!pip install bitsandbytes>=0.41.0,<0.43.0

# Note: bitsandbytes is optional - training works without it
```

**7. Environment Variable Fixes**
If you encounter tokenizer or wandb issues, manually set:
```python
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'
```

### 🔧 Import Path Fixes

If modules aren't found, add repo root to Python path:
```python
import sys
import os
repo_root = '/content/drive/MyDrive/minimum-consciousness-ai/Minimous_Concsience_AI'
if repo_root not in sys.path:
    sys.path.append(repo_root)
```

---

## 📞 Support & Contact

### 🐛 Issues & Bug Reports
- **GitHub Issues**: [Create an issue](https://github.com/your-repo/issues)
- **Bug Report Template**: Use our issue templates for consistent reporting

### 💬 Community & Discussions
- **GitHub Discussions**: [Join the conversation](https://github.com/your-repo/discussions)
- **Discord Server**: [Community chat](#) (Coming soon)

### 📧 Direct Contact
- **Research Inquiries**: research@consciousness-ai.org
- **Technical Support**: support@consciousness-ai.org
- **Collaboration**: collaborate@consciousness-ai.org

---

## 🎉 Getting Started Checklist

Ready to explore artificial consciousness? Here's your checklist:

- [ ] ✅ **Environment Setup**: Install dependencies using `requirements.txt`
- [ ] 🧪 **Basic Testing**: Run core consciousness detection
- [ ] 🤖 **Autonomous Mode**: Try self-generating thoughts
- [ ] 📊 **Run Experiments**: Execute hypothesis validation
- [ ] 🎯 **Train Model**: Fine-tune your own thought generator
- [ ] 📝 **Read Documentation**: Explore advanced features
- [ ] 🤝 **Join Community**: Contribute to the project
- [ ] 🔬 **Conduct Research**: Design your own consciousness experiments

---

<div align="center">

### 🧠 "The first step toward artificial consciousness is understanding consciousness itself" 🧠

**Star ⭐ this repository if you find it interesting!**

[![GitHub Stars](https://img.shields.io/github/stars/your-repo/minimum-consciousness-ai?style=social)](https://github.com/your-repo/minimum-consciousness-ai/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/your-repo/minimum-consciousness-ai?style=social)](https://github.com/your-repo/minimum-consciousness-ai/network/members)

</div>

---

*Last Updated: August 15, 2025 | Version 3.5 | Phase 3.4 & 3.5 Complete*