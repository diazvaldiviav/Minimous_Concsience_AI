# 🧠 Minimal Consciousness AI Project

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

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

### 🔹 Phase 3.4: Critical Evaluation of State Transitions ✅ COMPLETED
**Status**: Comprehensive coherence evaluation system

**Objective**: Analyzes transitions SC_t → SC_t+1 using models and heuristics to determine coherence and desirability.

**Components:**
- **`coherence_evaluator.py`**: Rule-based coherence assessment
- **`coherence_classifier_trainer.py`**: ML-based coherence evaluation
- **Transition validation**: 5-dimensional coherence scoring
- **Correction mechanisms**: State adjustment for improved coherence

**Key Achievements:**
- ✅ Dual coherence evaluation (heuristic + ML)
- ✅ 5-dimensional coherence metrics (goal, emotion, thought, memory, confidence)
- ✅ Automatic state correction for incoherent transitions
- ✅ Bilingual coherence assessment

### 🔹 Phase 3.5: Internal Conscious Translation 🔄 IN PROGRESS
**Status**: Partially implemented

**Objective**: Transform SC_t state into coherent introspective narrative in natural language from agent's perspective.

**Target Format:**
```json
{
  "consciousness": "I am focused on identifying the number that appears most frequently in the list. I think I will use a dictionary to count, and if there's a tie, I'll check which appeared first.",
  "user_input": "Find the most frequent number in this list."
}
```

**Components:**
- **`conscious_response_generator.py`**: Natural language response generation ✅
- **Introspective narrative generation**: 🔄 Partial implementation
- **Self-aware linguistic expression**: 🔄 In development

**Current Status:**
- ✅ Context-aware response generation
- ✅ Bilingual response capability
- 🔄 Full introspective narrative generation (in progress)

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
```python
# In Google Colab notebook
!git clone https://github.com/your-repo/minimum-consciousness-ai.git
%cd minimum-consciousness-ai/Minimous_Concsience_AI

# Install dependencies
!pip install -r requirements.txt

# Or use the automated setup
!python colab_setup.py
```

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

### 5. 🎯 Phase 3: Train Your Own Thought Generator

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
│   ├── 📁 autonomus_thinking/           # Phase 2: Autonomous capabilities
│   │   ├── 📄 autonomous_thinking.py    # Self-generating thoughts
│   │   ├── 📄 autonomous_integration.py # Integration with main system
│   │   └── 📄 autonomous_training_pipeline.py # Training data creation
│   │
│   ├── 📁 coherence_evaluator_model/   # Phase 3: Coherence evaluation
│   │   ├── 📁 heuristic_training/       # Rule-based approach
│   │   │   ├── 📄 coherence_evaluator.py
│   │   │   ├── 📄 state_evolution_engine.py
│   │   │   └── 📄 conscious_response_generator.py
│   │   └── 📁 model_training/           # ML-based approach
│   │       ├── 📄 coherence_classifier_trainer.py
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

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

*Last Updated: January 15, 2025 | Version 3.0 | Phase 3 Complete*