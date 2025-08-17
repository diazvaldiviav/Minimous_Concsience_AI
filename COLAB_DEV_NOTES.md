# 🚀 Google Colab Development Notes

## 📋 Environment Configuration (July-2025 Runtime)

### Critical Dependencies & Versions

**Core Environment Variables:**
```python
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Prevents multiprocessing issues
os.environ['WANDB_DISABLED'] = 'true'           # Disables weights & biases logging
```

**Pinned Versions for Stability:**
- **NumPy**: `1.26.4` (exact version, installed with `--no-deps`)
- **PyTorch**: `2.1.2` with CUDA 11.8 support
- **Transformers**: `>=4.41.0,<4.42.0` (compatible with our TrainOutput usage)
- **Accelerate**: `>=0.25.0,<0.28.0`
- **PEFT**: `>=0.7.0,<0.9.0`
- **BitsAndBytes**: `0.42.0` (exact version for stability)
- **SentenceTransformers**: `>=2.2.2,<2.8.0`

### Installation Order (Critical for Success)

1. **NumPy First**: Install `numpy==1.26.4 --no-deps` to prevent pip resolver conflicts
2. **PyTorch Stack**: Install torch, torchvision, torchaudio with cu118 index
3. **HuggingFace Ecosystem**: Install transformers, accelerate, peft, etc.
4. **BitsAndBytes**: Install separately after PyTorch
5. **Utilities**: Install remaining packages

## 🔄 Three-Phase Setup Process

### Why Three Phases Are Required

NumPy/PyTorch binary incompatibility is a critical issue in Colab that causes `ValueError: numpy.dtype size changed, may indicate binary incompatibility`. This happens when NumPy C-ABI extensions don't match the loaded NumPy version. Our three-phase approach ensures clean binary compatibility:

### Process Flow

```
1. Phase 1: Clean NumPy installation (pip uninstall → clean install → RESTART)
2. Phase 2: Install PyTorch stack compatible with clean NumPy (RESTART)  
3. Phase 3: Install ML ecosystem + verification
4. Ready for Training: Environment is stable with verified binary compatibility
```

### Critical Binary Incompatibility Prevention

- **Phase 1**: Completely removes all NumPy installations and installs clean NumPy 1.26.4 with `--no-deps`
- **Restart 1**: Clears Python import cache and C extension bindings
- **Phase 2**: Installs PyTorch 2.1.2 compiled against compatible NumPy version
- **Restart 2**: Ensures PyTorch CUDA contexts initialize with correct NumPy bindings
- **Phase 3**: Adds remaining packages and tests for `numpy.dtype size changed` errors

## 🏠 Persistence Strategies

### Recommended: Google Drive Mount

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive
!git clone <repo-url>
```

**Pros:**
- Survives runtime restarts
- Persistent across sessions
- No need to re-clone

**Cons:**
- Slightly slower I/O
- Requires Drive space

### Alternative: /content Directory

```python
%cd /content
!git clone <repo-url>
```

**Pros:**
- Faster I/O (local SSD)
- No Drive space required

**Cons:**
- Lost on runtime restart
- Must re-clone frequently

## 🛠️ Dependency Conflict Resolution

### NumPy Version Conflicts

**Problem**: Many packages want different NumPy versions:
- PyTorch/SentenceTransformers: Need NumPy 1.x for compatibility
- OpenCV/spaCy: Want NumPy ≥2.0 for new features
- Colab Default: NumPy 1.26.4 (good baseline)

**Solution Strategy:**
1. Pin NumPy to 1.26.4 (compatible with most packages)
2. For conflicting packages, either:
   - Uninstall if not needed: `!pip uninstall opencv-python spacy`
   - Pin to compatible versions: `opencv-python==4.7.0.72 spacy<3.7 thinc<8.3`

### Common Conflict Patterns

```
# OpenCV wanting newer NumPy
opencv-python>=4.8.0 requires numpy>=2.0

# spaCy ecosystem conflicts  
spacy>=3.7 requires numpy>=2.0
thinc>=8.3 requires numpy>=2.0

# Solution: Pin to older compatible versions
opencv-python==4.7.0.72  # Last version supporting NumPy 1.x
spacy<3.7                 # Compatible with NumPy 1.x
thinc<8.3                 # Compatible with NumPy 1.x
```

## 📦 Module Import Patterns

### Proper Module Execution

Always run scripts as modules from the repository root:

```python
# ✅ Correct: Module execution from repo root
!python -m conscious_ai.scripts.train_coherence_classifier --config config.yaml
!python -m conscious_ai.scripts.evaluate_consciousness --model_path ./models/

# ❌ Incorrect: Direct script execution
!python scripts/train_coherence_classifier.py  # Will fail with import errors
```

### Import Path Management

If you encounter `ModuleNotFoundError`, add repo root to Python path:

```python
import sys
import os

# Get the repository root directory
repo_root = os.path.abspath('/content/drive/MyDrive/minimum-consciousness-ai/Minimous_Concsience_AI')

# Add to Python path if not already present
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Now imports should work
from conscious_ai.core.consciousness import ConsciousnessState
```

### Project Structure Assumptions

The module system assumes this directory structure:
```
Minimous_Concsience_AI/                 # Repository root
├── conscious_ai/                       # Main package
│   ├── __init__.py
│   ├── scripts/                        # Executable scripts
│   │   ├── train_coherence_classifier.py
│   │   └── evaluate_consciousness.py
│   ├── core/                          # Core modules
│   └── utils/                         # Utility modules
└── colab_setup.py                     # Setup script
```

## ⚡ Performance Optimizations

### Memory Management

```python
# Clear CUDA cache before training
import torch
torch.cuda.empty_cache()

# Monitor memory usage
!nvidia-smi

# Free up Python memory
import gc
gc.collect()
```

### Batch Size Recommendations

**For T4 GPU (15GB):**
- Training batch size: 4-8
- Evaluation batch size: 16-32
- Gradient accumulation: 2-4 steps

**Memory-efficient Training:**
```python
# Use gradient checkpointing
model.gradient_checkpointing_enable()

# Use 8-bit optimization
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
```

## 🔧 Debugging Common Issues

### 1. "Numpy is not available"

```python
# Check NumPy installation
import numpy as np
print(f"NumPy version: {np.__version__}")
print(f"NumPy path: {np.__file__}")

# Reinstall if needed
!pip install --force-reinstall numpy==1.26.4 --no-deps
```

### 2. CUDA Initialization Errors

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

# Force CUDA reinitialization (restart runtime if this fails)
torch.cuda.init()
```

### 3. Tokenizer Parallel Processing Issues

```python
# Disable parallelism (should be set automatically by setup script)
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Verify setting
print(f"Tokenizers parallelism: {os.environ.get('TOKENIZERS_PARALLELISM')}")
```

### 4. BitsAndBytes Installation Failures

```python
ModuleNotFoundError: No module named 'bitsandbytes'
```

**Common Causes:**
- CUDA version mismatch
- Compilation issues on Colab's environment
- Dependency conflicts during installation

**Solution Strategy:**
Our setup script tries 6 different methods:
1. Standard installation: `pip install bitsandbytes==0.42.0`
2. No-cache installation: `pip install --no-cache-dir bitsandbytes==0.42.0`
3. Compatible version range: `pip install bitsandbytes>=0.41.0,<0.43.0`
4. Force reinstall: `pip install --force-reinstall bitsandbytes==0.42.0`
5. Build from source: `pip install --no-binary bitsandbytes bitsandbytes==0.42.0`
6. Latest version: `pip install bitsandbytes`

**Manual Fix:**
```python
# Try each method until one works
!pip install --no-cache-dir bitsandbytes==0.42.0

# If all fail, you can continue without bitsandbytes
# Quantization features will be disabled, but training still works
```

**Note:** bitsandbytes is optional - your training pipeline can work without it (just without quantization optimizations).

### 5. Import Errors After Package Updates

```python
# Check installed versions
!pip list | grep -E "(torch|transformers|accelerate|peft|bitsandbytes)"

# Force reimport modules
import importlib
import sys

# Remove from cache and reimport
if 'conscious_ai.core.consciousness' in sys.modules:
    del sys.modules['conscious_ai.core.consciousness']
    
# Now import again
from conscious_ai.core.consciousness import ConsciousnessState
```

## 📝 Best Practices

### 1. Environment Verification

Always run verification after setup:
```python
# Run the setup script's verification function
!python -c "from colab_setup import verify_installation; verify_installation()"
```

### 2. Checkpoint Frequently

```python
# Save model checkpoints to Drive
checkpoint_dir = '/content/drive/MyDrive/consciousness_checkpoints/'
!mkdir -p {checkpoint_dir}

# Save during training
torch.save(model.state_dict(), f'{checkpoint_dir}/model_epoch_{epoch}.pt')
```

### 3. Log Everything

```python
# Use Python logging instead of print for production
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting consciousness training...")
```

### 4. Error Recovery

```python
# Wrap critical operations in try-catch
try:
    model = load_consciousness_model()
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    # Fallback to base model or re-download
```

## 🚀 Quick Start Commands

```python
# 1. Setup (run twice with restart in between)
!python colab_setup.py

# 2. Train coherence classifier
!python -m conscious_ai.scripts.train_coherence_classifier

# 3. Evaluate consciousness metrics
!python -m conscious_ai.scripts.evaluate_consciousness

# 4. Generate autonomous thoughts
!python -m conscious_ai.scripts.generate_thoughts --mode autonomous

# 5. Monitor training progress
!python -m conscious_ai.scripts.monitor_training --checkpoint_dir ./checkpoints/
```

## 📊 Monitoring & Metrics

### Resource Usage

```python
# Check system resources
!free -h          # RAM usage
!df -h            # Disk usage  
!nvidia-smi       # GPU usage

# Python memory profiling
import psutil
process = psutil.Process()
memory_info = process.memory_info()
print(f"RSS: {memory_info.rss / 1024 / 1024:.2f} MB")
```

### Training Metrics

```python
# Log key metrics during training
metrics = {
    'consciousness_score': 0.85,
    'coherence_loss': 0.23,
    'autonomy_metric': 0.91,
    'memory_usage_mb': memory_info.rss / 1024 / 1024
}

# Save to Drive for persistence
import json
with open('/content/drive/MyDrive/training_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

**Last Updated**: August 2025  
**Runtime Tested**: Google Colab July-2025  
**Python Version**: 3.11+  
**CUDA Version**: 11.8