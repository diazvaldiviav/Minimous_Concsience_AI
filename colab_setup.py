"""
Google Colab Setup Script for Phase 3 Training Pipeline
=======================================================
Optimized for Google Colab July-2025 runtime with stable dependencies.

USAGE:
1. Run this script once
2. Restart runtime when prompted
3. Run this script again to verify installation

IMPORTANT: For persistence, clone repo to /content/drive/MyDrive/ after mounting Drive.
"""

import subprocess
import sys
import os
from typing import List

# Set environment variables for stability
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

def install_requirements():
    """Install all required packages for the training pipeline with stable versions."""
    
    print("🚀 Setting up Phase 3 Training Environment for Google Colab July-2025")
    print("=" * 60)
    print("Environment variables set:")
    print(f"  TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}")
    print(f"  WANDB_DISABLED={os.environ.get('WANDB_DISABLED')}")
    print("=" * 60)
    
    # Step 1: Install NumPy first with --no-deps to avoid resolver churn
    print("📦 Step 1: Installing NumPy 1.26.4 with --no-deps...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "--no-deps"], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to install NumPy: {result.stderr}")
        return False
    else:
        print("✅ NumPy 1.26.4 installed successfully")
    
    # Step 2: Install PyTorch with CUDA 11.8 support
    print("\n📦 Step 2: Installing PyTorch CUDA 11.8 stack...")
    pytorch_packages = [
        "torch==2.1.2",
        "torchvision==0.16.2", 
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    result = subprocess.run([sys.executable, "-m", "pip", "install"] + pytorch_packages,
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to install PyTorch: {result.stderr}")
        return False
    else:
        print("✅ PyTorch CUDA 11.8 stack installed successfully")
    
    # Step 3: Install HuggingFace ecosystem with compatible versions
    print("\n📦 Step 3: Installing HuggingFace ecosystem...")
    hf_packages = [
        "transformers>=4.41.0,<4.42.0",
        "accelerate>=0.25.0,<0.28.0", 
        "peft>=0.7.0,<0.9.0",
        "sentence-transformers>=2.2.2,<2.8.0",
        "datasets>=2.14.0,<2.20.0",
        "evaluate>=0.4.0,<0.5.0",
        "safetensors>=0.3.0",
    ]
    
    for package in hf_packages:
        print(f"Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}: {result.stderr}")
        else:
            print(f"✅ Successfully installed {package}")
    
    # Step 4: Install bitsandbytes separately
    print("\n📦 Step 4: Installing bitsandbytes...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes==0.42.0"],
                          capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to install bitsandbytes: {result.stderr}")
    else:
        print("✅ bitsandbytes 0.42.0 installed successfully")
    
    # Step 5: Additional utility packages
    print("\n📦 Step 5: Installing additional utility packages...")
    additional_packages = [
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
        "psutil>=5.9.0",
        "colorama>=0.4.6",
        "rich>=13.0.0",
    ]
    
    for package in additional_packages:
        print(f"Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}: {result.stderr}")
        else:
            print(f"✅ Successfully installed {package}")
    
    print("\n🔍 Verifying installation...")
    verify_installation()

def verify_installation():
    """Verify that all critical packages are installed correctly."""
    
    critical_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("bitsandbytes", "BitsAndBytes"),
        ("accelerate", "Accelerate"),
        ("datasets", "Datasets"),
        ("sentence_transformers", "SentenceTransformers"),
    ]
    
    all_good = True
    
    for package, name in critical_packages:
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError as e:
            print(f"❌ {name} import failed: {e}")
            all_good = False
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            print("⚠️ CUDA not available - will fall back to CPU")
    except Exception as e:
        print(f"❌ CUDA check failed: {e}")
        all_good = False
    
    # Check NumPy version
    try:
        import numpy as np
        print(f"✅ NumPy version: {np.__version__}")
        if not np.__version__.startswith("1.26.4"):
            print("⚠️ Warning: NumPy version may cause compatibility issues")
    except Exception as e:
        print(f"❌ NumPy check failed: {e}")
        all_good = False
    
    if all_good:
        print("\n🎉 Environment setup completed successfully!")
        print("\n" + "="*60)
        print("🔄 CRITICAL: RESTART RUNTIME NOW")
        print("="*60)
        print("📋 Next steps:")
        print("1. Go to Runtime → Restart runtime")
        print("2. Run this setup script again to verify installation")
        print("3. Then proceed with training/evaluation")
        print("="*60)
    else:
        print("\n❌ Some packages failed to install. Please check the errors above.")
        print("Try restarting runtime and running again.")
    
    return all_good

def check_persistence():
    """Check if repository is in a persistent location."""
    cwd = os.getcwd()
    if cwd.startswith('/content/drive'):
        print("✅ Repository is in Google Drive - will persist after restart")
    elif cwd.startswith('/content'):
        print("⚠️ WARNING: Repository is in /content - will be lost after restart!")
        print("💡 Recommendation: Clone to /content/drive/MyDrive/ for persistence")
    else:
        print(f"📍 Current location: {cwd}")

def download_training_script():
    """Download the training script if needed."""
    script_content = '''# The autonomous_training_pipeline.py content would be here
# In Colab, you would typically upload the file or clone from repository
print("Upload the autonomous_training_pipeline.py file to your Colab environment")
'''
    
    with open('autonomous_training_pipeline.py', 'w') as f:
        f.write(script_content)
    
    print("📝 Training script template created")

if __name__ == "__main__":
    # Check if running in Colab
    try:
        from google.colab import files
        print("🌟 Running in Google Colab environment")
        IN_COLAB = True
    except ImportError:
        print("⚠️ Not running in Google Colab")
        IN_COLAB = False
    
    if IN_COLAB:
        check_persistence()
        print()
    
    # Install packages
    install_requirements()
    
    if IN_COLAB:
        print("\n" + "="*60)
        print("🎯 REMEMBER:")
        print("• Restart runtime when prompted above")
        print("• Run this script again after restart")
        print("• Use 'python -m conscious_ai.script.module_name' for imports")
        print("• Mount Drive for persistence: drive.mount('/content/drive')")
        print("="*60)