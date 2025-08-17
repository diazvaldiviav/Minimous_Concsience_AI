"""
Google Colab Setup Script for Phase 3 Training Pipeline
=======================================================
Run this first in Google Colab to ensure proper environment setup.
"""

import subprocess
import sys
import os
from typing import List

def install_requirements():
    """Install all required packages for the training pipeline."""
    
    print("🚀 Setting up Phase 3 Training Environment for Google Colab")
    print("=" * 60)
    
    # Core packages that need specific installation order
    core_packages = [
        "torch>=2.1.0,<2.3.0 torchvision>=0.16.0 torchaudio>=2.1.0 --index-url https://download.pytorch.org/whl/cu118",
        "transformers>=4.36.0,<4.42.0",
        "accelerate>=0.25.0,<0.28.0", 
        "peft>=0.7.0,<0.9.0",
        "bitsandbytes>=0.41.0,<0.43.0",
        "datasets>=2.14.0,<2.20.0",
        "sentence-transformers>=2.2.2,<2.8.0",
    ]
    
    # Additional packages
    additional_packages = [
        "scikit-learn>=1.3.0",
        "numpy==1.23.5", 
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
        "psutil>=5.9.0",
        "colorama>=0.4.6",
        "rich>=13.0.0",
        "safetensors>=0.3.0",
        "evaluate>=0.4.0",
    ]
    
    print("📦 Installing core packages...")
    for package in core_packages:
        print(f"Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install"] + package.split(), 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}")
            print(result.stderr)
        else:
            print(f"✅ Successfully installed {package}")
    
    print("\n📦 Installing additional packages...")
    for package in additional_packages:
        print(f"Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}")
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
    
    if all_good:
        print("\n🎉 Environment setup completed successfully!")
        print("✨ Ready to run the training pipeline!")
    else:
        print("\n❌ Some packages failed to install. Please check the errors above.")
    
    return all_good

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
    
    # Install packages
    install_requirements()
    
    if IN_COLAB:
        print("\n" + "="*60)
        print("🎯 NEXT STEPS:")
        print("1. Upload your autonomous_thought_data.jsonl file")
        print("2. Upload the autonomous_training_pipeline.py script") 
        print("3. Run: python autonomous_training_pipeline.py")
        print("="*60)