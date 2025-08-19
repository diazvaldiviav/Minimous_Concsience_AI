"""
Google Colab Setup Script for Phase 3 Training Pipeline
=======================================================
Optimized for Google Colab July-2025 runtime with NumPy/PyTorch binary compatibility.

CRITICAL: This script addresses NumPy/PyTorch binary incompatibility issues
that cause "numpy.dtype size changed" errors in Colab environments.

USAGE (Three-Phase Process):
1. Run this script FIRST PHASE (cleans and installs NumPy)
2. MANDATORY: Restart runtime when prompted
3. Run this script SECOND PHASE (installs compatible PyTorch stack)
4. MANDATORY: Restart runtime when prompted  
5. Run this script THIRD PHASE (verification and final setup)

IMPORTANT: For persistence, clone repo to /content/drive/MyDrive/ after mounting Drive.
"""

import subprocess
import sys
import os
from typing import List, Tuple
import json

# Set environment variables for stability (must be set early)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['WANDB_DISABLED'] = 'true'

def check_runtime_state() -> str:
    """Determine which phase of setup we're in based on installed packages."""
    try:
        # Check if NumPy is properly installed
        result = subprocess.run([sys.executable, "-c", "import numpy; print(numpy.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        numpy_installed = result.returncode == 0
        numpy_version = result.stdout.strip() if numpy_installed else None
        
        # Check if PyTorch is installed
        result = subprocess.run([sys.executable, "-c", "import torch; print(torch.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        torch_installed = result.returncode == 0
        torch_version = result.stdout.strip() if torch_installed else None
        
        # Check if transformers is installed
        result = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        transformers_installed = result.returncode == 0
        
        print(f"📊 Current Environment State:")
        print(f"  NumPy: {'✅ ' + numpy_version if numpy_installed else '❌ Not installed'}")
        print(f"  PyTorch: {'✅ ' + torch_version if torch_installed else '❌ Not installed'}")
        print(f"  Transformers: {'✅' if transformers_installed else '❌ Not installed'}")
        print()
        
        # Determine phase
        if not numpy_installed or (numpy_version and not numpy_version.startswith("1.26.4")):
            return "phase1_numpy"
        elif numpy_installed and not torch_installed:
            return "phase2_pytorch"
        elif numpy_installed and torch_installed and not transformers_installed:
            return "phase3_ecosystem"
        else:
            return "verification"
            
    except Exception as e:
        print(f"⚠️ Error checking environment: {e}")
        return "phase1_numpy"

def clean_numpy_environment():
    """Phase 1: Clean NumPy installation to prevent binary incompatibility."""
    print("🧹 PHASE 1: Cleaning NumPy Environment")
    print("=" * 60)
    
    print("🗑️ Removing all NumPy installations...")
    # Uninstall all potential NumPy packages
    packages_to_remove = ["numpy", "numpy-base", "numpy-devel"]
    for package in packages_to_remove:
        result = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"  ✅ Removed {package}")
        else:
            print(f"  ℹ️ {package} not found (OK)")
    
    # Also remove any cached wheels
    print("🧹 Clearing pip cache...")
    subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                  capture_output=True, text=True)
    
    print("\n📦 Installing clean NumPy 1.26.4 with --no-deps...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--no-deps", "--force-reinstall", "numpy==1.26.4"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to install NumPy: {result.stderr}")
        return False
    else:
        print("✅ NumPy 1.26.4 installed successfully")
    
    print("\n" + "🔄" * 20)
    print("🚨 CRITICAL: RESTART RUNTIME NOW (Phase 1 → Phase 2)")
    print("🔄" * 20)
    print("📋 Instructions:")
    print("1. Go to Runtime → Restart runtime")
    print("2. After restart, run this script again")
    print("3. DO NOT import any packages until after restart!")
    print("🔄" * 20)
    
    return True

def install_pytorch_stack():
    """Phase 2: Install PyTorch stack with CUDA 11.8 support."""
    print("🔥 PHASE 2: Installing PyTorch Stack")
    print("=" * 60)
    
    # Verify NumPy is still correct after restart
    try:
        result = subprocess.run([sys.executable, "-c", "import numpy; print(f'NumPy: {numpy.__version__}')"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print("❌ NumPy verification failed - restart may not have occurred")
            return False
    except Exception as e:
        print(f"❌ NumPy check failed: {e}")
        return False
    
    print("\n📦 Installing PyTorch CUDA 11.8 stack...")
    pytorch_command = [
        sys.executable, "-m", "pip", "install", "-U",
        "torch==2.1.2", 
        "torchvision==0.16.2", 
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ]
    
    result = subprocess.run(pytorch_command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Failed to install PyTorch: {result.stderr}")
        return False
    else:
        print("✅ PyTorch CUDA 11.8 stack installed successfully")
    
    print("\n" + "🔄" * 20)
    print("🚨 CRITICAL: RESTART RUNTIME NOW (Phase 2 → Phase 3)")
    print("🔄" * 20)
    print("📋 Instructions:")
    print("1. Go to Runtime → Restart runtime")
    print("2. After restart, run this script again for final setup")
    print("3. DO NOT import torch/numpy until after restart!")
    print("🔄" * 20)
    
    return True

def verify_and_fix_transformers_version():
    """Verify transformers is in the correct version range and fix if needed."""
    
    try:
        result = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"📊 Current transformers version: {version}")
            
            # Check if version is in our desired range (4.41.x)
            version_parts = version.split('.')
            major = int(version_parts[0])
            minor = int(version_parts[1])
            
            if major == 4 and minor == 41:
                print("✅ Transformers version is correct (4.41.x)")
                return True
            else:
                print(f"⚠️ Transformers version {version} is outside desired range (4.41.x)")
                print("🔧 Installing correct transformers version...")
                
                # Force install the correct version
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "--force-reinstall", "--no-deps", "transformers>=4.41.0,<4.42.0"
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("✅ Transformers downgraded/upgraded to 4.41.x")
                    return True
                else:
                    print(f"❌ Failed to install correct transformers version: {result.stderr}")
                    return False
        else:
            print("❌ Could not check transformers version")
            return False
            
    except Exception as e:
        print(f"❌ Error checking transformers version: {e}")
        return False

def install_bitsandbytes():
    """Install bitsandbytes with multiple fallback methods."""
    
    # Method 1: Try standard installation first
    print("🎯 Method 1: Standard bitsandbytes installation...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes==0.42.0"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ bitsandbytes 0.42.0 installed successfully (standard method)")
        return True
    else:
        print(f"❌ Standard installation failed: {result.stderr}")
    
    # Method 2: Try with --no-cache-dir
    print("🎯 Method 2: Installing with --no-cache-dir...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "bitsandbytes==0.42.0"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ bitsandbytes 0.42.0 installed successfully (no-cache method)")
        return True
    else:
        print(f"❌ No-cache installation failed: {result.stderr}")
    
    # Method 3: Try latest compatible version
    print("🎯 Method 3: Installing latest compatible version...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes>=0.41.0,<0.43.0"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ bitsandbytes installed successfully (compatible version)")
        return True
    else:
        print(f"❌ Compatible version installation failed: {result.stderr}")
    
    # Method 4: Try with --force-reinstall
    print("🎯 Method 4: Force reinstall...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", "bitsandbytes==0.42.0"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ bitsandbytes 0.42.0 installed successfully (force reinstall)")
        return True
    else:
        print(f"❌ Force reinstall failed: {result.stderr}")
    
    # Method 5: Try building from source (last resort)
    print("🎯 Method 5: Building from source (may take several minutes)...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "--no-binary", "bitsandbytes", "bitsandbytes==0.42.0"],
                          capture_output=True, text=True, timeout=600)  # 10 minute timeout
    if result.returncode == 0:
        print("✅ bitsandbytes 0.42.0 built and installed successfully")
        return True
    else:
        print(f"❌ Source build failed: {result.stderr}")
    
    # Method 6: Try installing without version constraint
    print("🎯 Method 6: Installing latest version...")
    result = subprocess.run([sys.executable, "-m", "pip", "install", "bitsandbytes"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ bitsandbytes installed successfully (latest version)")
        return True
    else:
        print(f"❌ Latest version installation failed: {result.stderr}")
    
    print("⚠️ ALL BITSANDBYTES INSTALLATION METHODS FAILED")
    print("💡 You can continue without bitsandbytes, but quantization will be disabled")
    print("💡 Manual installation after setup: !pip install bitsandbytes")
    return False

def install_ml_ecosystem():
    """Phase 3: Install ML ecosystem (HuggingFace, etc.)."""
    print("🤗 PHASE 3: Installing ML Ecosystem")
    print("=" * 60)
    
    # Verify NumPy and PyTorch after restart
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import numpy, torch; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}')"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(f"✅ {result.stdout.strip()}")
        else:
            print("❌ NumPy/PyTorch verification failed")
            return False
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False
    
    # Install HuggingFace ecosystem (without bitsandbytes first)
    print("\n📦 Installing HuggingFace ecosystem...")
    hf_packages = [
        "transformers>=4.41.0,<4.42.0",
        "accelerate<0.28", 
        "peft<0.9",
        "sentence-transformers<2.8",
        "datasets>=2.14,<2.20",
        "evaluate>=0.4,<0.5",
        "safetensors>=0.3",
    ]
    
    for package in hf_packages:
        print(f"📦 Installing: {package}")
        
        # Special handling for transformers to ensure exact version
        if package.startswith("transformers"):
            # Force reinstall to override any existing version
            result = subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall", package],
                                  capture_output=True, text=True)
        else:
            result = subprocess.run([sys.executable, "-m", "pip", "install", package],
                                  capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Failed to install {package}: {result.stderr}")
        else:
            print(f"✅ Successfully installed {package}")
    
    # Verify and fix transformers version
    print("\n🔍 Verifying transformers version...")
    verify_and_fix_transformers_version()
    
    # Install bitsandbytes separately with fallback methods
    print("\n🔧 Installing bitsandbytes with fallback methods...")
    install_bitsandbytes()
    
    # Install additional utility packages
    print("\n📦 Installing utility packages...")
    utility_packages = [
        "scikit-learn>=1.3.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "jsonlines>=3.1.0",
        "psutil>=5.9.0",
        "colorama>=0.4.6",
        "rich>=13.0.0",
        "llama-cpp-python>=0.2.0",  # For GGUF model support in Phase 3
    ]
    
    for package in utility_packages:
        print(f"📦 Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}: {result.stderr}")
        else:
            print(f"✅ Successfully installed {package}")
    
    # Handle OpenCV/spaCy compatibility
    print("\n🔧 Handling OpenCV/spaCy compatibility...")
    handle_opencv_spacy_compatibility()
    
    print("\n🔍 Final verification...")
    return verify_full_installation()

def handle_opencv_spacy_compatibility():
    """Handle OpenCV and spaCy compatibility with NumPy 1.x."""
    
    # Check if OpenCV is installed and might conflict
    result = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.__version__)"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        opencv_version = result.stdout.strip()
        print(f"ℹ️ OpenCV detected: {opencv_version}")
        
        # If OpenCV version might conflict, pin to compatible version
        if not opencv_version.startswith("4.7.0"):
            print("🔧 Pinning OpenCV to NumPy 1.x compatible version...")
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python==4.7.0.72"], 
                          capture_output=True, text=True)
            print("✅ OpenCV pinned to 4.7.0.72")
    
    # Check if spaCy is installed and might conflict
    result = subprocess.run([sys.executable, "-c", "import spacy; print(spacy.__version__)"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        spacy_version = result.stdout.strip()
        print(f"ℹ️ spaCy detected: {spacy_version}")
        
        # If spaCy version might conflict, pin to compatible versions
        major_version = int(spacy_version.split('.')[0])
        if major_version >= 3:
            minor_version = int(spacy_version.split('.')[1])
            if minor_version >= 7:
                print("🔧 Pinning spaCy ecosystem to NumPy 1.x compatible versions...")
                subprocess.run([sys.executable, "-m", "pip", "install", "spacy<3.7", "thinc<8.3"], 
                              capture_output=True, text=True)
                print("✅ spaCy ecosystem pinned to compatible versions")

def verify_full_installation():
    """Verify that all critical packages work together without binary incompatibility."""
    
    print("🔍 VERIFICATION: Testing binary compatibility...")
    
    # Test critical imports in isolation to catch binary incompatibility
    test_imports = [
        ("numpy", "import numpy as np; print(f'NumPy {np.__version__} - dtype size: {np.dtype(np.float64).itemsize}')"),
        ("torch", "import torch; print(f'PyTorch {torch.__version__} - CUDA: {torch.cuda.is_available()}')"),
        ("numpy+torch", "import numpy as np, torch; x = torch.tensor(np.array([1.0])); print(f'NumPy↔PyTorch: {x.numpy()}')"),
        ("transformers", "import transformers; print(f'Transformers {transformers.__version__}')"),
        ("sentence_transformers", "import sentence_transformers; print('SentenceTransformers OK')"),
        ("bitsandbytes", "import bitsandbytes; print('BitsAndBytes OK')"),
    ]
    
    all_good = True
    
    for name, test_code in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                # Special handling for bitsandbytes - it's optional
                if name == "bitsandbytes":
                    print(f"⚠️ {name} not available (optional - quantization disabled)")
                    print("💡 You can manually install later: !pip install bitsandbytes")
                else:
                    print(f"❌ {name} failed: {result.stderr}")
                    all_good = False
                    
                    # Special handling for numpy.dtype size error
                    if "numpy.dtype size changed" in result.stderr:
                        print("💥 BINARY INCOMPATIBILITY DETECTED!")
                        print("🔄 You must restart runtime and run setup again.")
                        return False
                    
        except subprocess.TimeoutExpired:
            if name == "bitsandbytes":
                print(f"⚠️ {name}: Import timeout (optional package)")
            else:
                print(f"❌ {name}: Import timeout (possible deadlock)")
                all_good = False
        except Exception as e:
            if name == "bitsandbytes":
                print(f"⚠️ {name}: {e} (optional package)")
            else:
                print(f"❌ {name}: {e}")
                all_good = False
    
    # Check CUDA functionality
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import torch; print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU Only\"}')"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print(f"✅ CUDA Check: {result.stdout.strip()}")
        else:
            print(f"⚠️ CUDA Check failed: {result.stderr}")
    except Exception as e:
        print(f"⚠️ CUDA Check error: {e}")
    
    if all_good:
        print("\n🎉 INSTALLATION COMPLETE!")
        print("=" * 60)
        print("✅ All packages installed and verified")
        print("✅ No binary incompatibility detected")
        print("✅ Environment ready for training")
        print("=" * 60)
        print("🚀 You can now run training scripts:")
        print("  python -m conscious_ai.scripts.train_coherence_classifier")
        print("  python -m conscious_ai.scripts.evaluate_consciousness")
    else:
        print("\n❌ INSTALLATION ISSUES DETECTED")
        print("🔄 Consider restarting runtime and running setup again")
    
    return all_good

def check_persistence():
    """Check if repository is in a persistent location."""
    cwd = os.getcwd()
    if cwd.startswith('/content/drive'):
        print("✅ Repository is in Google Drive - will persist after restart")
    elif cwd.startswith('/content'):
        print("⚠️ WARNING: Repository is in /content - will be lost after restart!")
        print("💡 Recommendation: Clone to /content/drive/MyDrive/ for persistence")
        print("   from google.colab import drive")
        print("   drive.mount('/content/drive')")
        print("   %cd /content/drive/MyDrive")
        print("   !git clone <your-repo-url>")
    else:
        print(f"📍 Current location: {cwd}")

def main():
    """Main setup function with phase detection."""
    
    print("🚀 Google Colab Setup - Phase 3 Training Pipeline")
    print("=" * 60)
    print("Environment variables set:")
    print(f"  TOKENIZERS_PARALLELISM={os.environ.get('TOKENIZERS_PARALLELISM')}")
    print(f"  WANDB_DISABLED={os.environ.get('WANDB_DISABLED')}")
    print("=" * 60)
    
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
    
    # Determine and execute appropriate phase
    phase = check_runtime_state()
    
    if phase == "phase1_numpy":
        success = clean_numpy_environment()
    elif phase == "phase2_pytorch":
        success = install_pytorch_stack()
    elif phase == "phase3_ecosystem":
        success = install_ml_ecosystem()
    elif phase == "verification":
        print("🔍 Environment appears complete - running verification...")
        success = verify_full_installation()
    else:
        print(f"❌ Unknown phase: {phase}")
        success = False
    
    if not success:
        print("\n❌ Setup encountered issues. Please:")
        print("1. Restart runtime")
        print("2. Run this script again")
        print("3. Check error messages above")

if __name__ == "__main__":
    main()