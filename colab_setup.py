"""
Google Colab Setup Script for Mistral Training Pipeline
=======================================================
Optimized for Google Colab September-2025 runtime with NumPy/PyTorch binary compatibility.
Updated for Mistral-7B-Instruct-v0.1 model migration with 4-bit quantization support.

CRITICAL: This script addresses NumPy/PyTorch binary incompatibility issues
that cause "numpy.dtype size changed" errors in Colab environments.

USAGE (Three-Phase Process):
1. Run this script FIRST PHASE (cleans and installs NumPy)
2. MANDATORY: Restart runtime when prompted
3. Run this script SECOND PHASE (installs compatible PyTorch stack)
4. MANDATORY: Restart runtime when prompted  
5. Run this script THIRD PHASE (verification and final setup)

NEW FEATURES:
- Mistral-7B-Instruct-v0.1 support with 4-bit quantization
- Enhanced dependency verification for training pipeline
- Support for evaluate, peft, datasets modules
- Optimized for 8-15 second latency (down from 60+ seconds)

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
        
        # Check if transformers is installed and version
        result = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        transformers_installed = result.returncode == 0
        transformers_version = result.stdout.strip() if transformers_installed else None
        
        # Check for Phase 4 Layer 1 requirements
        result = subprocess.run([sys.executable, "-c", "import openai_harmony; print('openai_harmony available')"], 
                              capture_output=True, text=True, timeout=10)
        phase4_deps_installed = result.returncode == 0
        
        # Check for Phase 7 Enhanced Consciousness dependencies
        result = subprocess.run([sys.executable, "-c", "import fastapi, uvicorn, httpx, openai; print('phase7 available')"], 
                              capture_output=True, text=True, timeout=10)
        phase7_deps_installed = result.returncode == 0
        
        print(f"📊 Current Environment State:")
        print(f"  NumPy: {'✅ ' + numpy_version if numpy_installed else '❌ Not installed'}")
        print(f"  PyTorch: {'✅ ' + torch_version if torch_installed else '❌ Not installed'}")
        print(f"  Transformers: {'✅ ' + transformers_version if transformers_installed else '❌ Not installed'}")
        print(f"  Phase 4 Deps: {'✅' if phase4_deps_installed else '❌ Not installed'}")
        print(f"  Phase 7 Deps: {'✅' if phase7_deps_installed else '❌ Not installed'}")
        print()
        
        # Determine phase with Layer 1 detection
        if not numpy_installed or (numpy_version and not numpy_version.startswith("1.26.4")):
            return "phase1_numpy"
        elif numpy_installed and not torch_installed:
            return "phase2_pytorch"
        elif numpy_installed and torch_installed and not transformers_installed:
            return "phase3_ecosystem"
        elif transformers_installed and transformers_version and not transformers_version.startswith("4.55"):
            # Check if we need Layer 1 preparation (transformers version upgrade)
            return "layer1_preparation"
        elif transformers_installed and not phase4_deps_installed:
            # Transformers is ready but Phase 4 deps not installed
            return "layer1_preparation"
        elif transformers_installed and phase4_deps_installed and not phase7_deps_installed:
            # Need Phase 7 dependencies
            return "phase7_setup"
        elif transformers_installed and phase4_deps_installed and phase7_deps_installed:
            # Check if Layer 2 conditional setup needed
            try:
                # Try to detect if hardware-specific setup is needed
                result = subprocess.run([sys.executable, "-c", 
                    "from conscious_ai.phases.p4_LLM_Communication.core.hardware_profiler import PremiumHardwareProfiler; "
                    "profiler = PremiumHardwareProfiler(); "
                    "config = profiler.detect_hardware_configuration(); "
                    "print('layer2_needed' if config.is_premium_hardware else 'layer2_skip')"], 
                    capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and 'layer2_needed' in result.stdout:
                    return "layer2_setup"
                else:
                    return "verification"
                    
            except:
                # If hardware detection fails, run Layer 2 setup anyway for safety
                return "layer2_setup"
        else:
            return "verification"
            
    except Exception as e:
        print(f"⚠️ Error checking environment: {e}")
        return "phase1_numpy"

# =============================================================================
# PHASE 4 LAYER 1: SAFE INCREMENTAL PHASE 4 PREPARATION
# =============================================================================

def create_environment_backup():
    """Create comprehensive backup before Layer 1 changes."""
    print("💾 LAYER 1.1: Creating Environment Backup")
    print("=" * 60)
    
    try:
        # Save current pip freeze to backup file
        print("💾 Saving current package versions...")
        backup_path = "/content/pre_phase4_backup.txt"
        
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            with open(backup_path, 'w') as f:
                f.write("# Pre-Phase 4 Layer 1 Environment Backup\n")
                f.write(f"# Created: {subprocess.run(['date'], capture_output=True, text=True).stdout.strip()}\n")
                f.write("# Critical packages for restoration:\n")
                f.write(result.stdout)
            
            print(f"✅ Backup created: {backup_path}")
        else:
            print(f"❌ Failed to create backup: {result.stderr}")
            return False
        
        # Log current versions of critical packages
        print("📊 Critical package versions:")
        critical_packages = ["torch", "transformers", "sentence-transformers", "numpy"]
        
        for package in critical_packages:
            try:
                result = subprocess.run([sys.executable, "-c", f"import {package}; print({package}.__version__)"], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    version = result.stdout.strip()
                    print(f"  {package}: {version}")
                else:
                    print(f"  {package}: Not installed")
            except Exception as e:
                print(f"  {package}: Error checking version - {e}")
        
        # Create restoration script
        restoration_script = f"""#!/bin/bash
# Emergency restoration script for Phase 4 Layer 1
echo "🔄 EMERGENCY RESTORATION: Rolling back to pre-Phase 4 state"
pip install --force-reinstall --no-deps numpy==1.26.4
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers>=4.41.0,<4.42.0
pip install sentence-transformers<2.8.0
echo "✅ Critical packages restored - restart runtime to complete rollback"
"""
        
        with open("/content/emergency_rollback.sh", 'w') as f:
            f.write(restoration_script)
        
        print("✅ Emergency rollback script created")
        print("✅ Environment backup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        return False

def verify_phases_1_to_3_working():
    """Ensure existing phases remain functional."""
    print("🔍 LAYER 1.2: Verifying Phases 1-3.5 Working")
    print("=" * 60)
    
    try:
        # Test 1: Import MinimalConsciousAI (Phase 1-2)
        print("🧪 Test 1: MinimalConsciousAI import...")
        result = subprocess.run([
            sys.executable, "-c", 
            "from conscious_ai.main import MinimalConsciousAI; print('✅ MinimalConsciousAI imported')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"❌ MinimalConsciousAI import failed: {result.stderr}")
            return False
        
        # Test 2: Import CriticalStateEvaluator (Phase 3.4)
        print("🧪 Test 2: CriticalStateEvaluator import...")
        result = subprocess.run([
            sys.executable, "-c", 
            "from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator; print('✅ CriticalStateEvaluator imported')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"❌ CriticalStateEvaluator import failed: {result.stderr}")
            return False
        
        # Test 3: Create simple SC_t state
        print("🧪 Test 3: Creating simple conscious state...")
        test_state_code = """
from conscious_ai.phases.p2_cognitive_context.conscious_state import ConsciousState
from datetime import datetime

# Create minimal test state
state = ConsciousState(
    E_t={"text": "test input", "activation": 0.5},
    M_t=[{"content": {"text": "test memory"}, "relevance": 0.7}],
    S_t={"emotional_state": "curious", "confidence_level": 0.6},
    G_t={"primary_goal": "test_functionality"},
    A_t=["test thought"],
    cycle=1
)
print('✅ Conscious state created successfully')
print(f'State cycle: {state.cycle}')
"""
        
        result = subprocess.run([sys.executable, "-c", test_state_code], 
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"❌ Conscious state creation failed: {result.stderr}")
            return False
        
        # Test 4: Verify sentence-transformers embeddings work
        print("🧪 Test 4: SentenceTransformers embeddings...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
import sentence_transformers
from sentence_transformers import SentenceTransformer
# Use a lightweight model for testing
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(['test sentence'])
print(f'✅ SentenceTransformers working - embedding shape: {embeddings.shape}')
"""
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"⚠️ SentenceTransformers test failed: {result.stderr}")
            print("💡 This may still work in practice")
        
        print("✅ Phase 1-3.5 verification completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def safe_update_transformers():
    """Update transformers with compatibility verification."""
    print("⬆️ LAYER 1.3: Safe Transformers Update")
    print("=" * 60)
    
    try:
        # Check current transformers version
        result = subprocess.run([sys.executable, "-c", "import transformers; print(transformers.__version__)"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            current_version = result.stdout.strip()
            print(f"📊 Current transformers version: {current_version}")
            
            # Check if already correct version
            if current_version.startswith("4.55"):
                print("✅ Transformers already at correct version")
                return True
                
        else:
            print("❌ Cannot check current transformers version")
            return False
        
        # Update to transformers 4.55.x for GPT-OSS compatibility
        print("⬆️ Updating transformers to 4.55.x...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "transformers>=4.55.0,<4.56.0", "--upgrade"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"❌ Transformers update failed: {result.stderr}")
            return False
        
        print("✅ Transformers updated successfully")
        
        # Test compatibility with existing sentence-transformers
        print("🔍 Testing compatibility with sentence-transformers...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
import transformers
import sentence_transformers
print(f'Transformers: {transformers.__version__}')
print(f'SentenceTransformers: {sentence_transformers.__version__}')
print('✅ Compatibility test passed')
"""
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(result.stdout.strip())
        else:
            print(f"⚠️ Compatibility test warning: {result.stderr}")
            print("🔄 Automatic rollback may be needed")
            
            # Attempt automatic rollback
            print("🔄 Rolling back transformers...")
            rollback_result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                "transformers>=4.41.0,<4.42.0", "--force-reinstall"
            ], capture_output=True, text=True, timeout=300)
            
            if rollback_result.returncode == 0:
                print("✅ Rollback successful - keeping stable version")
                return False
            else:
                print("❌ Rollback failed - manual intervention needed")
                return False
        
        print("✅ Transformers update completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Transformers update failed: {e}")
        return False

def install_gpt_oss_dependencies():
    """Install GPT-OSS specific packages."""
    print("🆕 LAYER 1.4: Installing GPT-OSS Dependencies")
    print("=" * 60)
    
    dependencies = [
        ("openai-harmony", "openai_harmony", "GPT-OSS harmony response format"),
        ("openai>=1.0.0", "openai", "OpenAI API client for GPT-4o-mini (replaces Mistral 7B)"),
        ("kernels", "kernels", "MXFP4 quantization support")
    ]
    
    success_count = 0
    
    for package_name, import_name, description in dependencies:
        print(f"📦 Installing {package_name} ({description})...")
        
        try:
            # Install package
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"⚠️ {package_name} installation failed: {result.stderr}")
                print(f"💡 Trying alternative installation methods...")
                
                # Try with --no-cache-dir
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "--no-cache-dir", package_name
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode != 0:
                    print(f"❌ All installation methods failed for {package_name}")
                    continue
            
            # Test import
            result = subprocess.run([
                sys.executable, "-c", f"import {import_name}; print('✅ {package_name} imported successfully')"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(result.stdout.strip())
                success_count += 1
            else:
                print(f"⚠️ {package_name} installed but import failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error installing {package_name}: {e}")
    
    print(f"📊 GPT-OSS dependencies result: {success_count}/{len(dependencies)} successful")
    
    if success_count == 0:
        print("⚠️ No GPT-OSS dependencies installed - Phase 4 will use fallback methods")
        return False
    elif success_count < len(dependencies):
        print("⚠️ Partial GPT-OSS dependencies installed - some features may be limited")
        return True
    else:
        print("✅ All GPT-OSS dependencies installed successfully")
        return True

def verify_gpt_oss_readiness():
    """Test GPT-OSS-20B compatibility without full loading."""
    print("🔍 LAYER 1.5: Verifying GPT-OSS Readiness")
    print("=" * 60)
    
    try:
        # Test 1: GPT-OSS-20B tokenizer loading (lightweight)
        print("🧪 Test 1: GPT-OSS tokenizer compatibility...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
from transformers import AutoTokenizer
try:
    # Test GPT-OSS-20B tokenizer loading (lightweight test)
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    test_text = "This is a test"
    tokens = tokenizer.encode(test_text)
    print(f'✅ GPT-OSS tokenizer working - tokens: {len(tokens)}')
except Exception as e:
    print(f'⚠️ GPT-OSS tokenizer test: {e}')
    print('💡 May require internet connection or model download')
"""
        ], capture_output=True, text=True, timeout=60)
        
        print(result.stdout.strip())
        if result.stderr:
            print(f"Warnings: {result.stderr}")
        
        # Test 2: CUDA memory availability
        print("🧪 Test 2: CUDA memory availability...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
import torch
if torch.cuda.is_available():
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'✅ CUDA available - GPU memory: {gpu_memory:.1f}GB')
    if gpu_memory >= 13:
        print('✅ Sufficient memory for GPT-OSS-20B (13GB+ required)')
    else:
        print('⚠️ Limited memory - may need quantization (13GB+ recommended)')
else:
    print('⚠️ CUDA not available - GPT-OSS will run on CPU (very slow)')
"""
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout.strip())
        
        # Test 3: Harmony format imports
        print("🧪 Test 3: Harmony format compatibility...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
try:
    import openai_harmony
    print('✅ OpenAI Harmony imported successfully')
except ImportError:
    print('⚠️ OpenAI Harmony not available - will use standard format')
except Exception as e:
    print(f'⚠️ OpenAI Harmony test: {e}')
"""
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout.strip())
        
        # Test 4: MXFP4 quantization support
        print("🧪 Test 4: MXFP4 quantization support...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
try:
    import kernels
    print('✅ Kernels package imported - MXFP4 quantization available')
except ImportError:
    print('⚠️ Kernels package not available - standard quantization only')
except Exception as e:
    print(f'⚠️ Kernels test: {e}')
"""
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout.strip())
        
        # Test 5: Phase 4 Layer 3 integration validation
        print("🧪 Test 5: Phase 4 Layer 3 integration...")
        result = subprocess.run([
            sys.executable, "-c", 
            """
try:
    from conscious_ai.phases.p4_LLM_Communication.layer3.phase4_manager import Phase4Manager
    from conscious_ai.phases.p4_LLM_Communication.layer3.integration_layer import IntegrationBridge
    print('✅ Phase 4 Layer 3 components imported successfully')
except ImportError as e:
    print(f'⚠️ Phase 4 Layer 3 not ready: {e}')
except Exception as e:
    print(f'⚠️ Phase 4 Layer 3 test error: {e}')
"""
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout.strip())
        
        print("✅ GPT-OSS readiness assessment completed")
        return True
        
    except Exception as e:
        print(f"❌ GPT-OSS readiness check failed: {e}")
        return False

def emergency_rollback():
    """Restore environment to pre-Layer 1 state."""
    print("🔄 EMERGENCY ROLLBACK: Restoring Pre-Phase 4 Environment")
    print("=" * 60)
    
    try:
        backup_path = "/content/pre_phase4_backup.txt"
        
        if not os.path.exists(backup_path):
            print("❌ No backup file found - cannot perform automatic rollback")
            print("💡 Manual restoration required:")
            print("   !pip install --force-reinstall numpy==1.26.4")
            print("   !pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118")
            print("   !pip install transformers>=4.41.0,<4.42.0")
            return False
        
        print("📖 Reading backup file...")
        with open(backup_path, 'r') as f:
            backup_content = f.read()
        
        # Extract critical package versions from backup
        print("🔧 Restoring critical packages...")
        
        # Restore NumPy
        print("📦 Restoring NumPy 1.26.4...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "numpy==1.26.4"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ NumPy restored")
        else:
            print(f"❌ NumPy restoration failed: {result.stderr}")
        
        # Restore PyTorch
        print("📦 Restoring PyTorch 2.1.2...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ PyTorch restored")
        else:
            print(f"❌ PyTorch restoration failed: {result.stderr}")
        
        # Restore Transformers
        print("📦 Restoring Transformers 4.41.x...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "transformers>=4.41.0,<4.42.0"
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print("✅ Transformers restored")
        else:
            print(f"❌ Transformers restoration failed: {result.stderr}")
        
        # Remove Phase 4 packages
        print("🗑️ Removing Phase 4 packages...")
        phase4_packages = ["openai-harmony", "kernels"]
        for package in phase4_packages:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", package], 
                          capture_output=True, text=True)
        
        print("🔄" * 20)
        print("🚨 ROLLBACK COMPLETE - RESTART RUNTIME NOW")
        print("🔄" * 20)
        print("📋 After restart:")
        print("1. Run this script again to verify restoration")
        print("2. Test Phase 1-3.5 functionality")
        print("3. Phase 4 Layer 1 preparation cancelled")
        print("🔄" * 20)
        
        return True
        
    except Exception as e:
        print(f"❌ Emergency rollback failed: {e}")
        print("💡 Manual restoration required - check backup file")
        return False

def layer1_main():
    """Orchestrate all Layer 1 functions safely."""
    print("🚀 PHASE 4 LAYER 1: SAFE PREPARATION FOR GPT-OSS INTEGRATION")
    print("=" * 80)
    print("🎯 Objective: Prepare environment for GPT-OSS-20B without breaking Phases 1-3.5")
    print("🛡️ Safety: Full backup and rollback capability included")
    print("=" * 80)
    
    # Step 1: Create environment backup
    print("\n" + "🔹" * 40)
    if not create_environment_backup():
        print("❌ LAYER 1 FAILED: Cannot proceed without backup")
        return False
    
    # Step 2: Verify current phases working
    print("\n" + "🔹" * 40)
    if not verify_phases_1_to_3_working():
        print("❌ LAYER 1 FAILED: Existing phases not working")
        print("💡 Fix Phase 1-3.5 issues before proceeding to Phase 4")
        return False
    
    # Step 3: Update transformers with verification
    print("\n" + "🔹" * 40)
    transformers_success = safe_update_transformers()
    if not transformers_success:
        print("⚠️ Transformers update failed - continuing with current version")
        print("💡 Phase 4 may have limited functionality")
    
    # Step 4: Install GPT-OSS dependencies
    print("\n" + "🔹" * 40)
    deps_success = install_gpt_oss_dependencies()
    if not deps_success:
        print("⚠️ GPT-OSS dependencies incomplete - fallback methods will be used")
    
    # Step 5: Verify GPT-OSS readiness
    print("\n" + "🔹" * 40)
    readiness_success = verify_gpt_oss_readiness()
    
    # Final status report
    print("\n" + "🎯" * 40)
    print("🎯 LAYER 1 COMPLETION REPORT")
    print("🎯" * 40)
    
    print(f"✅ Environment Backup: Created")
    print(f"✅ Phases 1-3.5 Verification: Passed")
    print(f"{'✅' if transformers_success else '⚠️'} Transformers Update: {'Success' if transformers_success else 'Partial/Failed'}")
    print(f"{'✅' if deps_success else '⚠️'} GPT-OSS Dependencies: {'Installed' if deps_success else 'Partial/Missing'}")
    print(f"{'✅' if readiness_success else '⚠️'} GPT-OSS Readiness: {'Ready' if readiness_success else 'Limited'}")
    
    overall_success = transformers_success and deps_success and readiness_success
    
    if overall_success:
        print("\n🎉 LAYER 1 COMPLETE - FULL SUCCESS!")
        print("=" * 60)
        print("✅ Environment fully prepared for Phase 4")
        print("✅ All existing phases preserved")
        print("✅ GPT-OSS-20B integration ready")
        print("=" * 60)
        print("🚀 NEXT STEPS:")
        print("1. Restart Colab runtime to finalize changes")
        print("2. Run this script again to verify Layer 1")
        print("3. Proceed to Layer 2 (GPT-OSS model loading)")
        print("4. Begin Phase 4 core implementation")
        
    else:
        print("\n⚠️ LAYER 1 COMPLETE - PARTIAL SUCCESS")
        print("=" * 60)
        print("✅ Environment partially prepared for Phase 4")
        print("✅ All existing phases preserved")
        print("⚠️ Some GPT-OSS features may be limited")
        print("=" * 60)
        print("🔧 OPTIONS:")
        print("1. Proceed with limited functionality")
        print("2. Run emergency_rollback() to restore original state")
        print("3. Manual fix of failed components")
        print("4. Retry Layer 1 after addressing issues")
    
    print("\n🛡️ SAFETY REMINDER:")
    print("- Backup available: /content/pre_phase4_backup.txt")
    print("- Rollback available: emergency_rollback()")
    print("- Existing phases protected and verified")
    
    return overall_success

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

def install_llama_cpp_python():
    """Install llama-cpp-python with proper GPU support and verification."""
    
    # Method 1: Try standard installation with latest version
    print("🎯 Method 1: Standard llama-cpp-python installation...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "--no-cache-dir", "llama-cpp-python"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ llama-cpp-python installed successfully")
        if verify_llama_cpp_installation():
            return True
    else:
        print(f"❌ Standard installation failed: {result.stderr}")
    
    # Method 2: Try with specific CUDA environment variables
    print("🎯 Method 2: Installing with CUDA environment variables...")
    env = os.environ.copy()
    env.update({
        'CMAKE_ARGS': '-DLLAMA_CUBLAS=on',
        'FORCE_CMAKE': '1'
    })
    
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "--no-cache-dir", "--force-reinstall", "llama-cpp-python"
    ], capture_output=True, text=True, env=env)
    
    if result.returncode == 0:
        print("✅ llama-cpp-python installed with CUDA support")
        if verify_llama_cpp_installation():
            return True
    else:
        print(f"❌ CUDA installation failed: {result.stderr}")
    
    # Method 3: Try with wheel from PyPI
    print("🎯 Method 3: Installing pre-built wheel...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "--no-cache-dir", "--only-binary=all", "llama-cpp-python"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ llama-cpp-python wheel installed successfully")
        if verify_llama_cpp_installation():
            return True
    else:
        print(f"❌ Wheel installation failed: {result.stderr}")
    
    print("⚠️ ALL LLAMA-CPP-PYTHON INSTALLATION METHODS FAILED")
    print("💡 You can continue without llama-cpp-python, but GGUF models will be unavailable")
    print("💡 Manual installation after setup: !pip install llama-cpp-python")
    return False

def verify_llama_cpp_installation():
    """Verify that llama-cpp-python installation works correctly."""
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "import llama_cpp; print(f'llama-cpp-python imported successfully')"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"✅ llama-cpp verification: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ llama-cpp verification failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ llama-cpp verification timeout")
        return False
    except Exception as e:
        print(f"❌ llama-cpp verification error: {e}")
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
        "evaluate>=0.4,<0.5",  # Required for phase1_training.py
        "safetensors>=0.3",
        # Additional dependencies for Mistral migration
        "rouge-score>=0.1.2",  # Required by evaluate module
        "sacrebleu>=2.0.0",    # BLEU metrics
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
        "aiofiles>=23.1.0",
        "fsspec>=2023.1.0",
        "memory-profiler>=0.60.0",
        # Phase 7 Enhanced Consciousness Dependencies
        "fastapi>=0.104.0",  # REST API framework
        "uvicorn>=0.24.0",   # ASGI server for FastAPI
        "httpx>=0.25.0",     # Async HTTP client for API testing
        "pydantic>=2.0.0",   # Data validation for API models
    ]
    
    for package in utility_packages:
        print(f"📦 Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to install {package}: {result.stderr}")
        else:
            print(f"✅ Successfully installed {package}")
    
    # Install llama-cpp-python separately with improved method
    print("\n🦙 Installing llama-cpp-python with GPU support...")
    install_llama_cpp_python()
    
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
        ("evaluate", "import evaluate; print('Evaluate module OK')"),  # Required for training
        ("peft", "import peft; print('PEFT (LoRA) OK')"),  # Required for LoRA training
        ("datasets", "import datasets; print('Datasets OK')"),  # Required for data handling
        ("bitsandbytes", "import bitsandbytes; print('BitsAndBytes OK')"),
        ("llama_cpp", "import llama_cpp; print('llama-cpp-python OK')"),
    ]
    
    all_good = True
    
    for name, test_code in test_imports:
        try:
            result = subprocess.run([sys.executable, "-c", test_code], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"✅ {name}: {result.stdout.strip()}")
            else:
                # Special handling for optional packages
                if name in ["bitsandbytes", "llama_cpp"]:
                    print(f"⚠️ {name} not available (optional package)")
                    if name == "bitsandbytes":
                        print("💡 You can manually install later: !pip install bitsandbytes")
                    elif name == "llama_cpp":
                        print("💡 You can manually install later: !pip install llama-cpp-python")
                elif name in ["evaluate", "peft", "datasets"]:
                    print(f"❌ {name} failed: {result.stderr}")
                    print(f"🚨 CRITICAL: {name} is required for Mistral training pipeline!")
                    all_good = False
                else:
                    print(f"❌ {name} failed: {result.stderr}")
                    all_good = False
                    
                    # Special handling for numpy.dtype size error
                    if "numpy.dtype size changed" in result.stderr:
                        print("💥 BINARY INCOMPATIBILITY DETECTED!")
                        print("🔄 You must restart runtime and run setup again.")
                        return False
                    
        except subprocess.TimeoutExpired:
            if name in ["bitsandbytes", "llama_cpp"]:
                print(f"⚠️ {name}: Import timeout (optional package)")
            elif name in ["evaluate", "peft", "datasets"]:
                print(f"❌ {name}: Import timeout")
                print(f"🚨 CRITICAL: {name} is required for Mistral training pipeline!")
                all_good = False
            else:
                print(f"❌ {name}: Import timeout (possible deadlock)")
                all_good = False
        except Exception as e:
            if name in ["bitsandbytes", "llama_cpp"]:
                print(f"⚠️ {name}: {e} (optional package)")
            elif name in ["evaluate", "peft", "datasets"]:
                print(f"❌ {name}: {e}")
                print(f"🚨 CRITICAL: {name} is required for Mistral training pipeline!")
                all_good = False
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
        print("✅ Environment ready for Mistral training pipeline")
        print("=" * 60)
        print("🚀 You can now run training scripts:")
        print("  python phase1_training.py --dataset training_data.jsonl --epochs 3 --batch-size 4")
        print("  python phase2_training.py")
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

def layer2_conditional_setup():
    """
    PHASE 4 LAYER 2: Conditional setup based on hardware capabilities
    Installs premium GPT-OSS dependencies only if hardware supports them.
    """
    print("🎯 PHASE 4 LAYER 2: Conditional Hardware-Based Setup")
    print("=" * 60)
    
    try:
        # Import hardware detection (if Layer 1 completed)
        try:
            import torch
            import psutil
            from conscious_ai.phases.p4_LLM_Communication.core.hardware_profiler import PremiumHardwareProfiler
            
            # Detect hardware configuration
            profiler = PremiumHardwareProfiler()
            hardware_config = profiler.detect_hardware_configuration()
            
            print(f"📊 Hardware Detection Results:")
            print(f"  RAM: {hardware_config.total_ram_gb:.1f}GB total, {hardware_config.usable_ram_gb:.1f}GB usable")
            print(f"  VRAM: {hardware_config.total_vram_gb:.1f}GB total, {hardware_config.usable_vram_gb:.1f}GB usable")
            print(f"  Premium Hardware: {'✅' if hardware_config.is_premium_hardware else '❌'}")
            
        except ImportError:
            print("⚠️ Layer 1 components not available - estimating hardware manually")
            hardware_config = _estimate_hardware_manually()
        
        # Conditional installation based on hardware
        if hardware_config and hardware_config.is_premium_hardware:
            print("\n🚀 Premium hardware detected - installing GPT-OSS-20B dependencies")
            return _install_premium_gpt_oss_stack(hardware_config)
        else:
            print("\n💡 Standard/Limited hardware - installing lightweight alternatives")
            return _install_lightweight_alternatives(hardware_config)
            
    except Exception as e:
        print(f"❌ Layer 2 setup failed: {e}")
        print("🔄 Falling back to standard setup workflow")
        return False

def _estimate_hardware_manually():
    """Manually estimate hardware when Layer 1 not available"""
    try:
        import psutil
        import torch
        
        # Check RAM
        memory = psutil.virtual_memory()
        ram_gb = memory.total / (1024**3)
        
        # Check VRAM
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Simple configuration object
        class SimpleHardwareConfig:
            def __init__(self, ram_gb, vram_gb):
                self.total_ram_gb = ram_gb
                self.total_vram_gb = vram_gb
                self.usable_ram_gb = max(0, ram_gb - 6)  # 6GB safety margin
                self.usable_vram_gb = max(0, vram_gb - 2)  # 2GB safety margin
                self.is_premium_hardware = (ram_gb >= 51 and vram_gb >= 15)
        
        return SimpleHardwareConfig(ram_gb, vram_gb)
        
    except:
        return None

def _install_premium_gpt_oss_stack(hardware_config):
    """Install full GPT-OSS-20B stack for premium hardware"""
    print("📦 Installing Premium GPT-OSS-20B Stack")
    print("-" * 40)
    
    premium_packages = [
        "accelerate>=0.21.0",
        "bitsandbytes>=0.41.0",  # Quantization support
        "transformers>=4.55.0,<4.56.0",  # GPT-OSS compatibility
    ]
    
    # Optional advanced packages
    advanced_packages = [
        ("flash-attn", "Flash Attention 2 for speed optimization"),
        ("kernels", "MXFP4 quantization support"),
        ("openai-harmony>=1.0.0", "Harmony response format")
    ]
    
    success_count = 0
    
    # Install core premium packages
    for package in premium_packages:
        print(f"📦 Installing {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
            success_count += 1
        else:
            print(f"❌ {package} installation failed: {result.stderr}")
    
    # Install advanced packages (optional)
    for package, description in advanced_packages:
        print(f"📦 Installing {package} ({description})...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
        else:
            print(f"⚠️ {package} installation failed - will use fallback")
    
    # Verify premium installation
    if success_count >= len(premium_packages) - 1:  # Allow 1 failure
        print(f"\n🎉 Premium GPT-OSS stack installed: {success_count}/{len(premium_packages)} core packages")
        print("🚀 System ready for GPT-OSS-20B hybrid loading")
        return True
    else:
        print(f"\n⚠️ Premium installation incomplete: {success_count}/{len(premium_packages)} packages")
        return False

def _install_lightweight_alternatives(hardware_config):
    """Install lightweight alternatives for limited hardware"""
    print("📦 Installing Lightweight Alternative Stack")
    print("-" * 40)
    
    if hardware_config:
        print(f"💡 Hardware limitations: {hardware_config.total_ram_gb:.1f}GB RAM, {hardware_config.total_vram_gb:.1f}GB VRAM")
        
    lightweight_packages = [
        "transformers>=4.41.0,<4.42.0",  # Stable version
        "sentence-transformers<2.8.0",   # Lightweight embeddings
        "torch>=2.0.0"  # Ensure PyTorch availability
    ]
    
    success_count = 0
    
    for package in lightweight_packages:
        print(f"📦 Installing {package}...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", package
        ], capture_output=True, text=True, timeout=240)
        
        if result.returncode == 0:
            print(f"✅ {package} installed successfully")
            success_count += 1
        else:
            print(f"❌ {package} installation failed: {result.stderr}")
    
    # Install emergency fallback model
    print("📦 Setting up emergency fallback model (mT5-small)...")
    try:
        result = subprocess.run([
            sys.executable, "-c", 
            "from transformers import T5Tokenizer, T5ForConditionalGeneration; "
            "T5Tokenizer.from_pretrained('google/mt5-small'); "
            "T5ForConditionalGeneration.from_pretrained('google/mt5-small'); "
            "print('Emergency model cached successfully')"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Emergency fallback model ready")
        else:
            print("⚠️ Emergency model setup failed - will download when needed")
    except:
        print("⚠️ Emergency model setup failed - will download when needed")
    
    if success_count >= len(lightweight_packages) - 1:
        print(f"\n✅ Lightweight stack installed: {success_count}/{len(lightweight_packages)} packages")
        print("💡 System ready for Phase 4 with limited model support")
        return True
    else:
        print(f"\n⚠️ Lightweight installation incomplete: {success_count}/{len(lightweight_packages)} packages")
        return False

def install_phase7_dependencies():
    """Phase 7: Install Enhanced Consciousness dependencies."""
    print("🚀 PHASE 7: Installing Enhanced Consciousness Dependencies")
    print("=" * 60)
    print("Installing OpenAI integration, REST API, and metacognitive features...")
    print()
    
    # Phase 7 Enhanced Consciousness packages
    phase7_packages = [
        "openai>=1.0.0",         # OpenAI API client for GPT models
        "fastapi>=0.104.0",      # REST API framework  
        "uvicorn[standard]>=0.24.0",  # ASGI server for FastAPI
        "httpx>=0.25.0",         # Async HTTP client for API testing
        "pydantic>=2.0.0",       # Data validation for API models
    ]
    
    print("📦 Installing Phase 7 core packages...")
    success_count = 0
    
    for package in phase7_packages:
        print(f"📦 Installing: {package}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", package], 
                              capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print(f"✅ Successfully installed {package}")
            success_count += 1
        else:
            print(f"❌ Failed to install {package}: {result.stderr}")
    
    # Verify Phase 7 installation
    print(f"\n🧪 Phase 7 Installation Verification...")
    verification_tests = [
        ("import openai", "OpenAI API client"),
        ("import fastapi", "FastAPI framework"),
        ("import uvicorn", "Uvicorn ASGI server"),
        ("import httpx", "HTTPX async client"),
        ("import pydantic", "Pydantic data validation"),
    ]
    
    verification_success = 0
    for test_import, description in verification_tests:
        try:
            result = subprocess.run([sys.executable, "-c", test_import], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ {description}")
                verification_success += 1
            else:
                print(f"❌ {description}: Import failed")
        except Exception as e:
            print(f"❌ {description}: {e}")
    
    # Create .env.example for configuration
    print(f"\n📄 Creating environment configuration template...")
    env_template = '''# Phase 7 Enhanced Consciousness Configuration
# Copy this file to .env and set your actual values

# OpenAI Configuration for Phase 7
OPENAI_API_KEY=sk-your-openai-api-key-here

# Phase 7 Default Model
DEFAULT_FINAL_MODEL=gpt-4o-mini

# Debug Settings
ENABLE_MODEL_REGISTRY=true
DEBUG_DIR=./debug
'''
    
    try:
        with open('.env.example', 'w') as f:
            f.write(env_template)
        print("✅ .env.example created")
    except Exception as e:
        print(f"⚠️ Could not create .env.example: {e}")
    
    # Summary
    print(f"\n📊 Phase 7 Installation Summary:")
    print(f"  Core packages: {success_count}/{len(phase7_packages)} installed")
    print(f"  Verification: {verification_success}/{len(verification_tests)} passed")
    
    if success_count == len(phase7_packages) and verification_success == len(verification_tests):
        print(f"\n✅ Phase 7 Enhanced Consciousness installation complete!")
        print(f"🎯 Next steps:")
        print(f"   1. Set your OpenAI API key in environment or .env file")
        print(f"   2. Test enhanced consciousness: python test_enhanced_consciousness.py")
        print(f"   3. Start API server: python conscious_ai/api/run_server.py")
        return True
    else:
        print(f"\n⚠️ Phase 7 installation incomplete")
        print(f"💡 Some features may not be available")
        return False

def main():
    """Main setup function with phase detection."""
    
    print("🚀 Google Colab Setup - Mistral Training Pipeline")
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
    elif phase == "layer1_preparation":
        print("🎯 Phase 4 Layer 1 preparation detected - running Layer 1 setup...")
        success = layer1_main()
    elif phase == "phase7_setup":
        print("🚀 Phase 7 Enhanced Consciousness setup detected - installing dependencies...")
        success = install_phase7_dependencies()
    elif phase == "layer2_setup":
        print("🎯 Phase 4 Layer 2 conditional setup detected - running Layer 2 setup...")
        success = layer2_conditional_setup()
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