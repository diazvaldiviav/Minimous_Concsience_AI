#!/usr/bin/env python3
"""
Hardware Detection Debug Script for Google Colab
=================================================
Debug script to analyze hardware detection issues in Google Colab.
Use this to troubleshoot GPU detection and hardware profiling problems.
"""

import logging
import sys

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_torch_cuda():
    """Debug PyTorch CUDA installation and GPU detection"""
    print("🔍 DEBUGGING PYTORCH CUDA CONFIGURATION")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ Device count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                print(f"\n📱 GPU {i}:")
                device_props = torch.cuda.get_device_properties(i)
                print(f"   Name: {device_props.name}")
                print(f"   Total memory: {device_props.total_memory / 1024**3:.2f} GB")
                print(f"   Compute capability: {device_props.major}.{device_props.minor}")
                
                # Test different attribute names for multiprocessor count
                mp_attrs = ['multiprocessor_count', 'multi_processor_count', 'sm_count']
                for attr in mp_attrs:
                    if hasattr(device_props, attr):
                        value = getattr(device_props, attr)
                        print(f"   {attr}: {value}")
                        break
                else:
                    print("   ⚠️ No multiprocessor count attribute found")
                
                # Memory info
                print(f"   Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
                print(f"   Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
                
                # Test tensor creation
                try:
                    test_tensor = torch.tensor([1.0]).cuda(i)
                    print(f"   ✅ Tensor creation successful")
                except Exception as e:
                    print(f"   ❌ Tensor creation failed: {e}")
        else:
            print("❌ CUDA not available")
            print("   Possible reasons:")
            print("   - No GPU allocated in Colab")
            print("   - GPU runtime not selected")
            print("   - PyTorch CPU-only installation")
            
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
    except Exception as e:
        print(f"❌ PyTorch CUDA debug failed: {e}")

def debug_hardware_profiler():
    """Debug the premium hardware profiler"""
    print("\n🔍 DEBUGGING HARDWARE PROFILER")
    print("=" * 60)
    
    try:
        from conscious_ai.phases.p4_LLM_Communication.core.hardware_profiler import PremiumHardwareProfiler
        print("✅ Hardware profiler imported successfully")
        
        profiler = PremiumHardwareProfiler()
        
        # Test RAM detection
        print("\n📊 RAM Detection:")
        ram_info = profiler._detect_ram_configuration()
        for key, value in ram_info.items():
            print(f"   {key}: {value}")
        
        # Test GPU detection
        print("\n🎮 GPU Detection:")
        gpu_info = profiler._detect_gpu_configuration()
        for key, value in gpu_info.items():
            print(f"   {key}: {value}")
        
        # Full hardware configuration
        print("\n💻 Full Hardware Configuration:")
        config = profiler.detect_hardware_configuration()
        print(f"   Total RAM: {config.total_ram_gb:.2f} GB")
        print(f"   Available RAM: {config.available_ram_gb:.2f} GB")
        print(f"   Usable RAM: {config.usable_ram_gb:.2f} GB")
        print(f"   Total VRAM: {config.total_vram_gb:.2f} GB")
        print(f"   Available VRAM: {config.available_vram_gb:.2f} GB")
        print(f"   Usable VRAM: {config.usable_vram_gb:.2f} GB")
        print(f"   GPU Name: {config.gpu_name}")
        print(f"   CUDA Version: {config.cuda_version}")
        print(f"   Architecture Type: {config.architecture_type}")
        print(f"   Premium Hardware: {config.is_premium_hardware}")
        print(f"   Memory Efficiency Score: {config.memory_efficiency_score:.1f}%")
        
    except ImportError as e:
        print(f"❌ Hardware profiler import failed: {e}")
        print("   This is expected if dependencies are missing")
    except Exception as e:
        print(f"❌ Hardware profiler failed: {e}")

def debug_system_info():
    """Debug general system information"""
    print("\n🔍 DEBUGGING SYSTEM INFORMATION")
    print("=" * 60)
    
    try:
        import platform
        print(f"✅ Platform: {platform.platform()}")
        print(f"✅ Python version: {platform.python_version()}")
        print(f"✅ Architecture: {platform.architecture()}")
        print(f"✅ Machine: {platform.machine()}")
        print(f"✅ Processor: {platform.processor()}")
        
        # Check if running in Colab
        try:
            import google.colab
            print("✅ Running in Google Colab")
            
            # Check available memory
            import psutil
            memory = psutil.virtual_memory()
            print(f"✅ Total RAM: {memory.total / 1024**3:.2f} GB")
            print(f"✅ Available RAM: {memory.available / 1024**3:.2f} GB")
            print(f"✅ Used RAM: {memory.used / 1024**3:.2f} GB")
            print(f"✅ Memory percent: {memory.percent:.1f}%")
            
        except ImportError:
            print("❌ Not running in Google Colab or missing dependencies")
            
    except Exception as e:
        print(f"❌ System info debug failed: {e}")

def main():
    """Run all debug functions"""
    print("🚀 HARDWARE DETECTION DEBUG SCRIPT")
    print("🎯 Analyzing hardware configuration issues")
    print("=" * 80)
    
    debug_system_info()
    debug_torch_cuda()
    debug_hardware_profiler()
    
    print("\n" + "=" * 80)
    print("🎯 DEBUG COMPLETE")
    print("💡 If issues persist:")
    print("   1. Ensure GPU runtime is selected in Colab")
    print("   2. Check that CUDA dependencies are installed")
    print("   3. Restart runtime and try again")
    print("   4. Verify PyTorch CUDA installation")
    print("=" * 80)

if __name__ == "__main__":
    main()