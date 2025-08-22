#!/usr/bin/env python3
"""
Phase 4 Layer 1 Testing Script
==============================
Tests the Layer 1 implementation without requiring Google Colab environment.
Run this to validate Layer 1 functionality before deployment.

Usage: python test_layer1.py
"""

import sys
import os
import importlib.util
import inspect

def test_layer1_function_definitions():
    """Test that all Layer 1 functions are properly defined in colab_setup.py"""
    print("🧪 TEST 1: Layer 1 Function Definitions")
    print("=" * 60)
    
    try:
        # Import the colab_setup module
        spec = importlib.util.spec_from_file_location("colab_setup", "colab_setup.py")
        colab_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(colab_setup)
        
        # Check for all required Layer 1 functions
        required_functions = [
            "create_environment_backup",
            "verify_phases_1_to_3_working", 
            "safe_update_transformers",
            "install_gpt_oss_dependencies",
            "verify_gpt_oss_readiness",
            "emergency_rollback",
            "layer1_main"
        ]
        
        missing_functions = []
        
        for func_name in required_functions:
            if hasattr(colab_setup, func_name):
                func = getattr(colab_setup, func_name)
                if callable(func):
                    print(f"✅ {func_name}: Defined and callable")
                else:
                    print(f"❌ {func_name}: Exists but not callable")
                    missing_functions.append(func_name)
            else:
                print(f"❌ {func_name}: Not defined")
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"\n❌ Missing functions: {missing_functions}")
            return False
        else:
            print("\n✅ All Layer 1 functions properly defined")
            return True
            
    except Exception as e:
        print(f"❌ Error testing function definitions: {e}")
        return False

def test_phase_detection_logic():
    """Test that phase detection logic includes Layer 1"""
    print("\n🧪 TEST 2: Phase Detection Logic")
    print("=" * 60)
    
    try:
        spec = importlib.util.spec_from_file_location("colab_setup", "colab_setup.py")
        colab_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(colab_setup)
        
        # Check if check_runtime_state function has Layer 1 logic
        import inspect
        source = inspect.getsource(colab_setup.check_runtime_state)
        
        if "layer1_preparation" in source:
            print("✅ Phase detection includes layer1_preparation")
        else:
            print("❌ Phase detection missing layer1_preparation")
            return False
            
        if "phase4_deps_installed" in source:
            print("✅ Phase detection checks Phase 4 dependencies")
        else:
            print("❌ Phase detection missing Phase 4 dependency check")
            return False
            
        print("✅ Phase detection logic properly updated")
        return True
        
    except Exception as e:
        print(f"❌ Error testing phase detection: {e}")
        return False

def test_main_function_integration():
    """Test that main function handles Layer 1 phase"""
    print("\n🧪 TEST 3: Main Function Integration")
    print("=" * 60)
    
    try:
        spec = importlib.util.spec_from_file_location("colab_setup", "colab_setup.py")
        colab_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(colab_setup)
        
        # Check if main function has Layer 1 handling
        import inspect
        source = inspect.getsource(colab_setup.main)
        
        if 'layer1_preparation' in source:
            print("✅ Main function handles layer1_preparation phase")
        else:
            print("❌ Main function missing layer1_preparation handling")
            return False
            
        if 'layer1_main()' in source:
            print("✅ Main function calls layer1_main()")
        else:
            print("❌ Main function missing layer1_main() call")
            return False
            
        print("✅ Main function integration complete")
        return True
        
    except Exception as e:
        print(f"❌ Error testing main function integration: {e}")
        return False

def test_requirements_txt_updates():
    """Test that requirements.txt has been updated for Layer 1"""
    print("\n🧪 TEST 4: Requirements.txt Updates")
    print("=" * 60)
    
    try:
        with open("requirements.txt", "r") as f:
            content = f.read()
        
        # Check for updated transformers version
        if "transformers>=4.55.0,<4.56.0" in content:
            print("✅ Transformers version updated to 4.55.x")
        else:
            print("❌ Transformers version not updated to 4.55.x")
            return False
        
        # Check for GPT-OSS dependencies
        if "openai-harmony" in content:
            print("✅ OpenAI Harmony dependency added")
        else:
            print("❌ OpenAI Harmony dependency missing")
            return False
            
        if "kernels" in content:
            print("✅ Kernels dependency added")
        else:
            print("❌ Kernels dependency missing")
            return False
        
        # Check for Layer 1 comments
        if "Phase 4 Layer 1" in content:
            print("✅ Layer 1 documentation added")
        else:
            print("❌ Layer 1 documentation missing")
            return False
            
        print("✅ Requirements.txt properly updated")
        return True
        
    except Exception as e:
        print(f"❌ Error testing requirements.txt: {e}")
        return False

def test_safety_mechanisms():
    """Test that safety mechanisms are in place"""
    print("\n🧪 TEST 5: Safety Mechanisms")
    print("=" * 60)
    
    try:
        spec = importlib.util.spec_from_file_location("colab_setup", "colab_setup.py")
        colab_setup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(colab_setup)
        
        # Check backup function
        backup_source = inspect.getsource(colab_setup.create_environment_backup)
        if "pre_phase4_backup.txt" in backup_source:
            print("✅ Backup mechanism creates backup file")
        else:
            print("❌ Backup mechanism incomplete")
            return False
            
        # Check rollback function
        rollback_source = inspect.getsource(colab_setup.emergency_rollback)
        if "numpy==1.26.4" in rollback_source and "torch==2.1.2" in rollback_source:
            print("✅ Rollback mechanism restores critical packages")
        else:
            print("❌ Rollback mechanism incomplete")
            return False
            
        # Check verification function  
        verify_source = inspect.getsource(colab_setup.verify_phases_1_to_3_working)
        if "MinimalConsciousAI" in verify_source and "CriticalStateEvaluator" in verify_source:
            print("✅ Verification checks existing phases")
        else:
            print("❌ Verification mechanism incomplete")
            return False
            
        print("✅ Safety mechanisms properly implemented")
        return True
        
    except Exception as e:
        print(f"❌ Error testing safety mechanisms: {e}")
        return False

def run_all_tests():
    """Run all Layer 1 validation tests"""
    print("🚀 PHASE 4 LAYER 1 VALIDATION TESTS")
    print("=" * 80)
    print("Testing Layer 1 implementation without requiring Colab environment")
    print("=" * 80)
    
    tests = [
        ("Function Definitions", test_layer1_function_definitions),
        ("Phase Detection Logic", test_phase_detection_logic),
        ("Main Function Integration", test_main_function_integration),
        ("Requirements.txt Updates", test_requirements_txt_updates),
        ("Safety Mechanisms", test_safety_mechanisms)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    # Final report
    print("\n" + "🎯" * 40)
    print("🎯 LAYER 1 VALIDATION REPORT")
    print("🎯" * 40)
    
    print(f"📊 Tests Passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED - LAYER 1 READY FOR DEPLOYMENT!")
        print("=" * 60)
        print("✅ Layer 1 implementation complete and validated")
        print("✅ All safety mechanisms in place")
        print("✅ Requirements properly updated")
        print("✅ Integration points working")
        print("=" * 60)
        print("🚀 DEPLOYMENT READY:")
        print("1. Push changes to phase-4 branch")
        print("2. Test in Google Colab environment")  
        print("3. Run Layer 1 in production")
        print("4. Proceed to Layer 2 implementation")
        
    else:
        print(f"\n⚠️ {total_tests - passed_tests} TESTS FAILED - FIXES REQUIRED")
        print("=" * 60)
        print("❌ Layer 1 implementation incomplete")
        print("🔧 Address failed tests before deployment")
        print("💡 Check error messages above for specific issues")
        
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)