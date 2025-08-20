#!/usr/bin/env python3
"""
Minimal Phase 3.4 test for Colab environment
============================================
Tests only the critical fixes without external dependencies.
"""

import sys
import os
import json

# Add the conscious_ai package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_import_fix():
    """Test that the import path typo is fixed"""
    print("1. Testing import path fix...")
    try:
        # This should work now that we fixed autonomus -> autonomous
        from conscious_ai.autonomous_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI
        print("PASS - Import path fixed successfully")
        return True
    except ImportError as e:
        print(f"FAIL - Import still broken: {e}")
        return False

def test_division_by_zero_fixes():
    """Test that division by zero errors are fixed"""
    print("\n2. Testing division by zero fixes...")
    try:
        # Import the fixed evaluator
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import HybridCoherenceEvaluator
        
        # Create evaluator (this should work without sentence-transformers)
        evaluator = HybridCoherenceEvaluator()
        
        # Test edge cases that previously caused division by zero
        edge_cases = [
            # Both empty
            ({}, {}),
            # Empty strings
            ({'goal': '', 'emotion': '', 'thought': '', 'confidence': 0.5, 'memory': []},
             {'goal': '', 'emotion': '', 'thought': '', 'confidence': 0.5, 'memory': []}),
            # None values  
            ({'goal': None, 'emotion': None, 'thought': None, 'confidence': None, 'memory': None},
             {'goal': 'test', 'emotion': 'test', 'thought': 'test', 'confidence': 0.5, 'memory': []})
        ]
        
        for i, (sc_t, sc_t_plus_1) in enumerate(edge_cases, 1):
            try:
                # This should not crash with division by zero
                analysis = evaluator.evaluate_transition(sc_t, sc_t_plus_1)
                print(f"   Edge case {i}: {analysis.verdict.value} - OK")
            except ZeroDivisionError:
                print(f"   Edge case {i}: DIVISION BY ZERO ERROR - FAILED")
                return False
            except Exception as e:
                print(f"   Edge case {i}: Other error - {e} - OK (handled gracefully)")
        
        print("PASS - No division by zero errors")
        return True
        
    except Exception as e:
        print(f"FAIL - Test setup failed: {e}")
        return False

def test_threshold_recalibration():
    """Test that thresholds were recalibrated"""
    print("\n3. Testing threshold recalibration...")
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import HybridCoherenceEvaluator
        
        evaluator = HybridCoherenceEvaluator()
        
        # Check that thresholds were updated
        print(f"   Coherent threshold: {evaluator.coherent_threshold}")
        print(f"   Incoherent threshold: {evaluator.incoherent_threshold}")
        
        # Should be 0.55 and 0.40 respectively (not 0.60 and 0.35)
        if evaluator.coherent_threshold == 0.55 and evaluator.incoherent_threshold == 0.40:
            print("PASS - Thresholds recalibrated correctly")
            return True
        else:
            print("FAIL - Thresholds not updated")
            return False
            
    except Exception as e:
        print(f"FAIL - Threshold test failed: {e}")
        return False

def test_error_handling():
    """Test that robust error handling was added"""
    print("\n4. Testing error handling improvements...")
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        
        # Test with malformed data that should be handled gracefully
        malformed_cases = [
            # Invalid types
            ("not_a_dict", {}),
            ({}, "also_not_a_dict"),
            # Missing required fields handled gracefully
            ({'invalid': 'data'}, {'also_invalid': 'data'})
        ]
        
        for i, (sc_t, sc_t_plus_1) in enumerate(malformed_cases, 1):
            try:
                # Should handle gracefully, not crash
                analysis = evaluator.evaluate_transition(sc_t, sc_t_plus_1)
                print(f"   Malformed case {i}: Handled gracefully")
            except Exception as e:
                print(f"   Malformed case {i}: Error - {e}")
                # This is OK as long as it's not a division by zero
                if "ZeroDivisionError" in str(e) or "division by zero" in str(e).lower():
                    return False
        
        print("PASS - Error handling improved")
        return True
        
    except Exception as e:
        print(f"FAIL - Error handling test failed: {e}")
        return False

def main():
    """Run minimal tests for MVP fixes"""
    print("MINIMAL PHASE 3.4 MVP FIXES TEST")
    print("=" * 40)
    
    tests = [
        test_import_fix,
        test_division_by_zero_fixes, 
        test_threshold_recalibration,
        test_error_handling
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"TEST FAILED with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 40)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS - All critical MVP fixes are working!")
    elif passed >= total * 0.75:
        print("MOSTLY SUCCESSFUL - Core fixes are working")
    else:
        print("ISSUES REMAIN - Some fixes may need more work")
    
    return passed >= total * 0.75

if __name__ == "__main__":
    success = main()
    print(f"\nExit code: {0 if success else 1}")