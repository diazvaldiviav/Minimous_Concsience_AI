#!/usr/bin/env python3
"""
Simple Phase 3.4 Hybrid Fix Test
================================
Basic test to verify the hybrid evaluator is working correctly.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Test basic functionality of the hybrid evaluator"""
    print("Testing basic hybrid evaluator functionality...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator, CoherenceVerdict, create_critical_evaluator
        )
        
        # Create evaluator
        evaluator = create_critical_evaluator()
        print("SUCCESS: CriticalStateEvaluator created")
        
        # Test states
        coherent_prev = {
            "goal": "understand consciousness",
            "emotion": "curious",
            "thought": "I wonder about awareness",
            "confidence": 0.6,
            "memory": ["exploring consciousness"]
        }
        
        coherent_curr = {
            "goal": "analyze consciousness patterns", 
            "emotion": "analytical",
            "thought": "I examine awareness structures",
            "confidence": 0.65,
            "memory": ["exploring consciousness", "finding patterns"]
        }
        
        # Mock generator
        def mock_generator(**kwargs):
            return coherent_curr
        
        generation_context = {"temperature": 0.7}
        
        # Test evaluation
        final_state, eval_result = evaluator.evaluate_and_correct_state(
            coherent_prev,
            coherent_curr,
            mock_generator,
            generation_context
        )
        
        print(f"Evaluation verdict: {eval_result.verdict.value}")
        print(f"Attempts made: {eval_result.attempts_made}")
        print(f"Method: {eval_result.evaluation_method}")
        print(f"Confidence: {eval_result.confidence_score:.3f}")
        
        # Check results
        if eval_result.verdict == CoherenceVerdict.COHERENT:
            print("SUCCESS: Coherent detection working")
            return True
        else:
            print("FAIL: Expected coherent verdict")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_incoherent_detection():
    """Test incoherent state detection"""
    print("\nTesting incoherent detection...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        
        # Incoherent transition
        prev_state = {
            "goal": "understand consciousness",
            "emotion": "curious",
            "thought": "exploring awareness",
            "confidence": 0.6,
            "memory": ["consciousness thoughts"]
        }
        
        incoherent_state = {
            "goal": "bake cookies",
            "emotion": "confused",
            "thought": "purple elephants are dancing",
            "confidence": 0.1,
            "memory": ["random stuff"]
        }
        
        result = evaluator.evaluate_transition(prev_state, incoherent_state)
        
        print(f"Incoherent test verdict: {result.verdict.value}")
        
        if result.verdict == CoherenceVerdict.INCOHERENT:
            print("SUCCESS: Incoherent detection working")
            return True
        else:
            print("FAIL: Expected incoherent verdict")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def test_statistics():
    """Test statistics interface"""
    print("\nTesting statistics interface...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator
        )
        
        evaluator = CriticalStateEvaluator()
        stats = evaluator.get_evaluation_statistics()
        
        print(f"Evaluation method: {stats.get('evaluation_method', 'unknown')}")
        print(f"ML available: {stats.get('ml_available', 'unknown')}")
        print(f"Hybrid available: {stats.get('hybrid_available', 'unknown')}")
        
        if stats.get('evaluation_method') == 'hybrid_approach':
            print("SUCCESS: Using hybrid approach")
            return True
        else:
            print("FAIL: Not using hybrid approach")
            return False
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    """Run all tests"""
    print("PHASE 3.4 HYBRID FIX - BASIC TESTS")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Incoherent Detection", test_incoherent_detection),
        ("Statistics Interface", test_statistics)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        result = test_func()
        status = "PASS" if result else "FAIL"
        print(f"\n{test_name}: {status}")
        if result:
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("ALL TESTS PASSED! Hybrid fix is working.")
        return True
    else:
        print("SOME TESTS FAILED! Please review implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)