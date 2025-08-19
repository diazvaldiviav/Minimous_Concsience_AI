#!/usr/bin/env python3
"""
Hybrid Phase 3.4 Fix Integration Test
====================================
Tests the new hybrid critical state evaluator to verify it fixes the 
ML classifier overfitting issues and "maximum attempts reached" errors.

Expected outcomes:
- Coherent detection: ✅ PASS
- Incoherent detection: ✅ PASS
- No "maximum attempts reached" errors
- >85% accuracy on test cases
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_hybrid_evaluator_direct():
    """Test the hybrid evaluator directly"""
    print("🧪 Testing HybridCoherenceEvaluator directly...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        print("✅ HybridCoherenceEvaluator initialized successfully")
        
        # Test case 1: Coherent transition
        coherent_prev = {
            "goal": "understand consciousness",
            "emotion": "curious",
            "thought": "I wonder about my awareness",
            "confidence": 0.6,
            "memory": ["previous thoughts about consciousness"]
        }
        
        coherent_curr = {
            "goal": "analyze consciousness patterns",
            "emotion": "analytical", 
            "thought": "I examine the structure of my awareness",
            "confidence": 0.65,
            "memory": ["previous thoughts about consciousness", "new insights about patterns"]
        }
        
        result = evaluator.evaluate_transition(coherent_prev, coherent_curr)
        print(f"Coherent test: {result.verdict.value} (expected: coherent)")
        
        if result.verdict == CoherenceVerdict.COHERENT:
            print("Coherent detection: PASS")
            coherent_success = True
        else:
            print("Coherent detection: FAIL")
            coherent_success = False
        
        # Test case 2: Incoherent transition
        incoherent_prev = {
            "goal": "understand consciousness",
            "emotion": "curious",
            "thought": "I wonder about my awareness",
            "confidence": 0.6,
            "memory": ["thoughts about consciousness"]
        }
        
        incoherent_curr = {
            "goal": "bake a cake",
            "emotion": "frustrated",
            "thought": "The weather is purple today",
            "confidence": 0.2,
            "memory": ["random unrelated items"]
        }
        
        result = evaluator.evaluate_transition(incoherent_prev, incoherent_curr)
        print(f"Incoherent test: {result.verdict.value} (expected: incoherent)")
        
        if result.verdict == CoherenceVerdict.INCOHERENT:
            print("Incoherent detection: PASS")
            incoherent_success = True
        else:
            print("Incoherent detection: FAIL")
            incoherent_success = False
        
        return coherent_success and incoherent_success
        
    except Exception as e:
        print(f"❌ Direct hybrid evaluator test failed: {e}")
        return False


def test_critical_state_evaluator_integration():
    """Test the full CriticalStateEvaluator with hybrid approach"""
    print("\nTesting CriticalStateEvaluator integration...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator, create_critical_evaluator, CoherenceVerdict
        )
        
        # Test factory function
        evaluator = create_critical_evaluator(max_attempts=2)
        print("✅ CriticalStateEvaluator created successfully")
        
        # Mock generator function for testing
        def mock_generator(**kwargs):
            return {
                "goal": "refined understanding",
                "emotion": "focused", 
                "thought": "I refine my understanding",
                "confidence": 0.7,
                "memory": ["previous thoughts", "refined insights"]
            }
        
        # Test case: Coherent state should be accepted immediately
        coherent_prev = {
            "goal": "understand consciousness",
            "emotion": "curious",
            "thought": "I wonder about awareness",
            "confidence": 0.6,
            "memory": ["initial thoughts"]
        }
        
        coherent_candidate = {
            "goal": "explore consciousness deeper",
            "emotion": "analytical",
            "thought": "I analyze patterns of awareness",
            "confidence": 0.65,
            "memory": ["initial thoughts", "new insights"]
        }
        
        generation_context = {
            "previous_state": coherent_prev,
            "temperature": 0.7
        }
        
        final_state, eval_result = evaluator.evaluate_and_correct_state(
            coherent_prev,
            coherent_candidate,
            mock_generator,
            generation_context
        )
        
        print(f"Integration test result: {eval_result.verdict.value}")
        print(f"Attempts made: {eval_result.attempts_made}")
        print(f"Evaluation method: {eval_result.evaluation_method}")
        print(f"Confidence: {eval_result.confidence_score:.3f}")
        
        success = (
            eval_result.verdict == CoherenceVerdict.COHERENT and
            eval_result.attempts_made == 1 and  # Should accept on first attempt
            eval_result.evaluation_method == "hybrid_evaluator"
        )
        
        if success:
            print("✅ Integration test: PASS")
        else:
            print("❌ Integration test: FAIL")
        
        return success
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def test_api_compatibility():
    """Test that the API is compatible with existing interfaces"""
    print("\nTesting API compatibility...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator, EvaluationStrategy
        )
        
        # Test legacy parameter handling
        evaluator = CriticalStateEvaluator(
            max_attempts=3,
            temperature_decay=0.3,
            use_ml_classifier=True,  # Legacy parameter - should be ignored
            strategy=EvaluationStrategy.ML_FIRST  # Legacy strategy - should use hybrid
        )
        
        print("✅ Legacy parameters handled correctly")
        
        # Test statistics interface
        stats = evaluator.get_evaluation_statistics()
        expected_keys = [
            'total_evaluations', 'success_rate_first_attempt', 
            'regeneration_rate', 'failure_rate', 'evaluation_method'
        ]
        
        for key in expected_keys:
            if key not in stats:
                print(f"❌ Missing expected statistics key: {key}")
                return False
        
        print("✅ Statistics interface compatible")
        
        # Test that ML classifier is properly disabled
        if stats['ml_available'] == False and stats['hybrid_available'] == True:
            print("✅ ML classifier properly replaced with hybrid approach")
        else:
            print("❌ ML classifier not properly replaced")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False


def test_batch_accuracy():
    """Test accuracy on a batch of transitions"""
    print("\nTesting batch accuracy...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        
        # Test cases with expected verdicts
        test_cases = [
            # Coherent cases
            {
                "prev": {"goal": "understand", "emotion": "curious", "thought": "exploring", "confidence": 0.6},
                "curr": {"goal": "analyze", "emotion": "analytical", "thought": "examining patterns", "confidence": 0.65},
                "expected": CoherenceVerdict.COHERENT
            },
            {
                "prev": {"goal": "learn", "emotion": "interested", "thought": "discovering", "confidence": 0.5},
                "curr": {"goal": "study deeper", "emotion": "focused", "thought": "investigating details", "confidence": 0.6},
                "expected": CoherenceVerdict.COHERENT
            },
            {
                "prev": {"goal": "reflect", "emotion": "contemplative", "thought": "pondering", "confidence": 0.7},
                "curr": {"goal": "understand better", "emotion": "insightful", "thought": "gaining clarity", "confidence": 0.75},
                "expected": CoherenceVerdict.COHERENT
            },
            
            # Incoherent cases
            {
                "prev": {"goal": "understand consciousness", "emotion": "curious", "thought": "exploring awareness", "confidence": 0.6},
                "curr": {"goal": "fix the printer", "emotion": "frustrated", "thought": "the sky is falling", "confidence": 0.1},
                "expected": CoherenceVerdict.INCOHERENT
            },
            {
                "prev": {"goal": "analyze patterns", "emotion": "focused", "thought": "examining structure", "confidence": 0.7},
                "curr": {"goal": "dance with elephants", "emotion": "confused", "thought": "purple mathematics", "confidence": 0.2},
                "expected": CoherenceVerdict.INCOHERENT
            },
            
            # Ambiguous case
            {
                "prev": {"goal": "understand", "emotion": "curious", "thought": "wondering", "confidence": 0.5},
                "curr": {"goal": "think about something", "emotion": "neutral", "thought": "considering options", "confidence": 0.5},
                "expected": CoherenceVerdict.AMBIGUOUS
            }
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for i, case in enumerate(test_cases):
            result = evaluator.evaluate_transition(case["prev"], case["curr"])
            is_correct = result.verdict == case["expected"]
            
            status = "PASS" if is_correct else "FAIL"
            print(f"Test {i+1}: {result.verdict.value} (expected: {case['expected'].value}) {status}")
            
            if is_correct:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        print(f"\nBatch accuracy: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        if accuracy >= 0.85:
            print("✅ Batch accuracy test: PASS (≥85%)")
            return True
        else:
            print("❌ Batch accuracy test: FAIL (<85%)")
            return False
        
    except Exception as e:
        print(f"❌ Batch accuracy test failed: {e}")
        return False


def test_no_max_attempts_errors():
    """Test that the system no longer generates 'maximum attempts reached' errors"""
    print("\nTesting elimination of 'maximum attempts reached' errors...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator, CoherenceVerdict
        )
        
        evaluator = CriticalStateEvaluator(max_attempts=2)  # Low attempts to test
        
        # Mock generator that always generates coherent states
        def coherent_generator(**kwargs):
            return {
                "goal": "understand better",
                "emotion": "focused",
                "thought": "gaining deeper insight",
                "confidence": 0.7,
                "memory": ["building understanding"]
            }
        
        # Test with a slightly incoherent initial state that should trigger regeneration
        prev_state = {
            "goal": "understand consciousness",
            "emotion": "curious", 
            "thought": "exploring awareness",
            "confidence": 0.6,
            "memory": ["initial exploration"]
        }
        
        borderline_candidate = {
            "goal": "think about stuff",  # Somewhat vague
            "emotion": "uncertain",
            "thought": "not sure what to think",
            "confidence": 0.4,
            "memory": ["initial exploration"]
        }
        
        generation_context = {
            "previous_state": prev_state,
            "temperature": 0.7
        }
        
        final_state, eval_result = evaluator.evaluate_and_correct_state(
            prev_state,
            borderline_candidate,
            coherent_generator,
            generation_context
        )
        
        # With hybrid evaluator, we should get a definitive result, not maximum attempts error
        if eval_result.verdict in [CoherenceVerdict.COHERENT, CoherenceVerdict.INCOHERENT, CoherenceVerdict.AMBIGUOUS]:
            print("✅ No 'maximum attempts reached' errors: PASS")
            print(f"   Final verdict: {eval_result.verdict.value}")
            print(f"   Attempts made: {eval_result.attempts_made}")
            return True
        else:
            print("❌ Unexpected evaluation result")
            return False
        
    except Exception as e:
        if "maximum attempts" in str(e).lower():
            print("❌ Still getting 'maximum attempts reached' errors")
            return False
        else:
            print(f"❌ Test failed with different error: {e}")
            return False


def main():
    """Run all integration tests"""
    print("HYBRID PHASE 3.4 FIX - INTEGRATION TESTS")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Direct Hybrid Evaluator", test_hybrid_evaluator_direct()))
    test_results.append(("CriticalStateEvaluator Integration", test_critical_state_evaluator_integration()))
    test_results.append(("API Compatibility", test_api_compatibility()))
    test_results.append(("Batch Accuracy", test_batch_accuracy()))
    test_results.append(("No Max Attempts Errors", test_no_max_attempts_errors()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY:")
    print("=" * 60)
    
    passed_tests = 0
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:.<40} {status}")
        if result:
            passed_tests += 1
    
    overall_success = passed_tests == len(test_results)
    
    print(f"\nOverall Result: {passed_tests}/{len(test_results)} tests passed")
    
    if overall_success:
        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Phase 3.4 Hybrid Fix Implementation Successful!")
        print("\nExpected outcomes achieved:")
        print("✅ Coherent detection working")
        print("✅ Incoherent detection working") 
        print("✅ No 'maximum attempts reached' errors")
        print("✅ >85% accuracy on test cases")
        print("✅ API compatibility maintained")
        print("\n🚀 Ready for production use!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("🔧 Please review the implementation before deployment")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)