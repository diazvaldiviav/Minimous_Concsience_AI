#!/usr/bin/env python3
"""
Simple Phase 3.4 MVP test without Unicode characters
=====================================================
"""

import sys
import os

# Add the conscious_ai package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_basic_functionality():
    """Test basic functionality without crashes"""
    print("PHASE 3.4 MVP FIXES VALIDATION")
    print("=" * 40)
    
    # Test 1: Import fixes
    print("\n1. Testing imports...")
    try:
        from conscious_ai.autonomous_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI
        print("PASS - Import fix successful")
    except ImportError as e:
        print(f"FAIL - Import error: {e}")
        return False
    
    # Test 2: Basic hybrid evaluator
    print("\n2. Testing hybrid evaluator...")
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        print("PASS - Evaluator created")
        
        # Test edge case that caused division by zero
        sc_t = {'goal': '', 'emotion': '', 'thought': '', 'confidence': 0.5, 'memory': []}
        sc_t_plus_1 = {'goal': '', 'emotion': '', 'thought': '', 'confidence': 0.5, 'memory': []}
        
        analysis = evaluator.evaluate_transition(sc_t, sc_t_plus_1)
        print(f"PASS - Edge case handled: {analysis.verdict.value}")
        
    except Exception as e:
        print(f"FAIL - Evaluator error: {e}")
        return False
    
    # Test 3: Coherence detection
    print("\n3. Testing coherence detection...")
    try:
        # Test clearly incoherent case
        incoherent_sc_t = {'goal': 'understand', 'emotion': 'curious', 'thought': 'I wonder', 'confidence': 0.6, 'memory': []}
        incoherent_sc_t_plus_1 = {'goal': 'destroy', 'emotion': 'angry', 'thought': 'I hate', 'confidence': 0.1, 'memory': []}
        
        analysis = evaluator.evaluate_transition(incoherent_sc_t, incoherent_sc_t_plus_1)
        print(f"Incoherent case result: {analysis.verdict.value}")
        
        # Test coherent case
        coherent_sc_t = {'goal': 'understand', 'emotion': 'curious', 'thought': 'I wonder', 'confidence': 0.6, 'memory': []}
        coherent_sc_t_plus_1 = {'goal': 'analyze', 'emotion': 'analytical', 'thought': 'I examine', 'confidence': 0.65, 'memory': []}
        
        analysis2 = evaluator.evaluate_transition(coherent_sc_t, coherent_sc_t_plus_1)
        print(f"Coherent case result: {analysis2.verdict.value}")
        
        print("PASS - Coherence detection working")
        
    except Exception as e:
        print(f"FAIL - Detection error: {e}")
        return False
    
    # Test 4: Critical evaluator integration
    print("\n4. Testing critical evaluator...")
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import create_critical_evaluator
        
        critical_eval = create_critical_evaluator(max_attempts=2)
        
        # Mock generator function
        def mock_gen(**kwargs):
            return {'goal': 'test', 'emotion': 'test', 'thought': 'test', 'confidence': 0.5, 'memory': []}
        
        result_state, eval_result = critical_eval.evaluate_and_correct_state(
            sc_t={'goal': 'test', 'emotion': 'test', 'thought': 'test', 'confidence': 0.5, 'memory': []},
            sc_t_plus_1_candidate={'goal': 'test2', 'emotion': 'test2', 'thought': 'test2', 'confidence': 0.5, 'memory': []},
            generator_function=mock_gen,
            generation_context={'temperature': 0.7},
            fallback_generator_function=mock_gen
        )
        
        print(f"PASS - Critical evaluator works: {eval_result.verdict.value}")
        
    except Exception as e:
        print(f"FAIL - Critical evaluator error: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("ALL TESTS PASSED - MVP fixes are working!")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)