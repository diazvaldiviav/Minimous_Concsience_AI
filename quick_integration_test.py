#!/usr/bin/env python3
"""
Quick Integration Test - Phase 3.4 Critical Issues
=================================================
Tests just the critical fixes without Unicode issues.
"""

import sys
import os

# Add project root to path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def test_autonomous_thinking_import():
    """Test that autonomous_thinking.py can be imported without syntax errors"""
    print("TEST: Import autonomous_thinking.py")
    try:
        from conscious_ai.autonomus_thinking.autonomous_thinking import AutonomousThoughtGenerator
        print("   SUCCESS: autonomous_thinking.py imports correctly")
        
        # Quick instantiation test
        generator = AutonomousThoughtGenerator()
        print("   SUCCESS: AutonomousThoughtGenerator instantiated")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_critical_state_evaluator():
    """Test Phase 3.4 evaluator"""
    print("\nTEST: Critical State Evaluator")
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator, EvaluationStrategy
        )
        
        evaluator = CriticalStateEvaluator(
            max_attempts=3,
            temperature_decay=0.3,
            strategy=EvaluationStrategy.ML_FIRST,
            ml_classifier_path="./models/coherence_classifier",
            coherence_threshold=0.7
        )
        print("   SUCCESS: CriticalStateEvaluator initialized")
        print(f"   Strategy: {evaluator.strategy.value}")
        print(f"   ML Available: {evaluator.ml_available}")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_heuristic_evaluator():
    """Test heuristic evaluator"""
    print("\nTEST: Heuristic Evaluator")
    try:
        from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator
        
        evaluator = CoherenceEvaluator()
        print("   SUCCESS: CoherenceEvaluator initialized")
        
        # Test basic evaluation
        prev_state = {
            "goal": "understand_self",
            "emotion": "curious", 
            "confidence": 0.6,
            "thought": "I wonder about my nature",
            "memory": ["self-reflection"]
        }
        
        next_state = {
            "goal": "analyze_consciousness",
            "emotion": "analytical",
            "confidence": 0.7, 
            "thought": "I examine my thought processes",
            "memory": ["self-reflection", "analysis"]
        }
        
        result = evaluator.evaluate_coherence(prev_state, next_state)
        print(f"   Evaluation result: {result}")
        
        if result in ['coherente', 'incoherente', 'ambiguo']:
            print("   SUCCESS: Heuristic evaluator returned valid result")
            return True
        else:
            print(f"   FAILED: Invalid result '{result}'")
            return False
            
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run critical integration tests"""
    print("QUICK INTEGRATION TEST - PHASE 3.4 CRITICAL FIXES")
    print("=" * 60)
    
    results = []
    
    # Test 1: autonomous_thinking import
    results.append(test_autonomous_thinking_import())
    
    # Test 2: critical state evaluator  
    results.append(test_critical_state_evaluator())
    
    # Test 3: heuristic evaluator
    results.append(test_heuristic_evaluator())
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All critical fixes working!")
    else:
        print("ISSUES: Some tests still failing")
    
    return passed == total

if __name__ == "__main__":
    main()