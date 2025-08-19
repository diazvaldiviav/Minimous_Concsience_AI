#!/usr/bin/env python3
"""
Diagnostic script to test different evaluators and see their actual performance
"""

import sys
import os

def test_basic_transitions():
    """Test very basic coherent vs incoherent transitions"""
    
    print("TESTING BASIC COHERENT VS INCOHERENT TRANSITIONS")
    print("=" * 60)
    
    # Very obvious test cases
    test_cases = [
        # Clearly COHERENT transitions
        {
            "name": "Learning progression", 
            "prev": {"goal": "learn_python", "emotion": "curious", "thought": "I want to understand programming"},
            "next": {"goal": "practice_coding", "emotion": "focused", "thought": "Let me write some code to practice"},
            "expected": "coherent"
        },
        {
            "name": "Same state", 
            "prev": {"goal": "understand_self", "emotion": "reflective", "thought": "What am I?"},
            "next": {"goal": "understand_self", "emotion": "reflective", "thought": "What am I?"},
            "expected": "coherent"
        },
        
        # Clearly INCOHERENT transitions  
        {
            "name": "Random topic jump",
            "prev": {"goal": "understand_consciousness", "emotion": "curious", "thought": "What is awareness?"},
            "next": {"goal": "cook_pizza", "emotion": "hungry", "thought": "I want pepperoni pizza"},
            "expected": "incoherent"
        },
        {
            "name": "Emotional contradiction",
            "prev": {"goal": "help_friend", "emotion": "caring", "thought": "I want to support them"},
            "next": {"goal": "help_friend", "emotion": "hateful", "thought": "I want to hurt them"},
            "expected": "incoherent"
        }
    ]
    
    return test_cases

def test_heuristic_evaluator():
    """Test the heuristic evaluator"""
    print("\n1. TESTING HEURISTIC EVALUATOR")
    print("-" * 40)
    
    try:
        from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator
        
        evaluator = CoherenceEvaluator()
        test_cases = test_basic_transitions()
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            result = evaluator.evaluate_transition(case["prev"], case["next"])
            verdict = result.verdict.value
            expected = case["expected"]
            
            # Check if result matches expectation  
            is_correct = (
                (expected == "coherent" and verdict in ["coherente", "ambiguo"]) or
                (expected == "incoherent" and verdict == "incoherente")
            )
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status} {case['name']}: {verdict} (expected {expected})")
        
        accuracy = correct / total
        print(f"\nHeuristic Accuracy: {accuracy:.1%} ({correct}/{total})")
        return accuracy
        
    except Exception as e:
        print(f"Heuristic evaluator failed: {e}")
        return 0.0

def test_semantic_evaluator():
    """Test the semantic evaluator"""
    print("\n2. TESTING SEMANTIC EVALUATOR")
    print("-" * 40)
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import ModelBasedCoherenceEvaluator
        
        evaluator = ModelBasedCoherenceEvaluator(use_ml_classifier=False, coherence_threshold=0.6)
        test_cases = test_basic_transitions()
        
        correct = 0
        total = len(test_cases)
        
        for case in test_cases:
            result = evaluator.evaluate_transition(case["prev"], case["next"])
            verdict = result.verdict.value
            expected = case["expected"]
            
            # Check if result matches expectation
            is_correct = (
                (expected == "coherent" and verdict in ["coherente", "ambiguo"]) or
                (expected == "incoherent" and verdict == "incoherente")
            )
            
            if is_correct:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"{status} {case['name']}: {verdict} (expected {expected})")
        
        accuracy = correct / total
        print(f"\nSemantic Accuracy: {accuracy:.1%} ({correct}/{total})")
        return accuracy
        
    except Exception as e:
        print(f"Semantic evaluator failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def main():
    """Main diagnostic"""
    print("PHASE 3.4 EVALUATOR DIAGNOSTIC")
    print("=" * 50)
    print("Testing different evaluation methods on basic cases...")
    
    # Test each evaluator
    heuristic_acc = test_heuristic_evaluator()
    semantic_acc = test_semantic_evaluator()
    
    # Summary
    print(f"\n" + "=" * 50)
    print(f"DIAGNOSTIC SUMMARY")
    print(f"=" * 50)
    print(f"Heuristic evaluator: {heuristic_acc:.1%}")
    print(f"Semantic evaluator:  {semantic_acc:.1%}")
    
    # Recommendation
    if semantic_acc >= 0.75:
        print(f"\n✓ RECOMMENDATION: Use semantic evaluator (good performance)")
    elif heuristic_acc >= 0.75:
        print(f"\n✓ RECOMMENDATION: Use heuristic evaluator (semantic has issues)")
    else:
        print(f"\n⚠️ PROBLEM: Both evaluators have low accuracy")
        print(f"Need to debug the evaluation logic")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())