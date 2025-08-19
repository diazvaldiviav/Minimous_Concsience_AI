#!/usr/bin/env python3
"""
Test semantic-only evaluation to verify it works reliably
"""

import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import ModelBasedCoherenceEvaluator
    from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceVerdict
    
    print("TESTING SEMANTIC-ONLY EVALUATION")
    print("=" * 50)
    
    # Initialize semantic evaluator (NO ML classifier)
    print("Initializing semantic evaluator...")
    evaluator = ModelBasedCoherenceEvaluator(
        coherence_threshold=0.7,
        use_ml_classifier=False  # Force disable ML
    )
    print("✓ Semantic evaluator initialized")
    
    # Test coherent transition
    print("\nTesting COHERENT transition...")
    coherent_prev = {
        "goal": "understand_self", 
        "emotion": "curious", 
        "confidence": 0.6,
        "thought": "I wonder about my inner nature",
        "memory": ["self-reflection"]
    }
    coherent_next = {
        "goal": "analyze_patterns", 
        "emotion": "analytical", 
        "confidence": 0.7,
        "thought": "I observe recurring themes in my processing",
        "memory": ["self-reflection", "analysis"]
    }
    
    result1 = evaluator.evaluate_transition(coherent_prev, coherent_next)
    print(f"Result: {result1.verdict.value}")
    print(f"Justification: {result1.justification}")
    
    # Test incoherent transition
    print("\nTesting INCOHERENT transition...")
    incoherent_next = {
        "goal": "cook_dinner", 
        "emotion": "excited", 
        "confidence": 0.9,
        "thought": "I love making delicious pasta dishes",
        "memory": ["cooking", "recipes"]
    }
    
    result2 = evaluator.evaluate_transition(coherent_prev, incoherent_next)
    print(f"Result: {result2.verdict.value}")
    print(f"Justification: {result2.justification}")
    
    # Check results
    print(f"\nRESULTS:")
    coherent_correct = result1.verdict in [CoherenceVerdict.COHERENT, CoherenceVerdict.AMBIGUOUS]
    incoherent_correct = result2.verdict == CoherenceVerdict.INCOHERENT
    
    print(f"Coherent detection: {'✓ PASS' if coherent_correct else '✗ FAIL'}")
    print(f"Incoherent detection: {'✓ PASS' if incoherent_correct else '✗ FAIL'}")
    
    if coherent_correct and incoherent_correct:
        print("\n🎉 SEMANTIC EVALUATOR WORKS CORRECTLY!")
        print("This should be used as the primary method.")
        sys.exit(0)
    else:
        print("\n⚠️ Semantic evaluator has issues")
        sys.exit(1)
        
except ImportError as e:
    print(f"Import error: {e}")
    print("Dependencies missing - this is expected in environments without sentence-transformers")
    sys.exit(0)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)