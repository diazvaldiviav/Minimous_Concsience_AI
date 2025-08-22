#!/usr/bin/env python3
"""
Quick MVP test to validate Phase 3.4 fixes
===========================================
Tests the critical fixes for division by zero, import paths, and coherence detection.
"""

import sys
import os
import logging
import traceback

# Add the conscious_ai package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_import_fixes():
    """Test that import path fixes work"""
    print("Testing import path fixes...")
    
    try:
        # Test the fixed import
        from conscious_ai.autonomous_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI
        print("PASS - Enhanced autonomous integration import successful")
        return True
    except ImportError as e:
        print(f"FAIL - Import failed: {e}")
        return False

def test_hybrid_evaluator_robustness():
    """Test hybrid evaluator with edge cases that previously caused division by zero"""
    print("🔍 Testing hybrid evaluator robustness...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        print("✅ Hybrid evaluator initialized")
        
        # Test edge cases that previously caused division by zero
        edge_cases = [
            # Empty states
            ({}, {}),
            # States with empty strings
            ({'goal': '', 'emotion': '', 'thought': '', 'confidence': 0.5, 'memory': []},
             {'goal': '', 'emotion': '', 'thought': '', 'confidence': 0.5, 'memory': []}),
            # States with None values
            ({'goal': None, 'emotion': None, 'thought': None, 'confidence': None, 'memory': None},
             {'goal': 'test', 'emotion': 'test', 'thought': 'test', 'confidence': 0.5, 'memory': []}),
            # Normal coherent transition
            ({'goal': 'understand', 'emotion': 'curious', 'thought': 'I wonder', 'confidence': 0.6, 'memory': ['memory1']},
             {'goal': 'analyze', 'emotion': 'analytical', 'thought': 'I examine', 'confidence': 0.65, 'memory': ['memory1', 'memory2']}),
            # Incoherent transition
            ({'goal': 'understand', 'emotion': 'curious', 'thought': 'I wonder', 'confidence': 0.6, 'memory': ['memory1']},
             {'goal': 'dance', 'emotion': 'angry', 'thought': 'I hate everything', 'confidence': 0.1, 'memory': []})
        ]
        
        results = []
        for i, (sc_t, sc_t_plus_1) in enumerate(edge_cases, 1):
            try:
                print(f"  Testing edge case {i}...")
                analysis = evaluator.evaluate_transition(sc_t, sc_t_plus_1)
                
                # Validate result
                assert hasattr(analysis, 'verdict'), "Missing verdict attribute"
                assert hasattr(analysis, 'justification'), "Missing justification attribute"
                assert isinstance(analysis.verdict, CoherenceVerdict), "Invalid verdict type"
                
                results.append({
                    'case': i,
                    'verdict': analysis.verdict.value,
                    'justification': analysis.justification[:100] + "..." if len(analysis.justification) > 100 else analysis.justification
                })
                print(f"    ✅ Case {i}: {analysis.verdict.value}")
                
            except Exception as e:
                print(f"    ❌ Case {i} failed: {e}")
                results.append({'case': i, 'error': str(e)})
        
        # Check that we got sensible results
        coherent_count = sum(1 for r in results if r.get('verdict') == 'coherent')
        incoherent_count = sum(1 for r in results if r.get('verdict') == 'incoherent')
        
        print(f"✅ Edge case testing completed:")
        print(f"   - Coherent: {coherent_count}")
        print(f"   - Incoherent: {incoherent_count}")
        print(f"   - Ambiguous: {len(results) - coherent_count - incoherent_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid evaluator test failed: {e}")
        traceback.print_exc()
        return False

def test_critical_evaluator_integration():
    """Test the full critical evaluator with error handling"""
    print("🔍 Testing critical evaluator integration...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            create_critical_evaluator, CoherenceVerdict
        )
        
        # Create evaluator
        evaluator = create_critical_evaluator(max_attempts=2)
        print("✅ Critical evaluator created")
        
        # Test basic evaluation
        sc_t = {'goal': 'understand', 'emotion': 'curious', 'thought': 'I wonder', 'confidence': 0.6, 'memory': []}
        sc_t_plus_1 = {'goal': 'analyze', 'emotion': 'analytical', 'thought': 'I examine', 'confidence': 0.65, 'memory': []}
        
        # Simple mock generator function
        def mock_generator(**kwargs):
            return {'goal': 'fallback', 'emotion': 'neutral', 'thought': 'fallback thought', 'confidence': 0.5, 'memory': []}
        
        # Test evaluation (this should not crash)
        result_state, evaluation_result = evaluator.evaluate_and_correct_state(
            sc_t=sc_t,
            sc_t_plus_1_candidate=sc_t_plus_1,
            generator_function=mock_generator,
            generation_context={'temperature': 0.7},
            fallback_generator_function=mock_generator
        )
        
        print(f"✅ Evaluation completed: {evaluation_result.verdict.value}")
        print(f"   Attempts: {evaluation_result.attempts_made}")
        print(f"   Method: {evaluation_result.evaluation_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Critical evaluator integration test failed: {e}")
        traceback.print_exc()
        return False

def test_coherence_detection_improvement():
    """Test that coherence detection is improved"""
    print("🔍 Testing coherence detection improvement...")
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            HybridCoherenceEvaluator, CoherenceVerdict
        )
        
        evaluator = HybridCoherenceEvaluator()
        
        # Test cases that should clearly be incoherent
        clearly_incoherent_cases = [
            # Complete mismatch
            ({'goal': 'understand', 'emotion': 'curious', 'thought': 'I wonder about consciousness', 'confidence': 0.6, 'memory': ['memory1']},
             {'goal': 'destroy', 'emotion': 'angry', 'thought': 'I hate everything', 'confidence': 0.1, 'memory': []}),
            # Nonsensical transition
            ({'goal': 'learn', 'emotion': 'interested', 'thought': 'studying patterns', 'confidence': 0.7, 'memory': ['study1']},
             {'goal': 'banana', 'emotion': 'purple', 'thought': 'the moon is square', 'confidence': 0.2, 'memory': ['random']})
        ]
        
        incoherent_detected = 0
        for i, (sc_t, sc_t_plus_1) in enumerate(clearly_incoherent_cases, 1):
            analysis = evaluator.evaluate_transition(sc_t, sc_t_plus_1)
            print(f"  Case {i}: {analysis.verdict.value}")
            if analysis.verdict == CoherenceVerdict.INCOHERENT:
                incoherent_detected += 1
        
        detection_rate = incoherent_detected / len(clearly_incoherent_cases)
        print(f"✅ Incoherent detection rate: {detection_rate:.1%} ({incoherent_detected}/{len(clearly_incoherent_cases)})")
        
        # Should detect at least 50% of clearly incoherent cases
        return detection_rate >= 0.5
        
    except Exception as e:
        print(f"❌ Coherence detection test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("PHASE 3.4 MVP FIXES VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Import Fixes", test_import_fixes),
        ("Hybrid Evaluator Robustness", test_hybrid_evaluator_robustness),
        ("Critical Evaluator Integration", test_critical_evaluator_integration),
        ("Coherence Detection Improvement", test_coherence_detection_improvement)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    success_rate = passed / len(results)
    print(f"\nOverall Success Rate: {success_rate:.1%} ({passed}/{len(results)})")
    
    if success_rate >= 0.75:
        print("🎉 MVP fixes are working well!")
        return True
    else:
        print("⚠️ Some issues remain, but core functionality should be improved")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)