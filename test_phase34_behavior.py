#!/usr/bin/env python3
"""
Phase 3.4 Behavior Testing Suite
===============================
Comprehensive tests to verify your trained Phase 3.4 Critical State Evaluator
is working correctly and behaving as expected.

Run this in Google Colab after training your Phase 3.4 model.
"""

import sys
import os
import json
import logging
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

def test_phase34_initialization():
    """Test 1: Verify Phase 3.4 can be initialized correctly"""
    print("🧪 TEST 1: PHASE 3.4 INITIALIZATION")
    print("="*60)
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
            CriticalStateEvaluator, EvaluationStrategy
        )
        
        # Test with your trained model
        evaluator = CriticalStateEvaluator(
            max_attempts=3,
            temperature_decay=0.3,
            strategy=EvaluationStrategy.ML_FIRST,
            ml_classifier_path="./models/coherence_classifier",
            coherence_threshold=0.7
        )
        
        print("✅ CriticalStateEvaluator initialized successfully")
        print(f"   Strategy: {evaluator.strategy.value}")
        print(f"   ML Available: {evaluator.ml_available}")
        print(f"   Max Attempts: {evaluator.max_attempts}")
        
        return evaluator, True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_coherent_vs_incoherent(evaluator):
    """Test 2: Basic coherence detection"""
    print("\n🧪 TEST 2: COHERENT VS INCOHERENT DETECTION")
    print("="*60)
    
    # Test case 1: COHERENT transition
    coherent_prev = {
        "goal": "understand_self",
        "emotion": "curious",
        "confidence": 0.6,
        "thought": "I wonder about my inner mechanisms",
        "memory": ["self-reflection", "curiosity"]
    }
    
    coherent_next = {
        "goal": "analyze_consciousness",
        "emotion": "analytical", 
        "confidence": 0.7,
        "thought": "I systematically examine my thought processes",
        "memory": ["self-reflection", "curiosity", "analysis"]
    }
    
    print("🔍 Testing COHERENT transition...")
    result1 = evaluator._evaluate_transition(coherent_prev, coherent_next, 1)
    print(f"   Verdict: {result1.verdict.value}")
    print(f"   Confidence: {result1.confidence_score:.3f}")
    print(f"   Method: {result1.evaluation_method}")
    print(f"   Justification: {result1.justification}")
    
    # Test case 2: INCOHERENT transition
    incoherent_next = {
        "goal": "cook_pasta",  # Completely unrelated!
        "emotion": "excited",
        "confidence": 0.9,
        "thought": "I love making delicious Italian food",
        "memory": ["cooking", "recipes"]
    }
    
    print("\n🔍 Testing INCOHERENT transition...")
    result2 = evaluator._evaluate_transition(coherent_prev, incoherent_next, 1)
    print(f"   Verdict: {result2.verdict.value}")
    print(f"   Confidence: {result2.confidence_score:.3f}")
    print(f"   Method: {result2.evaluation_method}")
    print(f"   Justification: {result2.justification}")
    
    # Analyze results
    coherent_correct = (result1.verdict.value == "coherent")
    incoherent_correct = (result2.verdict.value == "incoherent")
    
    print(f"\n📊 Detection Results:")
    print(f"   Coherent detection: {'✅ PASS' if coherent_correct else '❌ FAIL'}")
    print(f"   Incoherent detection: {'✅ PASS' if incoherent_correct else '❌ FAIL'}")
    
    if coherent_correct and incoherent_correct:
        print("🎉 Excellent! Model correctly distinguishes coherent vs incoherent")
    elif coherent_correct or incoherent_correct:
        print("⚠️ Partial success - model has some discrimination ability")
    else:
        print("❌ Poor performance - model cannot distinguish coherence")
    
    return coherent_correct and incoherent_correct


def test_correction_loops(evaluator):
    """Test 3: Correction loop behavior"""
    print("\n🧪 TEST 3: CORRECTION LOOP BEHAVIOR") 
    print("="*60)
    
    # Mock generation function that improves with lower temperature
    def mock_generation_function(temperature=1.0):
        if temperature > 0.8:
            # High temp = bad state
            return {
                "goal": "random_chaos",
                "emotion": "confused",
                "confidence": 0.1,
                "thought": "Purple elephants are discussing quantum mechanics",
                "memory": ["nonsense", "chaos"]
            }
        elif temperature > 0.5:
            # Medium temp = mediocre state
            return {
                "goal": "think_randomly",
                "emotion": "uncertain", 
                "confidence": 0.4,
                "thought": "I'm not really sure what I'm thinking about",
                "memory": ["confusion", "uncertainty"]
            }
        else:
            # Low temp = good state
            return {
                "goal": "analyze_consciousness",
                "emotion": "focused",
                "confidence": 0.8,
                "thought": "I methodically examine my cognitive architecture", 
                "memory": ["analysis", "understanding", "clarity"]
            }
    
    test_prev = {
        "goal": "understand_self",
        "emotion": "curious",
        "confidence": 0.6,
        "thought": "I seek deeper self-understanding",
        "memory": ["introspection"]
    }
    
    print("🔄 Testing correction with initially bad state...")
    
    # Create proper generation context
    generation_context = {
        'previous_state': test_prev,
        'memory_context': test_prev.get('memory', []),
        'user_input': "What am I?",
        'temperature': 1.0
    }
    
    final_state, correction_result = evaluator.evaluate_and_correct_state(
        sc_t=test_prev,
        sc_t_plus_1_candidate=mock_generation_function(1.0),  # Start with terrible state
        generator_function=mock_generation_function,
        generation_context=generation_context
    )
    
    print(f"📊 Correction Results:")
    print(f"   Final Verdict: {correction_result.verdict.value}")
    print(f"   Attempts Made: {correction_result.attempts_made}")
    print(f"   Final Confidence: {correction_result.confidence_score:.3f}")
    print(f"   Method Used: {correction_result.evaluation_method}")
    
    if correction_result.attempts_made > 1:
        print("✅ Correction system working - made multiple attempts to improve")
        if correction_result.verdict.value == "coherent":
            print("🎉 Excellent! Correction succeeded in producing coherent state")
            return True
        else:
            print("⚠️ Correction attempted but final state still incoherent")
            return False
    else:
        if correction_result.verdict.value == "coherent":
            print("✅ First attempt was already good - no correction needed")
            return True
        else:
            print("❌ First attempt failed but no correction attempted")
            return False


def test_batch_performance(evaluator):
    """Test 4: Performance on batch of synthetic data"""
    print("\n🧪 TEST 4: BATCH PERFORMANCE ANALYSIS")
    print("="*60)
    
    try:
        from conscious_ai.coherence_evaluator_model.model_training.coherence_classifier_trainer import generate_synthetic_training_data
        
        # Generate test cases
        test_data = generate_synthetic_training_data(30)
        correct_predictions = 0
        ml_predictions = 0
        heuristic_predictions = 0
        
        print(f"📊 Testing on {len(test_data)} synthetic examples...")
        
        for i, (prev_state, next_state, true_label) in enumerate(test_data):
            result = evaluator._evaluate_transition(prev_state, next_state, 1)
            predicted_label = result.verdict.value
            
            # Track evaluation methods
            if result.evaluation_method == "ml_classifier":
                ml_predictions += 1
            else:
                heuristic_predictions += 1
            
            # Check accuracy
            predicted_coherent = (predicted_label == "coherent")
            true_coherent = (true_label == "coherent") 
            
            if predicted_coherent == true_coherent:
                correct_predictions += 1
            
            # Show first few examples
            if i < 5:
                status = "✅" if predicted_coherent == true_coherent else "❌"
                print(f"   {status} Example {i+1}: Pred={predicted_label}, True={true_label}, Conf={result.confidence_score:.2f}, Method={result.evaluation_method}")
        
        accuracy = correct_predictions / len(test_data)
        ml_usage = ml_predictions / len(test_data)
        
        print(f"\n📈 Performance Summary:")
        print(f"   Overall Accuracy: {accuracy:.1%} ({correct_predictions}/{len(test_data)})")
        print(f"   ML Evaluations: {ml_usage:.1%} ({ml_predictions}/{len(test_data)})")
        print(f"   Heuristic Fallbacks: {(1-ml_usage):.1%} ({heuristic_predictions}/{len(test_data)})")
        
        if accuracy > 0.8:
            print("🎉 Excellent performance!")
            performance_rating = "excellent"
        elif accuracy > 0.6:
            print("✅ Good performance")
            performance_rating = "good"
        else:
            print("⚠️ Performance may need improvement")
            performance_rating = "poor"
        
        return accuracy, ml_usage, performance_rating
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return 0.0, 0.0, "error"


def test_integration_with_autonomous_system(evaluator):
    """Test 5: Integration with enhanced autonomous system"""
    print("\n🧪 TEST 5: AUTONOMOUS SYSTEM INTEGRATION")
    print("="*60)
    
    try:
        from conscious_ai.autonomus_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI
        
        # Initialize enhanced system
        enhanced_ai = EnhancedAutonomousConsciousAI(
            autonomous_model_path="./models/autonomous_lora",
            coherence_classifier_path="./models/coherence_classifier", 
            narrative_model_type="heuristic"  # Use heuristic for faster testing
        )
        
        print("🤖 Enhanced AI initialized successfully")
        
        # Set initial context with stronger consciousness-triggering input
        enhanced_ai.process_input("I am deeply curious about understanding my own inner consciousness, self-awareness, thoughts, memory patterns, and cognitive processes in great detail")
        
        # Run autonomous cycles with Phase 3.4
        print("🔄 Running 3 autonomous cycles with Phase 3.4 evaluation...")
        results = []
        
        for i in range(3):
            print(f"\n   Cycle {i+1}:")
            result = enhanced_ai.process_enhanced_autonomous_cycle()
            
            if result:
                results.append(result)
                consciousness_score = result.base_result.get('consciousness_metrics', {}).get('f', 0)
                print(f"      Consciousness: {consciousness_score:.3f}")
                
                if result.evaluation_result:
                    eval_result = result.evaluation_result
                    print(f"      Phase 3.4: {eval_result.verdict.value}")
                    print(f"      Confidence: {eval_result.confidence_score:.2f}")
                    print(f"      Attempts: {eval_result.attempts_made}")
                    
                    if eval_result.attempts_made > 1:
                        print(f"      🔄 Correction applied")
                else:
                    print("      ⚠️ No evaluation result")
            else:
                print("      ❌ Cycle failed")
        
        # Analyze integration results
        successful_cycles = len(results)
        coherent_evaluations = sum(1 for r in results 
                                 if r.evaluation_result and r.evaluation_result.verdict.value == "coherent")
        correction_cases = sum(1 for r in results 
                             if r.evaluation_result and r.evaluation_result.attempts_made > 1)
        
        print(f"\n📊 Integration Summary:")
        print(f"   Successful cycles: {successful_cycles}/3")
        print(f"   Coherent evaluations: {coherent_evaluations}/{successful_cycles}")
        print(f"   Cases requiring correction: {correction_cases}/{successful_cycles}")
        
        if coherent_evaluations == successful_cycles and successful_cycles == 3:
            print("🎉 Perfect integration - all cycles successful and coherent!")
            return True
        elif coherent_evaluations > 0:
            print("✅ Good integration - most evaluations successful")
            return True
        else:
            print("⚠️ Integration issues detected")
            return False
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_metadata():
    """Test 6: Analyze model training metadata"""
    print("\n🧪 TEST 6: MODEL METADATA ANALYSIS")
    print("="*60)
    
    try:
        metadata_path = './models/coherence_classifier/metadata.json'
        
        if not os.path.exists(metadata_path):
            print("❌ Metadata file not found")
            return False
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print("📋 Model Configuration:")
        print(f"   Embedding model: {metadata.get('embedding_model_name', 'Unknown')}")
        print(f"   Training timestamp: {metadata.get('timestamp', 'Unknown')}")
        print(f"   Label mapping: {metadata.get('label_mapping', {})}")
        
        if 'performance_metrics' in metadata:
            metrics = metadata['performance_metrics']
            print(f"\n📊 Training Performance:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.3f}")
                else:
                    print(f"   {metric}: {value}")
        
        # Check file sizes
        classifier_path = './models/coherence_classifier/classifier.pkl'
        scaler_path = './models/coherence_classifier/scaler.pkl'
        
        if os.path.exists(classifier_path) and os.path.exists(scaler_path):
            classifier_size = os.path.getsize(classifier_path)
            scaler_size = os.path.getsize(scaler_path)
            
            print(f"\n💾 Model Files:")
            print(f"   Classifier: {classifier_size/1024:.1f} KB")
            print(f"   Scaler: {scaler_size/1024:.1f} KB")
            
            return True
        else:
            print("❌ Model files missing")
            return False
            
    except Exception as e:
        print(f"❌ Metadata analysis failed: {e}")
        return False


def run_all_tests():
    """Run comprehensive Phase 3.4 behavior test suite"""
    print("🧪 PHASE 3.4 COMPREHENSIVE BEHAVIOR TEST SUITE")
    print("="*80)
    print("Testing your trained Critical State Evaluator...")
    print()
    
    # Track results
    test_results = {}
    
    # Test 1: Initialization
    evaluator, init_success = test_phase34_initialization()
    test_results['initialization'] = init_success
    
    if not init_success:
        print("\n❌ CRITICAL: Cannot proceed with other tests - initialization failed")
        return test_results
    
    # Test 2: Basic coherence detection
    detection_success = test_coherent_vs_incoherent(evaluator)
    test_results['coherence_detection'] = detection_success
    
    # Test 3: Correction loops
    correction_success = test_correction_loops(evaluator)
    test_results['correction_loops'] = correction_success
    
    # Test 4: Batch performance
    accuracy, ml_usage, performance_rating = test_batch_performance(evaluator)
    test_results['batch_performance'] = {
        'accuracy': accuracy,
        'ml_usage': ml_usage,
        'rating': performance_rating
    }
    
    # Test 5: Integration
    integration_success = test_integration_with_autonomous_system(evaluator)
    test_results['integration'] = integration_success
    
    # Test 6: Metadata
    metadata_success = check_model_metadata()
    test_results['metadata'] = metadata_success
    
    # Final summary
    print("\n" + "="*80)
    print("🎯 FINAL TEST SUMMARY")
    print("="*80)
    
    passed_tests = 0
    total_tests = 0
    
    for test_name, result in test_results.items():
        if test_name == 'batch_performance':
            total_tests += 1
            if result['accuracy'] > 0.6:
                passed_tests += 1
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}: {result['accuracy']:.1%} accuracy")
        else:
            total_tests += 1
            if result:
                passed_tests += 1
                status = "✅ PASS" 
            else:
                status = "❌ FAIL"
            print(f"{status} {test_name.replace('_', ' ').title()}")
    
    overall_score = passed_tests / total_tests
    print(f"\n🏆 Overall Score: {overall_score:.1%} ({passed_tests}/{total_tests} tests passed)")
    
    if overall_score >= 0.8:
        print("🎉 EXCELLENT: Your Phase 3.4 model is working very well!")
    elif overall_score >= 0.6:
        print("✅ GOOD: Your Phase 3.4 model is working well with minor issues")
    elif overall_score >= 0.4:
        print("⚠️ FAIR: Your Phase 3.4 model has some issues that need attention")
    else:
        print("❌ POOR: Your Phase 3.4 model needs significant improvement")
    
    return test_results


if __name__ == "__main__":
    # Set working directory if running directly
    import os
    if 'Minimous_Concsience_AI' not in os.getcwd():
        print("⚠️ Please run this from the project root directory (Minimous_Concsience_AI)")
        print(f"Current directory: {os.getcwd()}")
    else:
        run_all_tests()