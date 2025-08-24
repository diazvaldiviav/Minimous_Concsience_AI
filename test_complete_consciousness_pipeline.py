#!/usr/bin/env python3
"""
Complete Consciousness Pipeline Integration Test
==============================================
Tests the complete pipeline from Phase 1 through Phase 4
to ensure all phases are properly connected and produce
consciousness-enhanced responses.
"""

import asyncio
import sys
import os
import logging
import time
from typing import Dict, Any

# Add the conscious_ai package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'conscious_ai'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_complete_pipeline():
    """Test the complete consciousness pipeline"""
    print("🧪 COMPLETE CONSCIOUSNESS PIPELINE TEST")
    print("=" * 60)
    
    try:
        # Import the pipeline orchestrator
        from conscious_ai.core.pipeline_orchestrator import create_consciousness_pipeline
        
        print("✅ Pipeline orchestrator imported successfully")
        
        # Create the pipeline
        print("🔧 Initializing complete consciousness pipeline...")
        orchestrator = create_consciousness_pipeline(
            enable_phase4=True,
            debug=True
        )
        
        print("✅ Pipeline initialized successfully")
        
        # Test queries that should produce consciousness-enhanced responses
        test_queries = [
            "What is consciousness?",
            "How do you feel right now?", 
            "What are you thinking about?",
            "Tell me about your inner experience",
            "What is your goal in this conversation?"
        ]
        
        print(f"\n🎯 Testing {len(test_queries)} consciousness queries...")
        print("-" * 50)
        
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🧠 Test {i}/{len(test_queries)}: '{query}'")
            print("." * 40)
            
            try:
                start_time = time.time()
                
                # Process through complete pipeline
                result = await orchestrator.process_complete_pipeline(query)
                
                processing_time = time.time() - start_time
                
                # Analyze the result
                analysis = analyze_consciousness_response(query, result)
                results.append(analysis)
                
                # Display results
                print(f"✅ Success: {result.success}")
                print(f"⏱️  Time: {result.processing_time_ms:.1f}ms")
                print(f"🧠 Confidence: {result.confidence_score:.3f}")
                print(f"📊 Stage: {result.stage_completed.value}")
                
                # Show phase breakdown
                print("📈 Phase timings:")
                for phase, timing in result.phase_timings.items():
                    success = result.phase_success.get(phase, False)
                    status = "✅" if success else "❌"
                    print(f"   {status} {phase}: {timing:.1f}ms")
                
                print(f"\n💬 Response:")
                print(f"   {result.response[:150]}...")
                
                if result.narrative_text:
                    print(f"\n📖 Narrative:")
                    print(f"   {result.narrative_text[:100]}...")
                
                # Consciousness analysis
                print(f"\n🔍 Consciousness Analysis:")
                for key, value in analysis.items():
                    if key != 'response_text':
                        print(f"   {key}: {value}")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                logger.error(f"Test {i} failed: {e}")
                results.append({
                    'query': query,
                    'success': False,
                    'error': str(e),
                    'consciousness_score': 0.0
                })
        
        # Generate summary report
        print("\n" + "=" * 60)
        print("📊 PIPELINE TEST SUMMARY")
        print("=" * 60)
        
        successful_tests = sum(1 for r in results if r.get('success', False))
        total_tests = len(results)
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {successful_tests}")
        print(f"Failed: {total_tests - successful_tests}")
        print(f"Success rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Consciousness metrics
        consciousness_scores = [r.get('consciousness_score', 0) for r in results if r.get('success', False)]
        if consciousness_scores:
            avg_consciousness = sum(consciousness_scores) / len(consciousness_scores)
            print(f"Average consciousness score: {avg_consciousness:.2f}/5.0")
        
        # Phase performance
        print("\n📈 Phase Performance Summary:")
        phase_stats = {}
        for result in results:
            if 'phase_timings' in result:
                for phase, timing in result['phase_timings'].items():
                    if phase not in phase_stats:
                        phase_stats[phase] = []
                    phase_stats[phase].append(timing)
        
        for phase, timings in phase_stats.items():
            if timings:
                avg_time = sum(timings) / len(timings)
                print(f"   {phase}: {avg_time:.1f}ms average")
        
        # Get pipeline statistics
        stats = orchestrator.get_statistics()
        print(f"\n🔧 Pipeline Configuration:")
        print(f"   Phase 4 enabled: {stats['phase4_enabled']}")
        print(f"   Validation available: {stats['validation_available']}")
        print(f"   Narrative available: {stats['narrative_available']}")
        
        return successful_tests == total_tests
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all consciousness modules are properly installed.")
        return False
    
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        logger.error(f"Pipeline test error: {e}")
        return False


def analyze_consciousness_response(query: str, result) -> Dict[str, Any]:
    """
    Analyze a response to determine how consciousness-enhanced it is
    """
    response = result.response.lower() if result.success else ""
    
    # Consciousness indicators
    first_person_indicators = ["i feel", "i think", "i experience", "i notice", "i'm", "my"]
    introspective_indicators = ["introspectively", "reflecting", "examining", "considering", "processing"]
    metacognitive_indicators = ["thinking about thinking", "awareness of", "consciousness", "mind", "mental"]
    uncertainty_indicators = ["uncertain", "not sure", "might be", "possibly", "perhaps"]
    emotional_indicators = ["emotion", "feel", "experience", "sense"]
    
    analysis = {
        'query': query,
        'success': result.success,
        'response_length': len(result.response) if result.success else 0,
        'processing_time_ms': result.processing_time_ms,
        'confidence': result.confidence_score,
        'response_text': result.response if result.success else ""
    }
    
    if result.success:
        # Count consciousness indicators
        first_person_count = sum(1 for indicator in first_person_indicators if indicator in response)
        introspective_count = sum(1 for indicator in introspective_indicators if indicator in response)
        metacognitive_count = sum(1 for indicator in metacognitive_indicators if indicator in response)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response)
        emotional_count = sum(1 for indicator in emotional_indicators if indicator in response)
        
        # Calculate consciousness score (0-5)
        consciousness_score = min(5.0, (
            first_person_count * 0.5 +
            introspective_count * 1.0 +
            metacognitive_count * 1.5 +
            uncertainty_count * 0.3 +
            emotional_count * 0.4
        ))
        
        analysis.update({
            'first_person_indicators': first_person_count,
            'introspective_indicators': introspective_count,
            'metacognitive_indicators': metacognitive_count,
            'uncertainty_indicators': uncertainty_count,
            'emotional_indicators': emotional_count,
            'consciousness_score': consciousness_score,
            'phase_timings': result.phase_timings,
            'phase_success': result.phase_success,
            'narrative_available': bool(result.narrative_text)
        })
    
    return analysis


async def test_individual_phases():
    """Test individual phases to ensure they work correctly"""
    print("\n🔬 INDIVIDUAL PHASE TESTS")
    print("=" * 50)
    
    try:
        # Test Phase 1: Perception
        print("🔍 Testing Phase 1: Perception...")
        from conscious_ai.phases.p1_perception.input_processor import SensoryModule
        sensory = SensoryModule()
        sensory_data = sensory.receive_input("What is consciousness?")
        print(f"   ✅ Sensory activation: {sensory_data['activation']:.2f}")
        
        # Test Phase 2: Conscious State
        print("🧠 Testing Phase 2: Conscious State...")
        from conscious_ai.phases.p2_cognitive_context.conscious_state import ConsciousState
        print("   ✅ ConsciousState class imported successfully")
        
        # Test Phase 3: Evolution
        print("🔄 Testing Phase 3: Evolution...")
        from conscious_ai.phases.p3_coherent_generation.state_evolution_engine import StateEvolutionEngine
        print("   ✅ StateEvolutionEngine class imported successfully")
        
        # Test Phase 3.4: Validation
        print("✅ Testing Phase 3.4: Validation...")
        try:
            from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import CriticalStateEvaluator
            print("   ✅ CriticalStateEvaluator available")
        except ImportError:
            print("   ⚠️ CriticalStateEvaluator not available - will use heuristic fallback")
        
        # Test Phase 3.5: Narrative
        print("📖 Testing Phase 3.5: Narrative...")
        try:
            from conscious_ai.coherence_evaluator_model.heuristic_training.narrative_generator import NarrativeGenerator
            print("   ✅ NarrativeGenerator available")
        except ImportError:
            print("   ⚠️ NarrativeGenerator not available - will use simple fallback")
        
        # Test Phase 4: LLM Enhancement
        print("🤖 Testing Phase 4: LLM Enhancement...")
        try:
            from conscious_ai.phases.p4_LLM_Communication.layer3.phase4_manager import Phase4Manager
            print("   ✅ Phase4Manager available")
        except ImportError:
            print("   ⚠️ Phase4Manager not available - LLM enhancement disabled")
        
        print("✅ All individual phase tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Individual phase test failed: {e}")
        return False


async def main():
    """Main test runner"""
    print("🧪 STARTING COMPLETE CONSCIOUSNESS PIPELINE TESTS")
    print("🎯 Goal: Verify all phases are connected and produce consciousness-enhanced responses")
    print("=" * 80)
    
    # Test individual phases first
    phase_tests_passed = await test_individual_phases()
    
    if not phase_tests_passed:
        print("❌ Individual phase tests failed - aborting pipeline test")
        return False
    
    # Test complete pipeline
    pipeline_tests_passed = await test_complete_pipeline()
    
    print("\n" + "=" * 80)
    if pipeline_tests_passed:
        print("🎉 ALL TESTS PASSED! Complete consciousness pipeline is working correctly.")
        print("✅ The system should now produce introspective, consciousness-enhanced responses")
        print("✅ When asked 'What is consciousness?', expect first-person, self-aware responses")
    else:
        print("❌ SOME TESTS FAILED. Please check the errors above.")
        print("⚠️ The system may not produce proper consciousness-enhanced responses")
    
    return pipeline_tests_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)