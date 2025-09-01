#!/usr/bin/env python3
"""
Test Phase 5.5: Narrative Recording of Consciousness
==================================================
Simple test script to verify Phase 5.5 implementation works correctly.
Tests all components and integration with the pipeline.
"""

import asyncio
import sys
import os

# Add the conscious_ai package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'conscious_ai'))

try:
    from conscious_ai.phases.p5_5_narrative import (
        ProcessLogger, DecisionTracker, MetacognitiveObserver, NarrativeSynthesizer
    )
    from conscious_ai.config.narrative_config import NarrativeVerbosity, EventType
    print("✅ Phase 5.5 imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def test_phase55_components():
    """Test individual Phase 5.5 components"""
    
    print("\n🧪 Testing Phase 5.5 Components")
    print("=" * 50)
    
    # Test ProcessLogger
    print("\n1. Testing ProcessLogger...")
    process_logger = ProcessLogger(max_events=10)
    
    # Log some test events
    process_logger.log_phase_transition("Phase 1", "Processing sensory input", 0.7, "curious")
    process_logger.log_decision_point("Phase 2", "Selected goal: understand", 0.8, ["assist", "analyze"], "Best fit for query type")
    process_logger.log_confidence_change("Phase 2", 0.6, 0.8, "Gained clarity on intent")
    process_logger.log_metacognitive_moment("Phase 3", "Observing my own processing patterns", 2, 0.7)
    
    events = process_logger.get_events()
    print(f"   Captured {len(events)} events")
    print(f"   Statistics: {process_logger.get_statistics()}")
    
    # Test DecisionTracker
    print("\n2. Testing DecisionTracker...")
    decision_tracker = DecisionTracker(max_decisions_per_cycle=5)
    
    from conscious_ai.phases.p5_5_narrative.decision_tracker import DecisionType
    
    # Track some decisions
    decision_tracker.track_goal_selection(
        "Phase 2", "understand_query", ["assist", "analyze", "respond"], 
        "Query analysis suggests understanding is primary need", 0.8
    )
    
    decision_tracker.track_response_strategy(
        "Phase 3", "analytical_approach", ["emotional", "creative", "factual"],
        "Analytical approach best suits the question type", 0.7, 0.8
    )
    
    decisions = decision_tracker.get_cycle_decisions()
    print(f"   Tracked {len(decisions)} decisions")
    print(f"   Quality analysis: {decision_tracker.analyze_decision_quality()}")
    
    # Test MetacognitiveObserver
    print("\n3. Testing MetacognitiveObserver...")
    metacog_observer = MetacognitiveObserver()
    
    # Record introspective events
    metacog_observer.observe_self_awareness_moment(
        "Phase 2", "I notice I'm forming goals based on input analysis", 0.8
    )
    
    metacog_observer.observe_recursive_thinking(
        "Phase 3", "I'm thinking about how I think about this problem", 3, 0.6
    )
    
    metacog_observer.observe_pattern_recognition(
        "Phase 3", "I see a recurring pattern in my cognitive processing", 0.7,
        "recursive_loops", "This creates feedback mechanisms"
    )
    
    # Test automatic pattern detection
    test_text = "I notice that I am examining my own thoughts and becoming aware of my thinking patterns."
    detected = metacog_observer.detect_metacognitive_patterns(test_text, "Phase 3")
    
    introspective_events = metacog_observer.get_introspective_events()
    print(f"   Recorded {len(introspective_events)} introspective events")
    print(f"   Auto-detected {len(detected)} patterns in text")
    print(f"   Recursion analysis: {metacog_observer.get_recursion_analysis()}")
    
    # Test NarrativeSynthesizer
    print("\n4. Testing NarrativeSynthesizer...")
    synthesizer = NarrativeSynthesizer(NarrativeVerbosity.STANDARD)
    
    # Test narrative synthesis
    context = {
        'confidence_score': 0.75,
        'processing_time_ms': 1250,
        'user_input': "What is consciousness?",
        'final_response': "Consciousness is a complex phenomenon involving self-awareness and subjective experience."
    }
    
    narrative_result = await synthesizer.synthesize_narrative(
        process_logger, decision_tracker, metacog_observer,
        NarrativeVerbosity.STANDARD, context
    )
    
    print(f"   Generated narrative: {narrative_result.word_count} words")
    print(f"   Success: {narrative_result.success}")
    print(f"   Quality score: {narrative_result.narrative_quality_score:.3f}")
    print(f"   Events processed: {narrative_result.events_processed}")
    print(f"   Generation time: {narrative_result.generation_time_ms:.1f}ms")
    
    print("\n📖 Sample Narrative:")
    print("-" * 40)
    print(narrative_result.narrative_text)
    print("-" * 40)
    
    return True

async def test_verbosity_modes():
    """Test different narrative verbosity modes"""
    
    print("\n🎛️ Testing Verbosity Modes")
    print("=" * 50)
    
    # Create test components with richer events
    process_logger = ProcessLogger()
    decision_tracker = DecisionTracker()
    metacog_observer = MetacognitiveObserver()
    synthesizer = NarrativeSynthesizer()
    
    # Add varied events
    process_logger.log_phase_transition("Phase 1", "Initial perception processing", 0.5, "neutral")
    process_logger.log_phase_transition("Phase 2", "Conscious state formation with goal evolution", 0.7, "curious")
    process_logger.log_phase_transition("Phase 3", "State evolution toward deeper understanding", 0.8, "focused")
    
    decision_tracker.track_goal_selection("Phase 2", "deep_analysis", ["surface_response", "clarification"], 
                                        "Query complexity requires thorough examination", 0.8)
    
    metacog_observer.observe_recursive_thinking("Phase 3", "Examining my examination of the query", 2, 0.7)
    metacog_observer.observe_pattern_recognition("Phase 3", "Recognizing feedback loops in processing", 0.8)
    
    # Test each verbosity mode
    for verbosity in [NarrativeVerbosity.MINIMAL, NarrativeVerbosity.STANDARD, NarrativeVerbosity.VERBOSE]:
        print(f"\n{verbosity.value.upper()} Mode:")
        
        context = {
            'confidence_score': 0.8,
            'processing_time_ms': 1800,
            'user_input': "How do you think about thinking?",
            'final_response': "Through metacognitive awareness, I observe recursive patterns in my processing."
        }
        
        result = await synthesizer.synthesize_narrative(
            process_logger, decision_tracker, metacog_observer, verbosity, context
        )
        
        print(f"Words: {result.word_count}, Quality: {result.narrative_quality_score:.3f}")
        print(f"Text: {result.narrative_text}")
        print()

def test_configuration():
    """Test configuration loading"""
    
    print("\n⚙️ Testing Configuration")
    print("=" * 50)
    
    from conscious_ai.config.narrative_config import NARRATIVE_CONFIG, ERROR_HANDLING_CONFIG
    
    print(f"Narrative enabled: {NARRATIVE_CONFIG['enabled']}")
    print(f"Default mode: {NARRATIVE_CONFIG['default_mode']}")
    print(f"Max events: {NARRATIVE_CONFIG['max_events']}")
    print(f"Generation timeout: {NARRATIVE_CONFIG['generation_timeout_ms']}ms")
    print(f"Error handling: {ERROR_HANDLING_CONFIG['fallback_enabled']}")
    
    # Test word targets
    word_targets = NARRATIVE_CONFIG.get('word_targets', {})
    for verbosity, (min_words, max_words) in word_targets.items():
        print(f"{verbosity.value}: {min_words}-{max_words} words")

async def main():
    """Run all Phase 5.5 tests"""
    
    print("🧠 Phase 5.5: Narrative Recording of Consciousness - Test Suite")
    print("=" * 70)
    
    try:
        # Test individual components
        await test_phase55_components()
        
        # Test verbosity modes
        await test_verbosity_modes()
        
        # Test configuration
        test_configuration()
        
        print("\n✅ All Phase 5.5 tests completed successfully!")
        print("\nTo test integration with the full pipeline, run:")
        print("python run_consciousness_cli.py --show-narrative --narrative-mode verbose")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)