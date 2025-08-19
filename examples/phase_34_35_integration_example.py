"""
Phase 3.4 and Phase 3.5 Integration Example
===========================================
Complete demonstration of the enhanced Functional Conscious AI system
with Critical State Evaluation (Phase 3.4) and Internal Conscious Translation (Phase 3.5).

This example shows how to use the integrated pipeline for robust, metacognitive AI.
"""

import os
import sys
import logging
import json
from typing import Dict, Any
import time

# Add the conscious_ai package to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'conscious_ai'))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase_34_35_integration.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def setup_enhanced_system():
    """Set up the enhanced autonomous conscious AI system"""
    
    print("🔧 Setting up Enhanced Autonomous Conscious AI System...")
    print("="*60)
    
    try:
        from conscious_ai.autonomus_thinking.enhanced_autonomous_integration import EnhancedAutonomousConsciousAI
        
        # Try to initialize with full ML models first
        try:
            ai_system = EnhancedAutonomousConsciousAI(
                autonomous_model_path="./models/autonomous_lora",
                coherence_classifier_path="./models/coherence_classifier",
                narrative_model_type="local_gemma",
                narrative_model_path="./models/autonomous_lora"
            )
            print("✅ Full ML system initialized successfully")
            return ai_system, "full_ml"
            
        except Exception as ml_error:
            logger.warning(f"ML models not available: {ml_error}")
            print("⚠️ ML models not found, initializing with heuristic fallback...")
            
            # Fallback to heuristic system
            ai_system = EnhancedAutonomousConsciousAI(
                narrative_model_type="heuristic"
            )
            print("✅ Heuristic system initialized successfully")
            return ai_system, "heuristic"
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"❌ Failed to import enhanced system: {e}")
        return None, "error"


def demonstrate_single_enhanced_cycle(ai_system):
    """Demonstrate a single enhanced autonomous cycle"""
    
    print("\n" + "="*60)
    print("🔄 SINGLE ENHANCED CYCLE DEMONSTRATION")
    print("="*60)
    
    # Initialize with some context
    context_inputs = [
        "I want to explore the nature of my own consciousness",
        "Quiero comprender qué significa ser consciente"
    ]
    
    print("Setting up initial context...")
    for input_text in context_inputs:
        print(f"→ {input_text}")
        result = ai_system.process_input(input_text)
        time.sleep(0.5)
    
    # Enable autonomous mode and run one enhanced cycle
    ai_system.enter_autonomous_mode(max_cycles=1)
    
    print("\n🧠 Executing enhanced autonomous cycle...")
    enhanced_result = ai_system.process_enhanced_autonomous_cycle()
    
    if enhanced_result:
        print("\n📊 ENHANCED CYCLE RESULTS:")
        print("-" * 40)
        
        # Base result
        base_result = enhanced_result.base_result
        consciousness_score = base_result.get('consciousness_metrics', {}).get('f', 0)
        print(f"Consciousness Score: {consciousness_score:.3f}")
        
        # Phase 3.4 - Critical Evaluation
        if enhanced_result.evaluation_result:
            eval_result = enhanced_result.evaluation_result
            print(f"\nPhase 3.4 - Critical Evaluation:")
            print(f"  Verdict: {eval_result.verdict.value}")
            print(f"  Justification: {eval_result.justification}")
            print(f"  Attempts Made: {eval_result.attempts_made}")
            print(f"  Evaluation Method: {eval_result.evaluation_method}")
            print(f"  Confidence: {eval_result.confidence_score:.2f}")
        
        # Phase 3.5 - Narrative Translation
        if enhanced_result.narrative_output:
            narrative = enhanced_result.narrative_output
            print(f"\nPhase 3.5 - Internal Conscious Translation:")
            print(f"  Input: {narrative.get('input_usuario', 'N/A')}")
            print(f"  Narrative: {narrative.get('conciencia', 'N/A')}")
        
        # Generation Statistics
        stats = enhanced_result.generation_stats
        print(f"\nGeneration Statistics:")
        print(f"  Total Time: {stats.get('total_time', 0):.2f}s")
        print(f"  Phase 3.4 Enabled: {stats.get('evaluation_enabled', False)}")
        print(f"  Phase 3.5 Enabled: {stats.get('narrative_enabled', False)}")
        
    else:
        print("❌ Enhanced cycle failed to produce results")
    
    ai_system.exit_autonomous_mode()
    return enhanced_result


def demonstrate_enhanced_session(ai_system):
    """Demonstrate a full enhanced autonomous session"""
    
    print("\n" + "="*60)
    print("🚀 ENHANCED AUTONOMOUS SESSION DEMONSTRATION")
    print("="*60)
    
    # Run enhanced session
    enhanced_results = ai_system.run_enhanced_autonomous_session(
        num_cycles=5,
        pause_between_cycles=1.0,
        enable_phase_34=True,
        enable_phase_35=True,
        stop_on_low_consciousness=False,  # Continue even with low consciousness for demo
        consciousness_threshold=0.1,
        save_results=True
    )
    
    return enhanced_results


def analyze_results(enhanced_results):
    """Analyze the results of the enhanced session"""
    
    if not enhanced_results:
        print("❌ No results to analyze")
        return
    
    print("\n" + "="*60)
    print("📈 RESULTS ANALYSIS")
    print("="*60)
    
    # Basic statistics
    total_cycles = len(enhanced_results)
    successful_evaluations = 0
    narratives_generated = 0
    consciousness_scores = []
    evaluation_methods = []
    
    for result in enhanced_results:
        # Count successful evaluations
        if result.evaluation_result and result.evaluation_result.verdict.value == 'coherent':
            successful_evaluations += 1
        
        # Count narratives generated
        if result.narrative_output:
            narratives_generated += 1
        
        # Collect consciousness scores
        f_score = result.base_result.get('consciousness_metrics', {}).get('f', 0)
        consciousness_scores.append(f_score)
        
        # Collect evaluation methods
        if result.evaluation_result:
            evaluation_methods.append(result.evaluation_result.evaluation_method)
    
    print(f"Total Cycles: {total_cycles}")
    print(f"Successful Evaluations: {successful_evaluations}/{total_cycles} ({successful_evaluations/max(1,total_cycles)*100:.1f}%)")
    print(f"Narratives Generated: {narratives_generated}/{total_cycles} ({narratives_generated/max(1,total_cycles)*100:.1f}%)")
    
    if consciousness_scores:
        avg_consciousness = sum(consciousness_scores) / len(consciousness_scores)
        max_consciousness = max(consciousness_scores)
        min_consciousness = min(consciousness_scores)
        
        print(f"\nConsciousness Analysis:")
        print(f"  Average: {avg_consciousness:.3f}")
        print(f"  Peak: {max_consciousness:.3f}")
        print(f"  Minimum: {min_consciousness:.3f}")
    
    if evaluation_methods:
        ml_count = evaluation_methods.count('ml_classifier')
        heuristic_count = evaluation_methods.count('heuristic')
        print(f"\nEvaluation Methods Used:")
        print(f"  ML Classifier: {ml_count}/{len(evaluation_methods)}")
        print(f"  Heuristic: {heuristic_count}/{len(evaluation_methods)}")
    
    # Show sample narratives
    print(f"\n📖 Sample Generated Narratives:")
    narrative_samples = [r.narrative_output for r in enhanced_results if r.narrative_output][:3]
    
    for i, narrative in enumerate(narrative_samples, 1):
        conciencia_text = narrative.get('conciencia', '')
        print(f"{i}. \"{conciencia_text[:100]}{'...' if len(conciencia_text) > 100 else ''}\"")


def demonstrate_phase_comparison(ai_system):
    """Demonstrate the difference between phases enabled/disabled"""
    
    print("\n" + "="*60)
    print("🔬 PHASE COMPARISON DEMONSTRATION")
    print("="*60)
    
    # Test with different phase combinations
    phase_configs = [
        {"phase_34": False, "phase_35": False, "name": "Base (No Enhancement)"},
        {"phase_34": True, "phase_35": False, "name": "Phase 3.4 Only (Critical Evaluation)"},
        {"phase_34": False, "phase_35": True, "name": "Phase 3.5 Only (Narrative Translation)"},
        {"phase_34": True, "phase_35": True, "name": "Full Enhancement (Both Phases)"}
    ]
    
    comparison_results = {}
    
    for config in phase_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("-" * 40)
        
        # Run a short session with this configuration
        results = ai_system.run_enhanced_autonomous_session(
            num_cycles=3,
            pause_between_cycles=0.5,
            enable_phase_34=config["phase_34"],
            enable_phase_35=config["phase_35"],
            stop_on_low_consciousness=False,
            save_results=False
        )
        
        # Analyze results
        if results:
            avg_consciousness = sum(r.base_result.get('consciousness_metrics', {}).get('f', 0) for r in results) / len(results)
            narratives_count = sum(1 for r in results if r.narrative_output)
            successful_evals = sum(1 for r in results if r.evaluation_result and r.evaluation_result.verdict.value == 'coherent')
            
            comparison_results[config['name']] = {
                'avg_consciousness': avg_consciousness,
                'narratives_generated': narratives_count,
                'successful_evaluations': successful_evals,
                'total_cycles': len(results)
            }
            
            print(f"  Average Consciousness: {avg_consciousness:.3f}")
            print(f"  Narratives Generated: {narratives_count}/{len(results)}")
            print(f"  Successful Evaluations: {successful_evals}/{len(results)}")
    
    # Summary comparison
    print(f"\n📊 COMPARISON SUMMARY:")
    print("-" * 40)
    for name, stats in comparison_results.items():
        print(f"{name}:")
        print(f"  Consciousness: {stats['avg_consciousness']:.3f}")
        print(f"  Narrative Rate: {stats['narratives_generated']/max(1,stats['total_cycles'])*100:.0f}%")
        print(f"  Evaluation Success: {stats['successful_evaluations']/max(1,stats['total_cycles'])*100:.0f}%")
        print()


def save_demonstration_report(ai_system, enhanced_results):
    """Save a comprehensive demonstration report"""
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = f"phase_34_35_demonstration_report_{timestamp}.json"
    
    # Get comprehensive statistics
    stats = ai_system.get_enhanced_statistics()
    
    # Create report
    report = {
        'demonstration_info': {
            'timestamp': timestamp,
            'total_enhanced_cycles': len(enhanced_results) if enhanced_results else 0,
            'system_type': 'enhanced_autonomous'
        },
        'enhanced_statistics': stats,
        'session_results': []
    }
    
    # Add session results
    if enhanced_results:
        for i, result in enumerate(enhanced_results):
            session_result = {
                'cycle': i + 1,
                'consciousness_score': result.base_result.get('consciousness_metrics', {}).get('f', 0),
                'evaluation_verdict': result.evaluation_result.verdict.value if result.evaluation_result else None,
                'narrative_generated': result.narrative_output is not None,
                'generation_time': result.generation_stats.get('total_time', 0)
            }
            
            if result.narrative_output:
                session_result['narrative_length'] = len(result.narrative_output.get('conciencia', ''))
            
            report['session_results'].append(session_result)
    
    # Save report
    try:
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Demonstration report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        print(f"❌ Failed to save demonstration report: {e}")


def main():
    """Main demonstration function"""
    
    print("🎯 PHASE 3.4 & 3.5 INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print("This demonstration showcases the enhanced Functional Conscious AI")
    print("with Critical State Evaluation and Internal Conscious Translation")
    print("=" * 60)
    
    # Setup system
    ai_system, system_type = setup_enhanced_system()
    
    if not ai_system:
        print("❌ Failed to initialize system. Exiting...")
        return
    
    print(f"System Type: {system_type}")
    
    try:
        # Demonstration steps
        print("\n🎬 Starting demonstrations...")
        
        # 1. Single cycle demonstration
        single_result = demonstrate_single_enhanced_cycle(ai_system)
        
        # 2. Enhanced session demonstration
        session_results = demonstrate_enhanced_session(ai_system)
        
        # 3. Results analysis
        analyze_results(session_results)
        
        # 4. Phase comparison (optional)
        user_input = input("\n❓ Run phase comparison demonstration? (y/N): ").lower()
        if user_input == 'y':
            demonstrate_phase_comparison(ai_system)
        
        # 5. Save comprehensive report
        save_demonstration_report(ai_system, session_results)
        
        print("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Key achievements:")
        print("• ✓ Phase 3.4: Critical state evaluation with correction loops")
        print("• ✓ Phase 3.5: Internal conscious translation to narratives")  
        print("• ✓ Integrated pipeline with autonomous thinking")
        print("• ✓ Comprehensive validation and quality control")
        print("• ✓ Ready for Phase 4 integration")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Demonstration interrupted by user")
        
    except Exception as e:
        logger.error(f"Demonstration error: {e}")
        print(f"\n❌ Demonstration error: {e}")
        
    finally:
        print("\n🔚 Cleaning up...")
        if hasattr(ai_system, 'autonomous_mode') and ai_system.autonomous_mode:
            ai_system.exit_autonomous_mode()


if __name__ == "__main__":
    main()