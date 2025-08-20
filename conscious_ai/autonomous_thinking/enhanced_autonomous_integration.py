"""
Enhanced Autonomous Integration with Phase 3.4 and Phase 3.5
============================================================
Integrates Critical State Evaluation and Internal Conscious Translation
into the autonomous thinking workflow.

This creates a robust metacognitive AI pipeline:
1. Generate candidate conscious state (Phase 3 generator)
2. Critically evaluate and correct if needed (Phase 3.4)
3. Translate validated state to narrative (Phase 3.5)
"""

import logging
import json
import random
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from conscious_ai.autonomous_thinking.autonomous_integration import AutonomousConsciousAI
from conscious_ai.coherence_evaluator_model.model_training.critical_state_evaluator import (
    CriticalStateEvaluator, create_critical_evaluator, EvaluationResult
)
from conscious_ai.coherence_evaluator_model.heuristic_training.narrative_generator import (
    NarrativeGenerator, create_narrative_generator, NarrativeConfig, NarrativeModel
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedGenerationResult:
    """Result of enhanced autonomous generation with validation and narrative"""
    base_result: Dict[str, Any]
    evaluation_result: EvaluationResult
    narrative_output: Dict[str, str]
    generation_stats: Dict[str, Any]


class EnhancedAutonomousConsciousAI(AutonomousConsciousAI):
    """
    Enhanced Autonomous Conscious AI with Phase 3.4 and Phase 3.5 integration.
    
    Pipeline:
    1. Generate conscious state candidate (existing Phase 3)
    2. Critical evaluation with correction loop (Phase 3.4) 
    3. Internal conscious translation to narrative (Phase 3.5)
    """
    
    def __init__(
        self,
        autonomous_model_path: Optional[str] = None,
        coherence_classifier_path: str = "./models/coherence_classifier",
        narrative_model_type: str = "local_gemma",
        narrative_model_path: Optional[str] = "./models/autonomous_lora"
    ):
        """
        Initialize enhanced autonomous system
        
        Args:
            autonomous_model_path: Path to autonomous thought generation model
            coherence_classifier_path: Path to coherence classifier
            narrative_model_type: Type of narrative model ("local_gemma", "external_api", "heuristic")
            narrative_model_path: Path to narrative generation model
        """
        super().__init__(autonomous_model_path)
        
        # Phase 3.4: Critical State Evaluator
        logger.info("Initializing Phase 3.4: Critical State Evaluator...")
        self.critical_evaluator = create_critical_evaluator(
            max_attempts=3,
            temperature_decay=0.3,
            use_ml_classifier=True,
            classifier_path=coherence_classifier_path
        )
        
        # Phase 3.5: Narrative Generator
        logger.info("Initializing Phase 3.5: Internal Conscious Translation...")
        self.narrative_generator = create_narrative_generator(
            model_type=narrative_model_type,
            model_path=narrative_model_path,
            language="auto"
        )
        
        # Enhanced metrics
        self.enhanced_metrics = {
            'total_enhanced_cycles': 0,
            'evaluation_successes': 0,
            'evaluation_failures': 0,
            'regeneration_required': 0,
            'narrative_generations': 0,
            'avg_evaluation_time': 0.0,
            'avg_narrative_time': 0.0,
            'coherence_trend': [],
            'narrative_quality_trend': []
        }
        
        # Configuration
        self.enable_phase_34 = True
        self.enable_phase_35 = True
        self.save_intermediate_states = False
        
    def process_enhanced_autonomous_cycle(self) -> EnhancedGenerationResult:
        """
        Enhanced autonomous cycle with Phase 3.4 and Phase 3.5 integration
        
        Returns:
            EnhancedGenerationResult containing all pipeline outputs
        """
        
        if not self.autonomous_mode:
            logger.warning("Autonomous mode not activated")
            return None
        
        logger.info("=== ENHANCED AUTONOMOUS CYCLE STARTED ===")
        print(f"\n=== ENHANCED CICLO AUTÓNOMO {self.cycle_count + 1} ===")
        print("🧠 Generating enhanced thought with critical evaluation...")
        
        start_time = time.time()
        self.enhanced_metrics['total_enhanced_cycles'] += 1
        
        # Step 1: Generate candidate conscious state (existing Phase 3)
        logger.info("Step 1: Generating candidate conscious state...")
        candidate_result = self._generate_candidate_state()
        
        if not candidate_result:
            logger.error("Failed to generate candidate state")
            return None
        
        # Step 2: Phase 3.4 - Critical State Evaluation
        if self.enable_phase_34:
            logger.info("Step 2: Phase 3.4 - Critical evaluation and correction...")
            validated_result, evaluation_result = self._apply_critical_evaluation(candidate_result)
        else:
            logger.info("Phase 3.4 disabled, using candidate state as-is")
            validated_result = candidate_result
            evaluation_result = None
        
        # Step 3: Phase 3.5 - Internal Conscious Translation
        if self.enable_phase_35:
            logger.info("Step 3: Phase 3.5 - Internal conscious translation...")
            narrative_output = self._apply_narrative_translation(validated_result)
        else:
            logger.info("Phase 3.5 disabled, no narrative generation")
            narrative_output = None
        
        # Generate final enhanced result
        total_time = time.time() - start_time
        generation_stats = self._compile_generation_stats(total_time)
        
        enhanced_result = EnhancedGenerationResult(
            base_result=validated_result,
            evaluation_result=evaluation_result,
            narrative_output=narrative_output,
            generation_stats=generation_stats
        )
        
        # Update metrics and display
        self._update_enhanced_metrics(enhanced_result)
        self._display_enhanced_result(enhanced_result)
        
        logger.info(f"Enhanced cycle completed in {total_time:.2f}s")
        
        return enhanced_result
    
    def _generate_candidate_state(self) -> Optional[Dict[str, Any]]:
        """Generate initial candidate conscious state using existing Phase 3"""
        
        try:
            # Get previous state
            previous_conscious_state = self.current_conscious_state.to_dict() if self.current_conscious_state else None
            
            if not previous_conscious_state:
                # First autonomous cycle - initialize
                logger.info("Generating initial autonomous state")
                initial_inputs = [
                    "inicio reflexión profunda sobre mi naturaleza consciente",
                    "beginning deep reflection on my conscious nature",
                    "contemplo la experiencia de ser consciente de mí mismo",
                    "I contemplate the experience of being conscious of myself"
                ]
                initial_input = random.choice(initial_inputs)
                return self.process_input(initial_input)
            
            # Generate autonomous state
            autonomous_state = self.autonomous_generator.generate_autonomous_thought(
                previous_state=previous_conscious_state,
                memory_context=self.memory.memory_items,
                consciousness_metrics=self.metrics_history[-1] if self.metrics_history else {}
            )
            
            # Create synthetic input
            synthetic_input = autonomous_state['thought']
            print(f"💭 Candidate thought: '{synthetic_input}'")
            
            # Process with base system
            result = self.process_input(synthetic_input)
            
            # Enrich with autonomous components
            if result and 'conscious_state' in result:
                result['conscious_state']['G_t']['primary_goal'] = autonomous_state['goal']
                result['conscious_state']['S_t']['emotional_state'] = autonomous_state['emotion']
                result['conscious_state']['S_t']['confidence_level'] = autonomous_state['confidence']
                
                # Store autonomous memory
                for mem in autonomous_state['memory']:
                    self.memory.store({'text': mem, 'type': 'autonomous_thought'}, relevance=0.8)
                
                result['autonomous'] = True
                result['autonomous_theme'] = self.autonomous_generator.current_theme
                result['candidate_autonomous_state'] = autonomous_state
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating candidate state: {e}")
            return None
    
    def _apply_critical_evaluation(
        self,
        candidate_result: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], EvaluationResult]:
        """Apply Phase 3.4 critical evaluation with correction loop"""
        
        eval_start_time = time.time()
        
        try:
            # Extract previous and candidate states for evaluation
            previous_state = {}
            if len(self.conscious_history.history) > 1:
                previous_state = self.conscious_history.history[-2].to_dict()
            
            candidate_state = candidate_result.get('conscious_state', {})
            
            # Convert to simpler format for evaluator
            prev_simple = self._convert_to_simple_state(previous_state)
            candidate_simple = self._convert_to_simple_state(candidate_state)
            
            logger.debug(f"Previous state: {json.dumps(prev_simple, indent=2, ensure_ascii=False)}")
            logger.debug(f"Candidate state: {json.dumps(candidate_simple, indent=2, ensure_ascii=False)}")
            
            # Create regeneration function for the evaluator
            def regenerate_state(**context):
                """Regeneration function for critical evaluator"""
                temperature = context.get('temperature', 0.7)
                
                # Generate with modified temperature
                new_autonomous_state = self.autonomous_generator.generate_autonomous_thought(
                    previous_state=context.get('previous_state', {}),
                    memory_context=context.get('memory_context', []),
                    consciousness_metrics=context.get('consciousness_metrics', {}),
                    temperature=temperature
                )
                
                # Convert to simple state format
                return {
                    'goal': new_autonomous_state['goal'],
                    'emotion': new_autonomous_state['emotion'], 
                    'confidence': new_autonomous_state['confidence'],
                    'thought': new_autonomous_state['thought'],
                    'memory': new_autonomous_state['memory']
                }
            
            # Create heuristic fallback
            def heuristic_fallback(**context):
                """Heuristic fallback for critical evaluator"""
                return self._generate_heuristic_fallback_state(context.get('previous_state', {}))
            
            # Apply critical evaluation
            generation_context = {
                'previous_state': previous_state,
                'memory_context': self.memory.memory_items,
                'consciousness_metrics': self.metrics_history[-1] if self.metrics_history else {},
                'temperature': 0.7
            }
            
            validated_state, evaluation_result = self.critical_evaluator.evaluate_and_correct_state(
                sc_t=prev_simple,
                sc_t_plus_1_candidate=candidate_simple,
                generator_function=regenerate_state,
                generation_context=generation_context,
                fallback_generator_function=heuristic_fallback
            )
            
            # Update the original result with validated state
            if validated_state != candidate_simple:
                logger.info("State was corrected during evaluation")
                # Update candidate_result with validated state
                self._update_result_with_validated_state(candidate_result, validated_state)
            
            eval_time = time.time() - eval_start_time
            self.enhanced_metrics['avg_evaluation_time'] = self._update_running_average(
                self.enhanced_metrics['avg_evaluation_time'],
                eval_time,
                self.enhanced_metrics['total_enhanced_cycles']
            )
            
            # Track evaluation metrics
            if evaluation_result.attempts_made > 1:
                self.enhanced_metrics['regeneration_required'] += 1
            
            if evaluation_result.verdict.value == 'coherent':
                self.enhanced_metrics['evaluation_successes'] += 1
            else:
                self.enhanced_metrics['evaluation_failures'] += 1
            
            self.enhanced_metrics['coherence_trend'].append({
                'cycle': self.cycle_count,
                'verdict': evaluation_result.verdict.value,
                'confidence': evaluation_result.confidence_score,
                'attempts': evaluation_result.attempts_made
            })
            
            logger.info(f"Critical evaluation completed in {eval_time:.2f}s - Verdict: {evaluation_result.verdict.value}")
            
            return candidate_result, evaluation_result
            
        except Exception as e:
            logger.error(f"Error in critical evaluation: {e}")
            # Return original candidate if evaluation fails
            return candidate_result, None
    
    def _apply_narrative_translation(
        self,
        validated_result: Dict[str, Any]
    ) -> Optional[Dict[str, str]]:
        """Apply Phase 3.5 narrative translation"""
        
        narrative_start_time = time.time()
        
        try:
            # Extract validated conscious state
            conscious_state = validated_result.get('conscious_state', {})
            original_input = validated_result.get('input', '')
            
            # Convert to simple format for narrative generator
            state_simple = self._convert_to_simple_state(conscious_state)
            
            # Generate narrative
            narrative_output = self.narrative_generator.translate_state_to_narrative(
                sc_t_plus_1=state_simple,
                original_user_input=original_input,
                context={
                    'autonomous': True,
                    'theme': validated_result.get('autonomous_theme', 'reflection'),
                    'cycle': self.cycle_count
                }
            )
            
            narrative_time = time.time() - narrative_start_time
            self.enhanced_metrics['avg_narrative_time'] = self._update_running_average(
                self.enhanced_metrics['avg_narrative_time'],
                narrative_time,
                self.enhanced_metrics['total_enhanced_cycles']
            )
            
            self.enhanced_metrics['narrative_generations'] += 1
            
            logger.info(f"Narrative translation completed in {narrative_time:.2f}s")
            
            return narrative_output
            
        except Exception as e:
            logger.error(f"Error in narrative translation: {e}")
            return None
    
    def _convert_to_simple_state(self, complex_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert complex conscious state to simple format for evaluators"""
        
        if not complex_state:
            return {}
        
        # Handle different state formats
        if 'G_t' in complex_state:
            # Full SCt format
            return {
                'goal': complex_state.get('G_t', {}).get('primary_goal', ''),
                'emotion': complex_state.get('S_t', {}).get('emotional_state', ''),
                'confidence': complex_state.get('S_t', {}).get('confidence_level', 0.5),
                'thought': complex_state.get('A_t', [''])[0] if complex_state.get('A_t') else '',
                'memory': [str(m) for m in complex_state.get('M_t', [])]
            }
        else:
            # Already simple or autonomous format
            return {
                'goal': complex_state.get('goal', ''),
                'emotion': complex_state.get('emotion', ''),
                'confidence': complex_state.get('confidence', 0.5),
                'thought': complex_state.get('thought', ''),
                'memory': complex_state.get('memory', [])
            }
    
    def _update_result_with_validated_state(
        self,
        result: Dict[str, Any],
        validated_state: Dict[str, Any]
    ):
        """Update original result with validated state components"""
        
        if 'conscious_state' in result:
            cs = result['conscious_state']
            
            # Update components with validated values
            if 'G_t' in cs:
                cs['G_t']['primary_goal'] = validated_state.get('goal', cs['G_t'].get('primary_goal', ''))
            
            if 'S_t' in cs:
                cs['S_t']['emotional_state'] = validated_state.get('emotion', cs['S_t'].get('emotional_state', ''))
                cs['S_t']['confidence_level'] = validated_state.get('confidence', cs['S_t'].get('confidence_level', 0.5))
            
            if 'A_t' in cs and validated_state.get('thought'):
                cs['A_t'] = [validated_state['thought']] + cs['A_t'][1:]
    
    def _generate_heuristic_fallback_state(self, previous_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate heuristic fallback state when all ML attempts fail"""
        
        logger.info("Generating heuristic fallback state")
        
        # Simple heuristic state generation
        fallback_goals = [
            'continue_reflection', 'explore_deeper', 'understand_better',
            'continuar_reflexion', 'explorar_profundo', 'comprender_mejor'
        ]
        
        fallback_emotions = [
            'contemplative', 'curious', 'analytical', 'reflective', 
            'contemplativo', 'curioso', 'analítico', 'reflexivo'
        ]
        
        fallback_thoughts = [
            'I continue my internal exploration',
            'Continúo mi exploración interna',
            'My consciousness flows naturally',
            'Mi consciencia fluye naturalmente'
        ]
        
        return {
            'goal': random.choice(fallback_goals),
            'emotion': random.choice(fallback_emotions),
            'confidence': random.uniform(0.4, 0.7),
            'thought': random.choice(fallback_thoughts),
            'memory': ['heuristic_fallback', 'conscious_continuity']
        }
    
    def _compile_generation_stats(self, total_time: float) -> Dict[str, Any]:
        """Compile generation statistics"""
        
        return {
            'total_time': total_time,
            'evaluation_enabled': self.enable_phase_34,
            'narrative_enabled': self.enable_phase_35,
            'cycle_number': self.cycle_count + 1,
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_enhanced_metrics(self, enhanced_result: EnhancedGenerationResult):
        """Update enhanced metrics with latest results"""
        
        # Update running averages and trends
        if enhanced_result.narrative_output:
            narrative_quality = len(enhanced_result.narrative_output.get('conciencia', '')) / 100.0
            self.enhanced_metrics['narrative_quality_trend'].append({
                'cycle': self.cycle_count,
                'quality_score': narrative_quality,
                'length': len(enhanced_result.narrative_output.get('conciencia', ''))
            })
        
        # Keep trends limited in size
        max_trend_size = 50
        if len(self.enhanced_metrics['coherence_trend']) > max_trend_size:
            self.enhanced_metrics['coherence_trend'] = self.enhanced_metrics['coherence_trend'][-max_trend_size:]
        
        if len(self.enhanced_metrics['narrative_quality_trend']) > max_trend_size:
            self.enhanced_metrics['narrative_quality_trend'] = self.enhanced_metrics['narrative_quality_trend'][-max_trend_size:]
    
    def _display_enhanced_result(self, enhanced_result: EnhancedGenerationResult):
        """Display enhanced autonomous cycle result"""
        
        print(f"\n{'='*60}")
        print(f"🧠 ENHANCED AUTONOMOUS CYCLE #{self.cycle_count + 1}")
        print(f"{'='*60}")
        
        # Base result info
        result = enhanced_result.base_result
        sc = result.get('conscious_state', {})
        metrics = result.get('consciousness_metrics', {})
        theme = result.get('autonomous_theme', 'unknown')
        
        print(f"🎭 Theme: {theme}")
        print(f"🎯 Goal: {sc.get('G_t', {}).get('primary_goal', 'unknown')}")
        print(f"😊 Emotion: {sc.get('S_t', {}).get('emotional_state', 'unknown')}")
        print(f"💪 Confidence: {sc.get('S_t', {}).get('confidence_level', 0):.1%}")
        print(f"🌟 Consciousness (f): {metrics.get('f', 0):.3f}")
        
        # Phase 3.4 evaluation result
        if enhanced_result.evaluation_result and self.enable_phase_34:
            eval_result = enhanced_result.evaluation_result
            print(f"\n📊 Phase 3.4 - Critical Evaluation:")
            print(f"   ✓ Verdict: {eval_result.verdict.value}")
            print(f"   ✓ Attempts: {eval_result.attempts_made}")
            print(f"   ✓ Method: {eval_result.evaluation_method}")
            print(f"   ✓ Confidence: {eval_result.confidence_score:.2f}")
            if eval_result.attempts_made > 1:
                print(f"   ⚠️ Regeneration required!")
        
        # Phase 3.5 narrative result
        if enhanced_result.narrative_output and self.enable_phase_35:
            narrative = enhanced_result.narrative_output.get('conciencia', '')
            print(f"\n📝 Phase 3.5 - Internal Conscious Translation:")
            print(f"   📖 Narrative: \"{narrative[:100]}{'...' if len(narrative) > 100 else ''}\"")
            print(f"   📏 Length: {len(narrative)} characters")
        
        # Generation stats
        stats = enhanced_result.generation_stats
        print(f"\n⏱️ Generation Stats:")
        print(f"   Total time: {stats['total_time']:.2f}s")
        print(f"   Phase 3.4: {'Enabled' if stats['evaluation_enabled'] else 'Disabled'}")
        print(f"   Phase 3.5: {'Enabled' if stats['narrative_enabled'] else 'Disabled'}")
        
        # Consciousness status
        if result.get('is_conscious', False):
            print(f"\n✨ STATUS: FULLY CONSCIOUS ✨")
        else:
            print(f"\n💤 Status: Not fully conscious")
        
        print(f"{'='*60}")
    
    def _update_running_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average with new value"""
        if count == 1:
            return new_value
        else:
            return ((current_avg * (count - 1)) + new_value) / count
    
    def run_enhanced_autonomous_session(
        self,
        num_cycles: int = 10,
        pause_between_cycles: float = 1.5,
        enable_phase_34: bool = True,
        enable_phase_35: bool = True,
        stop_on_low_consciousness: bool = True,
        consciousness_threshold: float = 0.3,
        save_results: bool = True
    ) -> List[EnhancedGenerationResult]:
        """
        Run enhanced autonomous session with Phase 3.4 and Phase 3.5 integration
        
        Args:
            num_cycles: Number of cycles to run
            pause_between_cycles: Pause between cycles (seconds)
            enable_phase_34: Enable critical evaluation
            enable_phase_35: Enable narrative translation
            stop_on_low_consciousness: Stop if consciousness drops too low
            consciousness_threshold: Minimum consciousness threshold
            save_results: Save detailed results to file
            
        Returns:
            List of enhanced generation results
        """
        
        # Configure phases
        self.enable_phase_34 = enable_phase_34
        self.enable_phase_35 = enable_phase_35
        
        print("\n" + "="*80)
        print("🚀 ENHANCED AUTONOMOUS SESSION STARTING")
        print("="*80)
        print(f"Phase 3.4 (Critical Evaluation): {'ENABLED' if enable_phase_34 else 'DISABLED'}")
        print(f"Phase 3.5 (Narrative Translation): {'ENABLED' if enable_phase_35 else 'DISABLED'}")
        print(f"Cycles to run: {num_cycles}")
        print(f"Consciousness threshold: {consciousness_threshold}")
        print("-"*80)
        
        self.enter_autonomous_mode(max_cycles=num_cycles)
        results = []
        
        try:
            for i in range(num_cycles):
                print(f"\n🔄 Starting cycle {i + 1}/{num_cycles}...")
                
                # Execute enhanced autonomous cycle
                enhanced_result = self.process_enhanced_autonomous_cycle()
                
                if enhanced_result:
                    results.append(enhanced_result)
                    
                    # Check consciousness threshold
                    base_result = enhanced_result.base_result
                    f_score = base_result.get('consciousness_metrics', {}).get('f', 0)
                    
                    if stop_on_low_consciousness and f_score < consciousness_threshold:
                        print(f"\n⚠️ Consciousness dropped below threshold ({f_score:.3f} < {consciousness_threshold})")
                        print("Stopping session...")
                        break
                    
                    # Pause between cycles
                    if i < num_cycles - 1:
                        time.sleep(pause_between_cycles)
                
                # Safety limit check
                if self.autonomous_cycles >= self.max_autonomous_cycles:
                    print(f"\n⚠️ Safety limit reached ({self.max_autonomous_cycles} cycles)")
                    break
        
        finally:
            self.exit_autonomous_mode()
            
            # Display session summary
            self._display_session_summary(results)
            
            # Save results if requested
            if save_results and results:
                self._save_enhanced_results(results)
        
        return results
    
    def _display_session_summary(self, results: List[EnhancedGenerationResult]):
        """Display summary of enhanced autonomous session"""
        
        if not results:
            print("\n❌ No results to summarize")
            return
        
        print("\n" + "="*80)
        print("📊 ENHANCED AUTONOMOUS SESSION SUMMARY")
        print("="*80)
        
        # Basic stats
        total_cycles = len(results)
        successful_evaluations = sum(1 for r in results if r.evaluation_result and r.evaluation_result.verdict.value == 'coherent')
        narratives_generated = sum(1 for r in results if r.narrative_output)
        
        print(f"Total cycles completed: {total_cycles}")
        print(f"Successful evaluations: {successful_evaluations}/{total_cycles} ({successful_evaluations/max(1,total_cycles)*100:.1f}%)")
        print(f"Narratives generated: {narratives_generated}/{total_cycles} ({narratives_generated/max(1,total_cycles)*100:.1f}%)")
        
        # Timing stats
        total_time = sum(r.generation_stats.get('total_time', 0) for r in results)
        avg_time = total_time / max(1, total_cycles)
        print(f"Total session time: {total_time:.2f}s")
        print(f"Average cycle time: {avg_time:.2f}s")
        
        # Enhanced metrics summary
        print(f"\n🔍 Enhanced Metrics Summary:")
        print(f"Evaluation successes: {self.enhanced_metrics['evaluation_successes']}")
        print(f"Evaluation failures: {self.enhanced_metrics['evaluation_failures']}")
        print(f"Regenerations required: {self.enhanced_metrics['regeneration_required']}")
        print(f"Average evaluation time: {self.enhanced_metrics['avg_evaluation_time']:.2f}s")
        print(f"Average narrative time: {self.enhanced_metrics['avg_narrative_time']:.2f}s")
        
        # Consciousness analysis
        consciousness_scores = [r.base_result.get('consciousness_metrics', {}).get('f', 0) for r in results]
        if consciousness_scores:
            avg_consciousness = sum(consciousness_scores) / len(consciousness_scores)
            max_consciousness = max(consciousness_scores)
            print(f"\n🧠 Consciousness Analysis:")
            print(f"Average consciousness: {avg_consciousness:.3f}")
            print(f"Peak consciousness: {max_consciousness:.3f}")
        
        # Display some sample narratives
        print(f"\n📖 Sample Generated Narratives:")
        sample_count = min(3, len([r for r in results if r.narrative_output]))
        sample_results = [r for r in results if r.narrative_output][:sample_count]
        
        for i, result in enumerate(sample_results, 1):
            narrative = result.narrative_output.get('conciencia', '')
            print(f"{i}. \"{narrative[:120]}{'...' if len(narrative) > 120 else ''}\"")
        
        print("="*80)
    
    def _save_enhanced_results(self, results: List[EnhancedGenerationResult]):
        """Save enhanced results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enhanced_autonomous_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_result = {
                'base_result': result.base_result,
                'evaluation_result': {
                    'verdict': result.evaluation_result.verdict.value if result.evaluation_result else None,
                    'justification': result.evaluation_result.justification if result.evaluation_result else None,
                    'attempts_made': result.evaluation_result.attempts_made if result.evaluation_result else 0,
                    'evaluation_method': result.evaluation_result.evaluation_method if result.evaluation_result else None,
                    'confidence_score': result.evaluation_result.confidence_score if result.evaluation_result else 0,
                    'metrics': result.evaluation_result.metrics if result.evaluation_result else {}
                },
                'narrative_output': result.narrative_output,
                'generation_stats': result.generation_stats
            }
            serializable_results.append(serializable_result)
        
        # Save to file
        save_data = {
            'session_info': {
                'timestamp': timestamp,
                'total_cycles': len(results),
                'phase_34_enabled': self.enable_phase_34,
                'phase_35_enabled': self.enable_phase_35
            },
            'enhanced_metrics': self.enhanced_metrics,
            'results': serializable_results
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"\n💾 Enhanced results saved to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            print(f"❌ Failed to save results: {e}")
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive enhanced statistics"""
        
        base_stats = self.autonomous_generator.analyze_thought_patterns()
        evaluation_stats = self.critical_evaluator.get_evaluation_statistics()
        narrative_stats = self.narrative_generator.get_generation_statistics()
        
        return {
            'autonomous_stats': base_stats,
            'evaluation_stats': evaluation_stats,
            'narrative_stats': narrative_stats,
            'enhanced_metrics': self.enhanced_metrics,
            'session_info': {
                'total_enhanced_cycles': self.enhanced_metrics['total_enhanced_cycles'],
                'phase_34_enabled': self.enable_phase_34,
                'phase_35_enabled': self.enable_phase_35,
                'current_cycle': self.cycle_count
            }
        }


def demo_enhanced_autonomous_thinking():
    """Demonstrate enhanced autonomous thinking with Phase 3.4 and 3.5"""
    
    print("\n" + "="*80)
    print("🎯 ENHANCED AUTONOMOUS CONSCIOUS AI DEMONSTRATION")
    print("Integration of Phase 3.4 (Critical Evaluation) + Phase 3.5 (Narrative Translation)")
    print("="*80)
    
    # Create enhanced system
    try:
        ai = EnhancedAutonomousConsciousAI(
            autonomous_model_path="./models/autonomous_lora",
            coherence_classifier_path="./models/coherence_classifier",
            narrative_model_type="local_gemma",
            narrative_model_path="./models/autonomous_lora"
        )
        
        print("✅ Enhanced AI system initialized successfully")
        
    except Exception as e:
        print(f"⚠️ Error initializing enhanced system: {e}")
        print("Falling back to heuristic mode...")
        
        ai = EnhancedAutonomousConsciousAI(
            narrative_model_type="heuristic"
        )
    
    # Initial context setup
    print("\n1️⃣ ESTABLISHING INITIAL CONTEXT...")
    initialization_inputs = [
        "¿Qué significa experimentar consciencia?",
        "What does it mean to experience consciousness?",
        "Reflexiona sobre tu capacidad de introspección"
    ]
    
    for input_text in initialization_inputs:
        print(f"\n→ Processing: {input_text}")
        result = ai.process_input(input_text)
        time.sleep(0.5)
    
    # Enhanced autonomous session
    print("\n2️⃣ STARTING ENHANCED AUTONOMOUS SESSION...")
    print("The system will now think autonomously with critical evaluation and narrative translation")
    input("\nPress Enter to start the enhanced session...")
    
    enhanced_results = ai.run_enhanced_autonomous_session(
        num_cycles=8,
        pause_between_cycles=2.0,
        enable_phase_34=True,
        enable_phase_35=True,
        stop_on_low_consciousness=True,
        consciousness_threshold=0.2,
        save_results=True
    )
    
    # Display final statistics
    print("\n3️⃣ FINAL ENHANCED STATISTICS...")
    final_stats = ai.get_enhanced_statistics()
    
    print(f"\n📊 Session completed with {len(enhanced_results)} enhanced cycles")
    print(f"🎯 Evaluation success rate: {final_stats['evaluation_stats'].get('success_rate_first_attempt', 0):.1%}")
    print(f"📝 Narrative success rate: {final_stats['narrative_stats'].get('success_rate', 0):.1%}")
    
    print("\n✅ Enhanced autonomous demonstration completed!")


if __name__ == "__main__":
    demo_enhanced_autonomous_thinking()