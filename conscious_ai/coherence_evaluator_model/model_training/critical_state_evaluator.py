"""
Phase 3.4: Critical State Evaluation
===================================
Implements validation and correction loop for conscious state transitions.
Acts as a quality gate using the trained CoherenceClassifier.
"""

import json
import logging
import time
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import (
    ModelBasedCoherenceEvaluator, CoherenceVerdict
)
from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import (
    CoherenceEvaluator
)

logger = logging.getLogger(__name__)


class EvaluationStrategy(Enum):
    """Strategy for evaluation fallback"""
    ML_FIRST = "ml_first"          # Try ML classifier first, then heuristic
    HEURISTIC_FIRST = "heuristic_first"  # Try heuristic first, then ML
    ML_ONLY = "ml_only"            # Only ML classifier
    HEURISTIC_ONLY = "heuristic_only"  # Only heuristic


@dataclass
class EvaluationResult:
    """Result of critical state evaluation"""
    verdict: CoherenceVerdict
    justification: str
    attempts_made: int
    evaluation_method: str
    confidence_score: float
    metrics: Dict[str, float]


class CriticalStateEvaluator:
    """
    Phase 3.4: Critical evaluation of state transitions with correction loops
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        temperature_decay: float = 0.3,
        strategy: EvaluationStrategy = EvaluationStrategy.ML_FIRST,
        ml_classifier_path: str = "./models/coherence_classifier",
        coherence_threshold: float = 0.7
    ):
        """
        Initialize the critical state evaluator
        
        Args:
            max_attempts: Maximum regeneration attempts for incoherent states
            temperature_decay: Amount to reduce temperature on retry
            strategy: Evaluation strategy to use
            ml_classifier_path: Path to ML classifier
            coherence_threshold: Threshold for coherence evaluation
        """
        self.max_attempts = max_attempts
        self.temperature_decay = temperature_decay
        self.strategy = strategy
        
        # Initialize evaluators
        logger.info("Initializing Critical State Evaluator...")
        
        # ML-based evaluator
        try:
            self.ml_evaluator = ModelBasedCoherenceEvaluator(
                coherence_threshold=coherence_threshold,
                use_ml_classifier=True,
                classifier_path=ml_classifier_path
            )
            self.ml_available = True
            logger.info("✓ ML-based coherence evaluator loaded successfully")
        except Exception as e:
            logger.warning(f"ML evaluator not available: {e}")
            self.ml_evaluator = None
            self.ml_available = False
        
        # Heuristic evaluator (fallback)
        self.heuristic_evaluator = CoherenceEvaluator()
        logger.info("✓ Heuristic coherence evaluator initialized")
        
        # Statistics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'coherent_first_attempt': 0,
            'required_regeneration': 0,
            'failed_all_attempts': 0,
            'ml_evaluations': 0,
            'heuristic_evaluations': 0
        }
    
    def evaluate_and_correct_state(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1_candidate: Dict[str, Any],
        generator_function,
        generation_context: Dict[str, Any],
        fallback_generator_function = None
    ) -> Tuple[Dict[str, Any], EvaluationResult]:
        """
        Core Phase 3.4 method: Evaluate candidate state and correct if needed
        
        Args:
            sc_t: Previous conscious state
            sc_t_plus_1_candidate: Candidate next state to evaluate
            generator_function: Function to regenerate state (should accept temperature param)
            generation_context: Context for regeneration (previous state, memory, etc.)
            fallback_generator_function: Heuristic fallback generator
            
        Returns:
            Tuple of (final_validated_state, evaluation_result)
        """
        
        logger.info("=== Phase 3.4: Critical State Evaluation Started ===")
        logger.debug(f"Candidate state to evaluate: {json.dumps(sc_t_plus_1_candidate, indent=2, ensure_ascii=False)}")
        
        self.evaluation_stats['total_evaluations'] += 1
        
        current_candidate = sc_t_plus_1_candidate
        original_temperature = generation_context.get('temperature', 0.7)
        current_temperature = original_temperature
        
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"--- Evaluation Attempt {attempt}/{self.max_attempts} ---")
            
            # Evaluate current candidate
            evaluation_result = self._evaluate_transition(sc_t, current_candidate, attempt)
            
            logger.info(f"Evaluation verdict: {evaluation_result.verdict.value}")
            logger.info(f"Justification: {evaluation_result.justification}")
            
            # If coherent, accept the state
            if evaluation_result.verdict == CoherenceVerdict.COHERENT:
                logger.info("✓ State accepted as coherent")
                if attempt == 1:
                    self.evaluation_stats['coherent_first_attempt'] += 1
                else:
                    self.evaluation_stats['required_regeneration'] += 1
                
                return current_candidate, evaluation_result
            
            # If incoherent and we have attempts left, try to regenerate
            if attempt < self.max_attempts:
                logger.warning(f"State rejected ({evaluation_result.verdict.value}), attempting regeneration...")
                
                # Reduce temperature for more stable generation
                current_temperature = max(0.1, current_temperature - self.temperature_decay)
                generation_context['temperature'] = current_temperature
                
                logger.info(f"Regenerating with reduced temperature: {current_temperature:.2f}")
                
                try:
                    # Try to regenerate with the model
                    current_candidate = generator_function(**generation_context)
                    logger.debug(f"Regenerated candidate: {json.dumps(current_candidate, indent=2, ensure_ascii=False)}")
                    
                except Exception as e:
                    logger.error(f"Error during regeneration: {e}")
                    # If regeneration fails, continue to next attempt or fallback
                    if attempt == self.max_attempts:
                        break
                    continue
            else:
                logger.warning("Maximum attempts reached")
                break
        
        # All attempts failed - use fallback generator
        logger.warning("All ML generation attempts failed, using heuristic fallback")
        self.evaluation_stats['failed_all_attempts'] += 1
        
        if fallback_generator_function:
            try:
                logger.info("Attempting heuristic fallback generation...")
                fallback_state = fallback_generator_function(**generation_context)
                
                # Evaluate fallback state
                fallback_evaluation = self._evaluate_transition(sc_t, fallback_state, self.max_attempts + 1)
                
                logger.info(f"Fallback evaluation: {fallback_evaluation.verdict.value}")
                
                return fallback_state, fallback_evaluation
                
            except Exception as e:
                logger.error(f"Fallback generation failed: {e}")
        
        # Last resort: return the last candidate with its evaluation
        logger.error("All generation methods failed, returning last candidate")
        return current_candidate, evaluation_result
    
    def _evaluate_transition(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """
        Evaluate a single state transition using the configured strategy
        """
        
        start_time = time.time()
        
        # Choose evaluation method based on strategy
        if self.strategy == EvaluationStrategy.ML_FIRST and self.ml_available:
            result = self._evaluate_with_ml(sc_t, sc_t_plus_1, attempt_number)
        elif self.strategy == EvaluationStrategy.HEURISTIC_FIRST:
            result = self._evaluate_with_heuristic(sc_t, sc_t_plus_1, attempt_number)
        elif self.strategy == EvaluationStrategy.ML_ONLY and self.ml_available:
            result = self._evaluate_with_ml_only(sc_t, sc_t_plus_1, attempt_number)
        elif self.strategy == EvaluationStrategy.HEURISTIC_ONLY:
            result = self._evaluate_with_heuristic_only(sc_t, sc_t_plus_1, attempt_number)
        else:
            # Fallback to heuristic if ML not available
            logger.warning("ML not available, falling back to heuristic evaluation")
            result = self._evaluate_with_heuristic(sc_t, sc_t_plus_1, attempt_number)
        
        evaluation_time = time.time() - start_time
        logger.debug(f"Evaluation completed in {evaluation_time:.3f}s using {result.evaluation_method}")
        
        return result
    
    def _evaluate_with_ml(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """Evaluate using ML classifier first"""
        
        try:
            self.evaluation_stats['ml_evaluations'] += 1
            
            analysis = self.ml_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
            
            return EvaluationResult(
                verdict=analysis.verdict,
                justification=analysis.justification,
                attempts_made=attempt_number,
                evaluation_method="ml_classifier",
                confidence_score=0.85,  # High confidence for ML
                metrics={
                    'goal_coherence': analysis.goal_coherence,
                    'emotion_coherence': analysis.emotion_coherence,
                    'thought_coherence': analysis.thought_coherence,
                    'memory_coherence': analysis.memory_coherence,
                    'confidence_change': abs(analysis.confidence_change)
                }
            )
            
        except Exception as e:
            logger.warning(f"ML evaluation failed: {e}, falling back to heuristic")
            return self._evaluate_with_heuristic_only(sc_t, sc_t_plus_1, attempt_number)
    
    def _evaluate_with_heuristic(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """Evaluate using heuristic method"""
        
        self.evaluation_stats['heuristic_evaluations'] += 1
        
        analysis = self.heuristic_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
        
        return EvaluationResult(
            verdict=analysis.verdict,
            justification=analysis.justification,
            attempts_made=attempt_number,
            evaluation_method="heuristic",
            confidence_score=0.7,  # Moderate confidence for heuristic
            metrics={
                'goal_coherence': analysis.goal_coherence,
                'emotion_coherence': analysis.emotion_coherence,
                'thought_coherence': analysis.thought_coherence,
                'memory_coherence': analysis.memory_coherence,
                'confidence_change': abs(analysis.confidence_change)
            }
        )
    
    def _evaluate_with_ml_only(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """Evaluate using only ML classifier"""
        return self._evaluate_with_ml(sc_t, sc_t_plus_1, attempt_number)
    
    def _evaluate_with_heuristic_only(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """Evaluate using only heuristic method"""
        return self._evaluate_with_heuristic(sc_t, sc_t_plus_1, attempt_number)
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics for monitoring"""
        
        if self.evaluation_stats['total_evaluations'] > 0:
            success_rate = (
                self.evaluation_stats['coherent_first_attempt'] / 
                self.evaluation_stats['total_evaluations']
            )
            regeneration_rate = (
                self.evaluation_stats['required_regeneration'] / 
                self.evaluation_stats['total_evaluations']
            )
            failure_rate = (
                self.evaluation_stats['failed_all_attempts'] / 
                self.evaluation_stats['total_evaluations']
            )
        else:
            success_rate = regeneration_rate = failure_rate = 0.0
        
        return {
            'total_evaluations': self.evaluation_stats['total_evaluations'],
            'success_rate_first_attempt': success_rate,
            'regeneration_rate': regeneration_rate,
            'failure_rate': failure_rate,
            'ml_evaluations': self.evaluation_stats['ml_evaluations'],
            'heuristic_evaluations': self.evaluation_stats['heuristic_evaluations'],
            'ml_available': self.ml_available,
            'strategy': self.strategy.value
        }
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'coherent_first_attempt': 0,
            'required_regeneration': 0,
            'failed_all_attempts': 0,
            'ml_evaluations': 0,
            'heuristic_evaluations': 0
        }
        logger.info("Evaluation statistics reset")


def create_critical_evaluator(
    max_attempts: int = 3,
    temperature_decay: float = 0.3,
    use_ml_classifier: bool = True,
    classifier_path: str = "./models/coherence_classifier"
) -> CriticalStateEvaluator:
    """
    Factory function to create a CriticalStateEvaluator with appropriate configuration
    """
    
    strategy = EvaluationStrategy.ML_FIRST if use_ml_classifier else EvaluationStrategy.HEURISTIC_ONLY
    
    return CriticalStateEvaluator(
        max_attempts=max_attempts,
        temperature_decay=temperature_decay,
        strategy=strategy,
        ml_classifier_path=classifier_path
    )