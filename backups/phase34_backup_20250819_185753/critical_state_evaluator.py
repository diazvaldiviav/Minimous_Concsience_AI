"""
Phase 3.4: Critical State Evaluation
===================================
Implements validation and correction loop for conscious state transitions.
Acts as a quality gate using the trained CoherenceClassifier.
"""

import json
import logging
import time
import os
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
        
        # Initialize evaluators with proper fallback logic
        logger.info("Initializing evaluators with fallback support...")
        
        # Try to initialize semantic evaluator
        try:
            from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import ModelBasedCoherenceEvaluator
            self.semantic_evaluator = ModelBasedCoherenceEvaluator(
                coherence_threshold=coherence_threshold,
                use_ml_classifier=False  # Start with semantic only
            )
            self.semantic_available = True
            logger.info("✅ Semantic evaluator initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Semantic evaluator unavailable: {e}")
            self.semantic_evaluator = None
            self.semantic_available = False
        
        # Try to initialize ML classifier
        try:
            if os.path.exists(ml_classifier_path):
                from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import ModelBasedCoherenceEvaluator
                self.ml_evaluator = ModelBasedCoherenceEvaluator(
                    coherence_threshold=coherence_threshold,
                    use_ml_classifier=True,
                    classifier_path=ml_classifier_path
                )
                self.ml_available = True
                logger.info("✅ ML classifier loaded successfully")
            else:
                logger.info("ℹ️ ML classifier path not found - using heuristic fallback")
                self.ml_evaluator = None
                self.ml_available = False
        except Exception as e:
            logger.warning(f"⚠️ ML classifier unavailable: {e}")
            self.ml_evaluator = None
            self.ml_available = False
        
        # FALLBACK: Heuristic evaluator (always available)
        self.heuristic_evaluator = CoherenceEvaluator()
        logger.info("✅ Heuristic evaluator initialized as fallback")
        
        # Log final configuration
        if self.semantic_available:
            logger.info("🎯 Using SEMANTIC evaluation as primary method")
        elif self.ml_available:
            logger.info("🎯 Using ML CLASSIFIER as primary method") 
        else:
            logger.info("🎯 Using HEURISTIC evaluation as primary method")
        
        # Statistics (updated to reflect new priority order)
        self.evaluation_stats = {
            'total_evaluations': 0,
            'coherent_first_attempt': 0,
            'required_regeneration': 0,
            'failed_all_attempts': 0,
            'semantic_evaluations': 0,      # Primary method
            'ml_evaluations': 0,            # Secondary method  
            'heuristic_evaluations': 0      # Fallback method
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
        Evaluate a single state transition using hierarchy:
        1. PRIMARY: Semantic evaluator (best accuracy when available)
        2. SECONDARY: ML classifier (for edge cases)  
        3. FALLBACK: Heuristic evaluator (always available)
        """
        
        start_time = time.time()
        
        # Try semantic evaluation first (highest accuracy)
        if self.semantic_available:
            try:
                logger.debug("Using semantic evaluation as primary method")
                result = self._evaluate_with_semantic(sc_t, sc_t_plus_1, attempt_number)
                evaluation_time = time.time() - start_time
                logger.debug(f"Semantic evaluation completed in {evaluation_time:.3f}s")
                return result
            except Exception as e:
                logger.warning(f"Semantic evaluation failed: {e}, falling back to ML classifier")
        
        # Try ML classifier as secondary
        if self.ml_available:
            try:
                logger.debug("Using ML classifier as secondary method")
                result = self._evaluate_with_ml(sc_t, sc_t_plus_1, attempt_number)
                evaluation_time = time.time() - start_time
                logger.debug(f"ML evaluation completed in {evaluation_time:.3f}s")
                return result
            except Exception as e:
                logger.warning(f"ML evaluation failed: {e}, falling back to heuristic")
        
        # FALLBACK: Use heuristic evaluation (always works)
        logger.debug("Using heuristic evaluation as fallback method")
        result = self._evaluate_with_heuristic_only(sc_t, sc_t_plus_1, attempt_number)
        
        evaluation_time = time.time() - start_time
        logger.debug(f"Heuristic evaluation completed in {evaluation_time:.3f}s")
        
        return result
    
    def _evaluate_with_semantic(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """Evaluate using PRIMARY semantic similarity approach"""
        
        self.evaluation_stats['semantic_evaluations'] += 1
        
        analysis = self.semantic_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
        
        return EvaluationResult(
            verdict=analysis.verdict,
            justification=f"Semantic analysis: {analysis.justification}",
            attempts_made=attempt_number,
            evaluation_method="semantic_similarity",
            confidence_score=0.9,  # High confidence for well-tested semantic approach
            metrics={
                'goal_coherence': analysis.goal_coherence,
                'emotion_coherence': analysis.emotion_coherence,
                'thought_coherence': analysis.thought_coherence,
                'memory_coherence': analysis.memory_coherence,
                'confidence_change': abs(analysis.confidence_change)
            }
        )
    
    def _evaluate_with_ml(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any],
        attempt_number: int
    ) -> EvaluationResult:
        """Evaluate using SECONDARY ML classifier for edge cases"""
        
        self.evaluation_stats['ml_evaluations'] += 1
        
        analysis = self.ml_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
        
        return EvaluationResult(
            verdict=analysis.verdict,
            justification=f"ML classifier (secondary): {analysis.justification}",
            attempts_made=attempt_number,
            evaluation_method="ml_classifier_secondary",
            confidence_score=0.75,  # Lower confidence as it's secondary method
            metrics={
                'goal_coherence': analysis.goal_coherence,
                'emotion_coherence': analysis.emotion_coherence,
                'thought_coherence': analysis.thought_coherence,
                'memory_coherence': analysis.memory_coherence,
                'confidence_change': abs(analysis.confidence_change)
            }
        )
    
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
        """Evaluate using FALLBACK heuristic method"""
        
        self.evaluation_stats['heuristic_evaluations'] += 1
        
        analysis = self.heuristic_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
        
        return EvaluationResult(
            verdict=analysis.verdict,
            justification=f"Heuristic fallback: {analysis.justification}",
            attempts_made=attempt_number,
            evaluation_method="heuristic_fallback",
            confidence_score=0.6,  # Lower confidence as fallback method
            metrics={
                'goal_coherence': analysis.goal_coherence,
                'emotion_coherence': analysis.emotion_coherence,
                'thought_coherence': analysis.thought_coherence,
                'memory_coherence': analysis.memory_coherence,
                'confidence_change': abs(analysis.confidence_change)
            }
        )
    
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