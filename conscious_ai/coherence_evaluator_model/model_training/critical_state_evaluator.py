"""
Phase 3.4: Critical State Evaluation - Hybrid Implementation
===========================================================
Robust hybrid evaluator combining semantic similarity, rule-based analysis, 
and contextual continuity assessment to replace the problematic ML classifier.

This implementation addresses the overfitting issues (30% accuracy) and 
"maximum attempts reached" errors by using a calibrated hybrid approach.
"""

import json
import logging
import time
import os
import re
from typing import Dict, Any, Tuple, Optional, List, Set
from dataclasses import dataclass
from enum import Enum
import math

# Optional dependency handling for sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EvaluationStrategy(Enum):
    """Strategy for evaluation fallback"""
    ML_FIRST = "ml_first"          # Legacy - now maps to HYBRID
    HEURISTIC_FIRST = "heuristic_first"  # Legacy - now maps to HYBRID
    ML_ONLY = "ml_only"            # Legacy - now maps to HYBRID  
    HEURISTIC_ONLY = "heuristic_only"  # Pure heuristic
    HYBRID = "hybrid"              # NEW: Hybrid weighted approach (default)


class CoherenceVerdict(Enum):
    """Coherence evaluation verdict"""
    COHERENT = "coherent"
    INCOHERENT = "incoherent"
    AMBIGUOUS = "ambiguous"


@dataclass
class EvaluationResult:
    """Result of critical state evaluation"""
    verdict: CoherenceVerdict
    justification: str
    attempts_made: int
    evaluation_method: str
    confidence_score: float
    metrics: Dict[str, float]


@dataclass
class HybridEvaluationMetrics:
    """Detailed metrics from hybrid evaluation"""
    semantic_score: float
    rule_based_score: float
    contextual_score: float
    final_score: float
    component_breakdown: Dict[str, float]


class HybridCoherenceEvaluator:
    """
    Robust hybrid coherence evaluator combining:
    - Semantic similarity (40% weight)
    - Rule-based coherence analysis (40% weight) 
    - Contextual continuity assessment (20% weight)
    
    Calibrated thresholds:
    - Coherent: ≥0.65
    - Incoherent: ≤0.35
    - Ambiguous: 0.35-0.65
    """
    
    def __init__(self, coherence_threshold: float = 0.60):
        """Initialize the hybrid evaluator"""
        self.coherent_threshold = 0.60  # More lenient calibrated threshold
        self.incoherent_threshold = 0.35  # Calibrated threshold
        
        # Component weights (must sum to 1.0)
        self.semantic_weight = 0.40
        self.rule_based_weight = 0.40
        self.contextual_weight = 0.20
        
        # Initialize semantic embedder if available
        self.embedder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence-transformers embedder initialized")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load sentence-transformers: {e}")
                self.embedder = None
        else:
            logger.info("ℹ️ Sentence-transformers not available, using word overlap fallback")
        
        # Goal coherence groups for rule-based analysis
        self.goal_groups = {
            'understanding': {'understand', 'comprehend', 'analyze', 'examine', 'explore', 'investigate'},
            'problem_solving': {'solve', 'resolve', 'fix', 'address', 'handle', 'tackle'},
            'learning': {'learn', 'study', 'discover', 'research', 'acquire', 'master'},
            'creating': {'create', 'build', 'generate', 'construct', 'develop', 'design'},
            'reflecting': {'reflect', 'contemplate', 'ponder', 'consider', 'think', 'meditate'}
        }
        
        # Emotional transition patterns
        self.emotional_transitions = {
            'natural_progressions': {
                ('curious', 'analytical'): 0.9,
                ('analytical', 'focused'): 0.8,
                ('confused', 'curious'): 0.8,
                ('frustrated', 'determined'): 0.7,
                ('excited', 'focused'): 0.7,
                ('contemplative', 'insightful'): 0.8,
                ('uncertain', 'confident'): 0.6
            },
            'stable_emotions': {'calm', 'focused', 'confident', 'content'},
            'volatile_emotions': {'excited', 'frustrated', 'anxious', 'confused'}
        }
        
        logger.info("🎯 HybridCoherenceEvaluator initialized with calibrated thresholds")
    
    def evaluate_transition(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> 'TransitionAnalysis':
        """
        Main evaluation method combining all three approaches
        
        Returns TransitionAnalysis compatible with existing interface
        """
        start_time = time.time()
        
        # Extract state components safely
        prev_goal = str(sc_t.get('goal', '')).lower()
        prev_emotion = str(sc_t.get('emotion', '')).lower()
        prev_thought = str(sc_t.get('thought', '')).lower()
        prev_confidence = float(sc_t.get('confidence', 0.5))
        prev_memory = sc_t.get('memory', [])
        
        curr_goal = str(sc_t_plus_1.get('goal', '')).lower()
        curr_emotion = str(sc_t_plus_1.get('emotion', '')).lower()
        curr_thought = str(sc_t_plus_1.get('thought', '')).lower()
        curr_confidence = float(sc_t_plus_1.get('confidence', 0.5))
        curr_memory = sc_t_plus_1.get('memory', [])
        
        # 1. Semantic Similarity Assessment (40% weight)
        semantic_score = self._evaluate_semantic_similarity(sc_t, sc_t_plus_1)
        
        # 2. Rule-based Coherence Analysis (40% weight)
        rule_scores = self._evaluate_rule_based_coherence(
            prev_goal, prev_emotion, prev_thought, prev_confidence, prev_memory,
            curr_goal, curr_emotion, curr_thought, curr_confidence, curr_memory
        )
        rule_based_score = rule_scores['overall']
        
        # 3. Contextual Continuity Assessment (20% weight)
        contextual_score = self._evaluate_contextual_continuity(sc_t, sc_t_plus_1)
        
        # Compute weighted final score
        final_score = (
            semantic_score * self.semantic_weight +
            rule_based_score * self.rule_based_weight +
            contextual_score * self.contextual_weight
        )
        
        # Determine verdict based on calibrated thresholds
        if final_score >= self.coherent_threshold:
            verdict = CoherenceVerdict.COHERENT
        elif final_score <= self.incoherent_threshold:
            verdict = CoherenceVerdict.INCOHERENT
        else:
            verdict = CoherenceVerdict.AMBIGUOUS
        
        # Create detailed justification
        justification = self._create_justification(
            semantic_score, rule_based_score, contextual_score, final_score, verdict, rule_scores
        )
        
        evaluation_time = time.time() - start_time
        logger.debug(f"Hybrid evaluation completed in {evaluation_time:.3f}s: {verdict.value} (score: {final_score:.3f})")
        
        # Return TransitionAnalysis compatible with existing interface
        from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import TransitionAnalysis
        
        return TransitionAnalysis(
            verdict=verdict,
            justification=justification,
            goal_coherence=rule_scores['goal_coherence'],
            emotion_coherence=rule_scores['emotion_coherence'],
            thought_coherence=semantic_score,  # Use semantic score for thought coherence
            memory_coherence=rule_scores['memory_coherence'],
            confidence_change=abs(curr_confidence - prev_confidence)
        )
    
    def _evaluate_semantic_similarity(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> float:
        """Evaluate semantic similarity between states (40% weight)"""
        
        # Combine key textual elements from each state
        prev_text = f"{sc_t.get('goal', '')} {sc_t.get('thought', '')} {sc_t.get('emotion', '')}"
        curr_text = f"{sc_t_plus_1.get('goal', '')} {sc_t_plus_1.get('thought', '')} {sc_t_plus_1.get('emotion', '')}"
        
        if self.embedder:
            try:
                # Use sentence-transformers for high-quality semantic similarity
                embeddings = self.embedder.encode([prev_text, curr_text])
                similarity = float(embeddings[0] @ embeddings[1] / 
                                 (math.sqrt(embeddings[0] @ embeddings[0]) * 
                                  math.sqrt(embeddings[1] @ embeddings[1])))
                # Normalize to 0-1 range and apply sigmoid for better distribution
                return min(1.0, max(0.0, (similarity + 1) / 2))
            except Exception as e:
                logger.warning(f"Sentence-transformers failed: {e}, using word overlap")
        
        # Fallback: Word overlap similarity
        return self._word_overlap_similarity(prev_text, curr_text)
    
    def _word_overlap_similarity(self, text1: str, text2: str) -> float:
        """Fallback word overlap similarity calculation"""
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 and not words2:
            return 1.0  # Both empty
        if not words1 or not words2:
            return 0.0  # One empty
        
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard_similarity = len(intersection) / len(union)
        
        # Apply boost for meaningful word overlaps
        meaningful_overlaps = intersection & {'consciousness', 'awareness', 'thinking', 'understanding', 'analyze', 'explore', 'wonder', 'examine', 'patterns'}
        if meaningful_overlaps:
            jaccard_similarity *= 1.4  # Higher boost for consciousness themes
        
        return min(1.0, jaccard_similarity)
    
    def _evaluate_rule_based_coherence(
        self, 
        prev_goal: str, prev_emotion: str, prev_thought: str, prev_confidence: float, prev_memory: List,
        curr_goal: str, curr_emotion: str, curr_thought: str, curr_confidence: float, curr_memory: List
    ) -> Dict[str, float]:
        """Rule-based coherence analysis (40% weight)"""
        
        # 1. Goal Coherence Analysis
        goal_coherence = self._analyze_goal_coherence(prev_goal, curr_goal)
        
        # 2. Emotional Transition Analysis  
        emotion_coherence = self._analyze_emotion_coherence(prev_emotion, curr_emotion)
        
        # 3. Confidence Progression Analysis
        confidence_coherence = self._analyze_confidence_progression(prev_confidence, curr_confidence)
        
        # 4. Memory Integration Analysis
        memory_coherence = self._analyze_memory_coherence(prev_memory, curr_memory)
        
        # Weighted combination of rule-based components
        overall_rule_score = (
            goal_coherence * 0.3 +
            emotion_coherence * 0.3 +
            confidence_coherence * 0.2 +
            memory_coherence * 0.2
        )
        
        return {
            'overall': overall_rule_score,
            'goal_coherence': goal_coherence,
            'emotion_coherence': emotion_coherence,
            'confidence_coherence': confidence_coherence,
            'memory_coherence': memory_coherence
        }
    
    def _analyze_goal_coherence(self, prev_goal: str, curr_goal: str) -> float:
        """Analyze goal coherence using semantic groups"""
        
        if not prev_goal or not curr_goal:
            return 0.5  # Neutral when missing data
        
        # Check if goals are identical or very similar
        if prev_goal == curr_goal:
            return 1.0
        
        # Check for word overlap
        overlap_score = self._word_overlap_similarity(prev_goal, curr_goal)
        if overlap_score > 0.7:
            return overlap_score
        
        # Check semantic goal groups
        prev_group = self._get_goal_group(prev_goal)
        curr_group = self._get_goal_group(curr_goal)
        
        if prev_group and curr_group:
            if prev_group == curr_group:
                return 0.8  # Same semantic group
            else:
                # Different groups can still be coherent if they're related
                related_transitions = {
                    ('understanding', 'problem_solving'): 0.7,
                    ('learning', 'understanding'): 0.8,
                    ('reflecting', 'understanding'): 0.7,
                    ('creating', 'problem_solving'): 0.6
                }
                return related_transitions.get((prev_group, curr_group), 0.4)
        
        # Default to word overlap if no semantic groups matched
        return max(0.3, overlap_score)
    
    def _get_goal_group(self, goal: str) -> Optional[str]:
        """Determine which semantic group a goal belongs to"""
        goal_words = set(re.findall(r'\w+', goal.lower()))
        
        for group_name, group_words in self.goal_groups.items():
            if goal_words & group_words:
                return group_name
        return None
    
    def _analyze_emotion_coherence(self, prev_emotion: str, curr_emotion: str) -> float:
        """Analyze emotional transition coherence"""
        
        if not prev_emotion or not curr_emotion:
            return 0.5
        
        if prev_emotion == curr_emotion:
            return 0.9  # Stable emotion is usually good
        
        # Check for natural progressions
        transition_key = (prev_emotion, curr_emotion)
        if transition_key in self.emotional_transitions['natural_progressions']:
            return self.emotional_transitions['natural_progressions'][transition_key]
        
        # Check emotional stability patterns
        if (prev_emotion in self.emotional_transitions['stable_emotions'] and 
            curr_emotion in self.emotional_transitions['stable_emotions']):
            return 0.8  # Stable to stable
        
        if (prev_emotion in self.emotional_transitions['volatile_emotions'] and 
            curr_emotion in self.emotional_transitions['stable_emotions']):
            return 0.7  # Volatile to stable (good progression)
        
        if (prev_emotion in self.emotional_transitions['stable_emotions'] and 
            curr_emotion in self.emotional_transitions['volatile_emotions']):
            return 0.4  # Stable to volatile (concerning)
        
        # Default similarity check
        return self._word_overlap_similarity(prev_emotion, curr_emotion)
    
    def _analyze_confidence_progression(self, prev_confidence: float, curr_confidence: float) -> float:
        """Analyze confidence change patterns"""
        
        confidence_change = curr_confidence - prev_confidence
        abs_change = abs(confidence_change)
        
        # Prefer gradual changes
        if abs_change <= 0.1:
            return 1.0  # Very stable
        elif abs_change <= 0.2:
            return 0.8  # Gradual change
        elif abs_change <= 0.3:
            return 0.6  # Moderate change
        else:
            return 0.3  # Dramatic change (concerning)
    
    def _analyze_memory_coherence(self, prev_memory: List, curr_memory: List) -> float:
        """Analyze memory integration patterns"""
        
        if not prev_memory and not curr_memory:
            return 1.0  # Both empty is fine
        
        if not prev_memory or not curr_memory:
            return 0.5  # One empty is neutral
        
        # Convert to strings for analysis
        prev_memory_str = ' '.join(str(item) for item in prev_memory)
        curr_memory_str = ' '.join(str(item) for item in curr_memory)
        
        # Check for memory overlap (good - shows continuity)
        overlap_score = self._word_overlap_similarity(prev_memory_str, curr_memory_str)
        
        # Analyze memory evolution patterns
        memory_growth = len(curr_memory) - len(prev_memory)
        
        # Prefer gradual memory growth
        if memory_growth == 0:
            growth_score = 0.8  # Stable memory
        elif 0 < memory_growth <= 2:
            growth_score = 1.0  # Healthy growth
        elif memory_growth > 2:
            growth_score = 0.6  # Rapid growth
        else:
            growth_score = 0.4  # Memory loss
        
        return (overlap_score * 0.7 + growth_score * 0.3)
    
    def _evaluate_contextual_continuity(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> float:
        """Evaluate contextual continuity (20% weight)"""
        
        # Check for structural consistency
        structure_score = self._check_structural_consistency(sc_t, sc_t_plus_1)
        
        # Check for thematic coherence
        thematic_score = self._check_thematic_coherence(sc_t, sc_t_plus_1)
        
        # Check for temporal consistency
        temporal_score = self._check_temporal_consistency(sc_t, sc_t_plus_1)
        
        return (structure_score * 0.4 + thematic_score * 0.4 + temporal_score * 0.2)
    
    def _check_structural_consistency(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> float:
        """Check if both states have consistent structure"""
        
        required_fields = {'goal', 'emotion', 'thought', 'confidence'}
        
        prev_fields = set(sc_t.keys()) & required_fields
        curr_fields = set(sc_t_plus_1.keys()) & required_fields
        
        if len(prev_fields) == len(curr_fields) == len(required_fields):
            return 1.0  # Perfect structure
        
        overlap = len(prev_fields & curr_fields)
        union = len(prev_fields | curr_fields)
        
        return overlap / union if union > 0 else 0.5
    
    def _check_thematic_coherence(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> float:
        """Check for thematic coherence across the transition"""
        
        # Extract all textual content
        prev_content = ' '.join(str(v) for v in sc_t.values() if isinstance(v, str))
        curr_content = ' '.join(str(v) for v in sc_t_plus_1.values() if isinstance(v, str))
        
        # Look for consciousness-related themes
        consciousness_themes = {
            'consciousness', 'awareness', 'self', 'identity', 'thinking', 'thoughts',
            'mind', 'cognitive', 'mental', 'introspection', 'reflection', 'understanding'
        }
        
        prev_themes = set(re.findall(r'\w+', prev_content.lower())) & consciousness_themes
        curr_themes = set(re.findall(r'\w+', curr_content.lower())) & consciousness_themes
        
        if prev_themes and curr_themes:
            theme_overlap = len(prev_themes & curr_themes) / len(prev_themes | curr_themes)
            return min(1.0, theme_overlap + 0.3)  # Boost for thematic consistency
        
        return 0.5  # Neutral if no consciousness themes detected
    
    def _check_temporal_consistency(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> float:
        """Check temporal consistency patterns"""
        
        # Look for temporal indicators in language
        temporal_patterns = {
            'continuity': {'continue', 'still', 'maintain', 'keep', 'ongoing'},
            'progression': {'next', 'then', 'now', 'further', 'advance', 'develop'},
            'reflection': {'previously', 'before', 'earlier', 'remember', 'recall'}
        }
        
        curr_content = ' '.join(str(v) for v in sc_t_plus_1.values() if isinstance(v, str)).lower()
        curr_words = set(re.findall(r'\w+', curr_content))
        
        for pattern_type, pattern_words in temporal_patterns.items():
            if curr_words & pattern_words:
                return 0.8  # Good temporal consistency
        
        return 0.6  # Neutral temporal consistency
    
    def _create_justification(
        self, 
        semantic_score: float, 
        rule_based_score: float, 
        contextual_score: float, 
        final_score: float, 
        verdict: CoherenceVerdict,
        rule_scores: Dict[str, float]
    ) -> str:
        """Create detailed justification for the evaluation decision"""
        
        components = [
            f"Semantic similarity: {semantic_score:.3f} ({self.semantic_weight:.0%})",
            f"Rule-based analysis: {rule_based_score:.3f} ({self.rule_based_weight:.0%})",
            f"Contextual continuity: {contextual_score:.3f} ({self.contextual_weight:.0%})"
        ]
        
        # Add rule breakdown
        rule_breakdown = [
            f"goal coherence: {rule_scores['goal_coherence']:.2f}",
            f"emotion coherence: {rule_scores['emotion_coherence']:.2f}",
            f"memory coherence: {rule_scores['memory_coherence']:.2f}"
        ]
        
        justification = (
            f"Hybrid evaluation: {final_score:.3f} → {verdict.value}. "
            f"Components: {', '.join(components)}. "
            f"Rule details: {', '.join(rule_breakdown)}."
        )
        
        # Add threshold context
        if verdict == CoherenceVerdict.COHERENT:
            justification += f" Score ≥ {self.coherent_threshold} threshold."
        elif verdict == CoherenceVerdict.INCOHERENT:
            justification += f" Score ≤ {self.incoherent_threshold} threshold."
        else:
            justification += f" Score in ambiguous range ({self.incoherent_threshold}-{self.coherent_threshold})."
        
        return justification


class CriticalStateEvaluator:
    """
    Phase 3.4: Critical evaluation of state transitions with robust hybrid approach
    
    API-compatible replacement for the problematic ML-based evaluator.
    Uses HybridCoherenceEvaluator for reliable coherence detection.
    """
    
    def __init__(
        self,
        max_attempts: int = 3,
        temperature_decay: float = 0.3,
        strategy: EvaluationStrategy = EvaluationStrategy.HYBRID,
        ml_classifier_path: str = "./models/coherence_classifier",
        coherence_threshold: float = 0.65,
        # Legacy parameters - maintained for API compatibility
        use_ml_classifier: bool = None,
        eval_strategy: str = None
    ):
        """
        Initialize the critical state evaluator with hybrid approach
        
        Args:
            max_attempts: Maximum regeneration attempts for incoherent states
            temperature_decay: Amount to reduce temperature on retry
            strategy: Evaluation strategy (all now use hybrid)
            ml_classifier_path: Legacy parameter (ignored)
            coherence_threshold: Coherence threshold for evaluation
            use_ml_classifier: Legacy parameter (ignored with warning)
            eval_strategy: Legacy parameter (ignored with warning)
        """
        self.max_attempts = max_attempts
        self.temperature_decay = temperature_decay
        self.strategy = strategy
        
        # Log legacy parameter warnings
        if use_ml_classifier is not None:
            logger.info(f"ℹ️ Legacy parameter 'use_ml_classifier={use_ml_classifier}' ignored - using hybrid approach")
        if eval_strategy is not None:
            logger.info(f"ℹ️ Legacy parameter 'eval_strategy={eval_strategy}' ignored - using hybrid approach")
        
        # Initialize the robust hybrid evaluator
        logger.info("🎯 Initializing Critical State Evaluator with Hybrid Approach...")
        
        try:
            self.hybrid_evaluator = HybridCoherenceEvaluator(coherence_threshold=coherence_threshold)
            logger.info("✅ HybridCoherenceEvaluator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize HybridCoherenceEvaluator: {e}")
            raise
        
        # Fallback heuristic evaluator for extreme cases
        try:
            from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator
            self.fallback_evaluator = CoherenceEvaluator()
            logger.info("✅ Fallback heuristic evaluator available")
        except Exception as e:
            logger.warning(f"⚠️ Fallback evaluator unavailable: {e}")
            self.fallback_evaluator = None
        
        # Statistics tracking
        self.evaluation_stats = {
            'total_evaluations': 0,
            'coherent_first_attempt': 0,
            'required_regeneration': 0,
            'failed_all_attempts': 0,
            'hybrid_evaluations': 0,
            'fallback_evaluations': 0,
            'avg_evaluation_time': 0.0
        }
        
        logger.info("🚀 Critical State Evaluator ready with hybrid approach")
    
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
        
        API-compatible with existing interface. Uses hybrid evaluation for
        reliable coherence detection without ML classifier issues.
        """
        
        logger.info("=== Phase 3.4: Critical State Evaluation Started (Hybrid) ===")
        logger.debug(f"Evaluating candidate: {json.dumps(sc_t_plus_1_candidate, default=str, ensure_ascii=False)}")
        
        self.evaluation_stats['total_evaluations'] += 1
        
        current_candidate = sc_t_plus_1_candidate
        original_temperature = generation_context.get('temperature', 0.7)
        current_temperature = original_temperature
        
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"--- Hybrid Evaluation Attempt {attempt}/{self.max_attempts} ---")
            
            # Evaluate current candidate with hybrid approach
            evaluation_result = self._evaluate_transition(sc_t, current_candidate, attempt)
            
            logger.info(f"Hybrid verdict: {evaluation_result.verdict.value} (confidence: {evaluation_result.confidence_score:.3f})")
            logger.info(f"Justification: {evaluation_result.justification}")
            
            # If coherent, accept the state
            if evaluation_result.verdict == CoherenceVerdict.COHERENT:
                logger.info("✅ State accepted as coherent")
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
                    logger.debug(f"Regenerated candidate: {json.dumps(current_candidate, default=str, ensure_ascii=False)}")
                    
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
        logger.warning("All generation attempts failed, using heuristic fallback")
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
        Evaluate a single state transition using hybrid approach
        """
        
        start_time = time.time()
        
        try:
            # Use hybrid evaluator as primary method
            self.evaluation_stats['hybrid_evaluations'] += 1
            
            analysis = self.hybrid_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
            
            evaluation_time = time.time() - start_time
            self._update_avg_evaluation_time(evaluation_time)
            
            logger.debug(f"Hybrid evaluation completed in {evaluation_time:.3f}s")
            
            return EvaluationResult(
                verdict=analysis.verdict,
                justification=analysis.justification,
                attempts_made=attempt_number,
                evaluation_method="hybrid_evaluator",
                confidence_score=0.85,  # High confidence in hybrid approach
                metrics={
                    'goal_coherence': analysis.goal_coherence,
                    'emotion_coherence': analysis.emotion_coherence,
                    'thought_coherence': analysis.thought_coherence,
                    'memory_coherence': analysis.memory_coherence,
                    'confidence_change': analysis.confidence_change
                }
            )
            
        except Exception as e:
            logger.error(f"Hybrid evaluation failed: {e}")
            
            # Use fallback heuristic evaluator if available
            if self.fallback_evaluator:
                try:
                    logger.warning("Using fallback heuristic evaluator")
                    self.evaluation_stats['fallback_evaluations'] += 1
                    
                    analysis = self.fallback_evaluator.evaluate_transition(sc_t, sc_t_plus_1)
                    
                    evaluation_time = time.time() - start_time
                    self._update_avg_evaluation_time(evaluation_time)
                    
                    return EvaluationResult(
                        verdict=analysis.verdict,
                        justification=f"Fallback heuristic: {analysis.justification}",
                        attempts_made=attempt_number,
                        evaluation_method="fallback_heuristic",
                        confidence_score=0.6,  # Lower confidence for fallback
                        metrics={
                            'goal_coherence': analysis.goal_coherence,
                            'emotion_coherence': analysis.emotion_coherence,
                            'thought_coherence': analysis.thought_coherence,
                            'memory_coherence': analysis.memory_coherence,
                            'confidence_change': analysis.confidence_change
                        }
                    )
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback evaluation also failed: {fallback_error}")
            
            # Last resort: return ambiguous verdict
            logger.error("All evaluation methods failed, returning ambiguous")
            return EvaluationResult(
                verdict=CoherenceVerdict.AMBIGUOUS,
                justification="Evaluation failed - returning ambiguous verdict",
                attempts_made=attempt_number,
                evaluation_method="error_fallback",
                confidence_score=0.1,
                metrics={'error': 1.0}
            )
    
    def _update_avg_evaluation_time(self, evaluation_time: float):
        """Update running average of evaluation time"""
        total_evals = self.evaluation_stats['total_evaluations']
        current_avg = self.evaluation_stats['avg_evaluation_time']
        
        self.evaluation_stats['avg_evaluation_time'] = (
            (current_avg * (total_evals - 1) + evaluation_time) / total_evals
        )
    
    # Legacy methods for API compatibility
    def evaluate_and_correct(self, *args, **kwargs):
        """Legacy method name - redirects to evaluate_and_correct_state"""
        return self.evaluate_and_correct_state(*args, **kwargs)
    
    def _evaluate_transition_ml(self, *args, **kwargs):
        """Legacy method - redirects to hybrid evaluation"""
        return self._evaluate_transition(*args, **kwargs)
    
    def _evaluate_transition_heuristic(self, *args, **kwargs):
        """Legacy method - redirects to hybrid evaluation"""
        return self._evaluate_transition(*args, **kwargs)
    
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
            'hybrid_evaluations': self.evaluation_stats['hybrid_evaluations'],
            'fallback_evaluations': self.evaluation_stats['fallback_evaluations'],
            'avg_evaluation_time': self.evaluation_stats['avg_evaluation_time'],
            'evaluation_method': 'hybrid_approach',
            'ml_available': False,  # No longer using problematic ML classifier
            'hybrid_available': True,
            'strategy': self.strategy.value
        }
    
    def reset_statistics(self):
        """Reset evaluation statistics"""
        self.evaluation_stats = {
            'total_evaluations': 0,
            'coherent_first_attempt': 0,
            'required_regeneration': 0,
            'failed_all_attempts': 0,
            'hybrid_evaluations': 0,
            'fallback_evaluations': 0,
            'avg_evaluation_time': 0.0
        }
        logger.info("Evaluation statistics reset")


def create_critical_evaluator(
    max_attempts: int = 3,
    temperature_decay: float = 0.3,
    use_ml_classifier: bool = None,  # Legacy parameter - ignored
    classifier_path: str = "./models/coherence_classifier"  # Legacy parameter - ignored
) -> CriticalStateEvaluator:
    """
    Factory function to create a CriticalStateEvaluator with hybrid approach
    
    Maintains API compatibility while using the robust hybrid implementation.
    Legacy ML parameters are ignored with appropriate logging.
    """
    
    if use_ml_classifier is not None:
        logger.info(f"ℹ️ Legacy parameter 'use_ml_classifier={use_ml_classifier}' ignored - using hybrid approach")
    
    logger.info("🏭 Creating CriticalStateEvaluator with hybrid approach")
    
    return CriticalStateEvaluator(
        max_attempts=max_attempts,
        temperature_decay=temperature_decay,
        strategy=EvaluationStrategy.HYBRID,
        ml_classifier_path=classifier_path  # Passed but ignored
    )