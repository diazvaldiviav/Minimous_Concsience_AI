"""
Truth Model v1 for SC Memory System.

This module validates key facts extracted from conversations to ensure they are
truthful and not hallucinations before using them for LoRA adapter training.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from ..core.config import Settings, get_settings
from ..core.exceptions import (
    TruthValidationError,
    ModelLoadError,
)
from ..core.models import KeyFact, ValidatedFact
from .embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)


class TruthModel:
    """
    Truth validation model for conversation facts.
    
    Uses simple NLI and consistency checks to validate that key facts
    extracted from conversations are truthful and consistent with the
    conversation context.
    
    Features:
    - Fact-context consistency checking
    - Configurable confidence thresholds
    - Basic Natural Language Inference
    - Hallucination detection
    """
    
    def __init__(
        self,
        embeddings_manager: Optional[EmbeddingsManager] = None,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the truth model.
        
        Args:
            embeddings_manager: Embeddings manager for semantic similarity
            settings: Application settings
        """
        self._settings = settings or get_settings()
        self._embeddings = embeddings_manager
        
        # Configuration
        self._confidence_threshold = self._settings.truth_model.confidence_threshold
        self._enable_nli = self._settings.truth_model.enable_nli_validation
        self._max_facts = self._settings.truth_model.max_facts_per_conversation
        
        # Simple ML components for basic NLI
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._nli_classifier: Optional[LogisticRegression] = None
        
        # State tracking
        self._is_initialized = False
        
        logger.info(
            f"Initialized TruthModel (threshold={self._confidence_threshold}, "
            f"nli_enabled={self._enable_nli})"
        )
    
    async def load_nli_model(self) -> None:
        """
        Load simple NLI model for basic fact checking.
        
        For MVP, this creates a simple classifier. In production,
        would load a proper pre-trained NLI model.
        """
        if self._is_initialized:
            logger.info("NLI model already loaded")
            return
        
        try:
            # Initialize embeddings if needed (with graceful fallback)
            if self._embeddings is None:
                try:
                    self._embeddings = EmbeddingsManager(settings=self._settings)
                    await self._embeddings.load_model()
                except Exception as embedding_error:
                    logger.warning(f"Embeddings unavailable: {embedding_error}")
                    logger.info("Truth validation will use fallback methods only")
                    self._embeddings = None
            
            # Initialize simple TF-IDF vectorizer and classifier for MVP
            self._vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            self._nli_classifier = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
            
            # Train with dummy data for MVP
            await self._train_dummy_nli_model()
            
            self._is_initialized = True
            logger.info("Simple NLI model loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load NLI model: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(
                message=error_msg,
                model_name="truth_model_nli",
                model_type="truth_validation"
            )
    
    async def _train_dummy_nli_model(self) -> None:
        """
        Train simple NLI classifier with dummy data.
        
        In production, this would load a proper pre-trained model.
        """
        # Generate simple training data for MVP
        entailment_pairs = [
            ("The user likes Python", "Python is the user's preferred language"),
            ("We discussed React development", "React was mentioned in the conversation"),
            ("The project requires authentication", "Authentication is needed for the project"),
            ("Database should use PostgreSQL", "PostgreSQL was chosen for the database"),
            ("User prefers dark mode", "Dark mode is the user's preference"),
        ]
        
        contradiction_pairs = [
            ("The user likes Python", "The user dislikes Python"),
            ("We discussed React", "We never talked about React"),
            ("Project needs authentication", "No authentication is required"),
            ("Use PostgreSQL database", "Use MongoDB database instead"),
            ("User prefers dark mode", "User prefers light mode"),
        ]
        
        # Create feature vectors
        premises = []
        hypotheses = []
        labels = []
        
        # Add entailment examples (label = 1)
        for premise, hypothesis in entailment_pairs:
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(1)
        
        # Add contradiction examples (label = 0)
        for premise, hypothesis in contradiction_pairs:
            premises.append(premise)
            hypotheses.append(hypothesis)
            labels.append(0)
        
        # Create combined features
        combined_texts = [f"{p} [SEP] {h}" for p, h in zip(premises, hypotheses)]
        
        # Fit vectorizer and train classifier
        X = self._vectorizer.fit_transform(combined_texts)
        self._nli_classifier.fit(X, labels)
        
        logger.debug("Trained simple NLI classifier with dummy data")
    
    async def validate_key_facts(
        self,
        key_facts: List[KeyFact],
        conversation_context: str
    ) -> List[ValidatedFact]:
        """
        Validate key facts against conversation context.
        
        Args:
            key_facts: List of facts to validate
            conversation_context: Full conversation context
            
        Returns:
            List of validated facts
        """
        if not self._is_initialized:
            await self.load_nli_model()
        
        if len(key_facts) > self._max_facts:
            logger.warning(
                f"Limiting validation to {self._max_facts} facts "
                f"(received {len(key_facts)})"
            )
            key_facts = key_facts[:self._max_facts]
        
        validated_facts = []
        
        for fact in key_facts:
            try:
                confidence = await self._check_fact_consistency(fact, conversation_context)
                is_valid = self._apply_truth_threshold(confidence)
                
                reason = self._get_validation_reason(confidence, is_valid)
                
                validated_fact = ValidatedFact(
                    original_fact=fact,
                    truth_score=confidence,
                    is_validated=is_valid,
                    validation_reason=reason
                )
                
                validated_facts.append(validated_fact)
                
                logger.debug(
                    f"Validated fact: '{fact.claim[:50]}...' "
                    f"(score={confidence:.3f}, valid={is_valid})"
                )
                
            except Exception as e:
                logger.error(f"Failed to validate fact '{fact.claim[:50]}...': {e}")
                # Include failed validation with low score
                validated_facts.append(ValidatedFact(
                    original_fact=fact,
                    truth_score=0.0,
                    is_validated=False,
                    validation_reason=f"Validation error: {str(e)}"
                ))
        
        valid_count = sum(1 for vf in validated_facts if vf.is_validated)
        logger.info(
            f"Validated {valid_count}/{len(validated_facts)} facts "
            f"(threshold={self._confidence_threshold})"
        )
        
        return validated_facts
    
    async def _check_fact_consistency(
        self,
        fact: KeyFact,
        context: str
    ) -> float:
        """
        Check consistency of fact with conversation context.
        
        Args:
            fact: Key fact to validate
            context: Conversation context
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Start with base confidence from fact extraction
        confidence = fact.confidence
        
        # Check semantic similarity
        semantic_score = await self._calculate_semantic_similarity(fact.claim, context)
        
        # Check for explicit contradictions
        contradiction_penalty = self._check_contradictions(fact.claim, context)
        
        # Apply NLI if enabled
        nli_score = 0.5  # neutral default
        if self._enable_nli and self._nli_classifier is not None:
            nli_score = self._calculate_nli_score(context, fact.claim)
        
        # Combine scores
        combined_confidence = (
            confidence * 0.3 +           # Original extraction confidence
            semantic_score * 0.3 +       # Semantic similarity
            nli_score * 0.3 +             # NLI entailment
            (1.0 - contradiction_penalty) * 0.1  # Contradiction check
        )
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, combined_confidence))
    
    async def _calculate_semantic_similarity(
        self,
        fact_claim: str,
        context: str
    ) -> float:
        """
        Calculate semantic similarity between fact and context.
        
        Args:
            fact_claim: The fact claim
            context: Conversation context
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Use embeddings for semantic similarity
            if self._embeddings is not None and self._embeddings.is_loaded():
                fact_embedding = await self._embeddings.encode_text(fact_claim)
                context_embedding = await self._embeddings.encode_text(context[:1000])  # Limit context
                
                # Calculate cosine similarity
                similarity = np.dot(fact_embedding, context_embedding) / (
                    np.linalg.norm(fact_embedding) * np.linalg.norm(context_embedding)
                )
                
                # Normalize to 0-1 range
                return (similarity + 1) / 2
            
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
        
        # Fallback to simple text overlap
        return self._calculate_text_overlap(fact_claim, context)
    
    def _calculate_text_overlap(self, fact_claim: str, context: str) -> float:
        """
        Calculate simple text overlap as fallback similarity measure.
        
        Args:
            fact_claim: The fact claim
            context: Conversation context
            
        Returns:
            Overlap score (0.0 to 1.0)
        """
        # Simple word overlap calculation
        fact_words = set(fact_claim.lower().split())
        context_words = set(context.lower().split())
        
        if len(fact_words) == 0:
            return 0.0
        
        overlap = len(fact_words.intersection(context_words))
        return overlap / len(fact_words)
    
    def _check_contradictions(self, fact_claim: str, context: str) -> float:
        """
        Check for explicit contradictions between fact and context.
        
        Args:
            fact_claim: The fact claim
            context: Conversation context
            
        Returns:
            Contradiction penalty (0.0 to 1.0, higher = more contradictory)
        """
        contradiction_patterns = [
            (r"not\s+", r"(?i)not\s+\w+"),  # Direct negation
            (r"never\s+", r"(?i)never\s+\w+"),  # Never statements
            (r"don't\s+", r"(?i)don't\s+\w+"),  # Don't statements
            (r"can't\s+", r"(?i)can't\s+\w+"),  # Can't statements
        ]
        
        penalty = 0.0
        
        for pattern_name, pattern in contradiction_patterns:
            if re.search(pattern, fact_claim) and re.search(pattern, context):
                # Both contain negation patterns - potential contradiction
                penalty += 0.2
        
        # Check for opposite preferences
        opposites = [
            ("like", "dislike"), ("prefer", "avoid"), ("want", "don't want"),
            ("need", "don't need"), ("use", "avoid"), ("yes", "no")
        ]
        
        fact_lower = fact_claim.lower()
        context_lower = context.lower()
        
        for pos, neg in opposites:
            if pos in fact_lower and neg in context_lower:
                penalty += 0.3
            elif neg in fact_lower and pos in context_lower:
                penalty += 0.3
        
        return min(1.0, penalty)
    
    def _calculate_nli_score(self, premise: str, hypothesis: str) -> float:
        """
        Calculate NLI entailment score.
        
        Args:
            premise: Context (premise)
            hypothesis: Fact claim (hypothesis)
            
        Returns:
            Entailment probability (0.0 to 1.0)
        """
        if self._vectorizer is None or self._nli_classifier is None:
            return 0.5  # Neutral score
        
        try:
            # Create feature vector
            combined_text = f"{premise[:500]} [SEP] {hypothesis}"
            X = self._vectorizer.transform([combined_text])
            
            # Get entailment probability
            prob = self._nli_classifier.predict_proba(X)[0]
            
            # Return probability of entailment (class 1)
            return prob[1] if len(prob) > 1 else 0.5
            
        except Exception as e:
            logger.warning(f"NLI score calculation failed: {e}")
            return 0.5
    
    def _apply_truth_threshold(self, confidence: float) -> bool:
        """
        Apply confidence threshold to determine if fact is valid.
        
        Args:
            confidence: Confidence score
            
        Returns:
            Whether fact passes threshold
        """
        return confidence >= self._confidence_threshold
    
    def _get_validation_reason(self, confidence: float, is_valid: bool) -> str:
        """
        Generate human-readable validation reason.
        
        Args:
            confidence: Confidence score
            is_valid: Whether fact is valid
            
        Returns:
            Validation reason
        """
        if is_valid:
            if confidence > 0.9:
                return "High confidence - strong evidence in conversation"
            elif confidence > 0.8:
                return "Good confidence - consistent with conversation context"
            else:
                return f"Acceptable confidence ({confidence:.2f}) - passes threshold"
        else:
            if confidence < 0.3:
                return "Low confidence - weak evidence or potential contradiction"
            elif confidence < 0.5:
                return "Below average confidence - insufficient supporting evidence"
            else:
                return f"Below threshold ({confidence:.2f} < {self._confidence_threshold})"
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with model stats
        """
        return {
            "is_initialized": self._is_initialized,
            "confidence_threshold": self._confidence_threshold,
            "nli_enabled": self._enable_nli,
            "max_facts_per_conversation": self._max_facts,
            "has_embeddings": self._embeddings is not None,
            "embeddings_loaded": (
                self._embeddings.is_loaded() if self._embeddings else False
            ),
        }


# Convenience function for creating truth model
def create_truth_model(
    embeddings_manager: Optional[EmbeddingsManager] = None,
    settings: Optional[Settings] = None
) -> TruthModel:
    """
    Create and return a TruthModel instance.
    
    Args:
        embeddings_manager: Optional embeddings manager
        settings: Application settings
        
    Returns:
        TruthModel instance
    """
    return TruthModel(
        embeddings_manager=embeddings_manager,
        settings=settings
    )


# Export main classes and functions
__all__ = [
    "TruthModel",
    "create_truth_model",
]