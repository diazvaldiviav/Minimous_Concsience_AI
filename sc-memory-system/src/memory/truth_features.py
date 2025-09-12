"""
Week 3 Truth Features Extraction (v2) for enhanced fact validation.

This module provides advanced truth validation features including:
- RAG-based validation against knowledge bases
- Advanced NLI capabilities with multiple techniques
- Provenance tracking for validation sources
- Ensemble validation with calibrated confidence scores
- Multi-feature truth validation pipeline
"""

import asyncio
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from ..core.config import Settings, get_settings
from ..core.exceptions import TruthValidationError
from ..core.models import (
    KeyFact,
    ProvenanceInfo,
    EnhancedValidatedFact
)


logger = logging.getLogger(__name__)


class RAGSupportValidator:
    """RAG-based fact validation using knowledge base retrieval."""

    def __init__(self, knowledge_base_path: Path, settings: Optional[Settings] = None):
        """
        Initialize RAG support validator.

        Args:
            knowledge_base_path: Path to knowledge base directory
            settings: Application settings
        """
        self.knowledge_base_path = knowledge_base_path
        self.settings = settings or get_settings()
        self.knowledge_base: List[Dict[str, str]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.kb_vectors: Optional[np.ndarray] = None
        
    async def load_knowledge_base(self) -> None:
        """Load and index the knowledge base."""
        try:
            if not self.knowledge_base_path.exists():
                logger.warning(f"Knowledge base path does not exist: {self.knowledge_base_path}")
                self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
                
                # Create sample knowledge base for testing
                await self._create_sample_kb()
                
            # Load knowledge base files
            for kb_file in self.knowledge_base_path.glob("*.json"):
                try:
                    with open(kb_file, 'r', encoding='utf-8') as f:
                        kb_data = json.load(f)
                        if isinstance(kb_data, list):
                            self.knowledge_base.extend(kb_data)
                        else:
                            self.knowledge_base.append(kb_data)
                except Exception as e:
                    logger.error(f"Failed to load knowledge base file {kb_file}: {e}")
            
            if not self.knowledge_base:
                logger.warning("No knowledge base loaded, creating minimal sample")
                await self._create_sample_kb()
            
            # Build TF-IDF index
            await self._build_index()
            
            logger.info(f"Loaded knowledge base with {len(self.knowledge_base)} entries")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            raise TruthValidationError(f"Knowledge base loading failed: {e}")

    async def _create_sample_kb(self) -> None:
        """Create a sample knowledge base for testing."""
        sample_kb = [
            {
                "id": "kb_001",
                "title": "Python Programming Language",
                "content": "Python is a high-level programming language known for its readability and versatility. It was created by Guido van Rossum and first released in 1991.",
                "category": "programming",
                "confidence": 0.95
            },
            {
                "id": "kb_002", 
                "title": "Machine Learning Basics",
                "content": "Machine Learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
                "category": "ai",
                "confidence": 0.9
            },
            {
                "id": "kb_003",
                "title": "Web Development",
                "content": "Web development involves creating websites and web applications using technologies like HTML, CSS, JavaScript, and various frameworks.",
                "category": "web",
                "confidence": 0.85
            }
        ]
        
        kb_file = self.knowledge_base_path / "sample_kb.json"
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(sample_kb, f, indent=2)
        
        self.knowledge_base = sample_kb

    async def _build_index(self) -> None:
        """Build TF-IDF index for knowledge base retrieval."""
        if not self.knowledge_base:
            return
            
        # Extract text content for indexing
        documents = []
        for entry in self.knowledge_base:
            # Combine title and content for better matching
            text = f"{entry.get('title', '')} {entry.get('content', '')}"
            documents.append(text)
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.kb_vectors = self.vectorizer.fit_transform(documents)
        logger.info(f"Built TF-IDF index with {self.kb_vectors.shape[0]} documents and {self.kb_vectors.shape[1]} features")

    async def validate_fact_with_rag(
        self, 
        fact_claim: str, 
        context: str = "",
        top_k: int = 3,
        threshold: float = 0.3
    ) -> ProvenanceInfo:
        """
        Validate a fact claim using RAG retrieval.

        Args:
            fact_claim: The claim to validate
            context: Additional context for validation
            top_k: Number of top knowledge base entries to retrieve
            threshold: Minimum similarity threshold for relevance

        Returns:
            ProvenanceInfo with RAG validation results
        """
        try:
            if not self.vectorizer or self.kb_vectors is None:
                await self.load_knowledge_base()
            
            # Vectorize the query
            query_text = f"{fact_claim} {context}".strip()
            query_vector = self.vectorizer.transform([query_text])
            
            # Compute similarities
            similarities = cosine_similarity(query_vector, self.kb_vectors)[0]
            
            # Get top-k most similar entries
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Filter by threshold
            relevant_entries = []
            for idx in top_indices:
                if similarities[idx] >= threshold:
                    entry = self.knowledge_base[idx]
                    relevant_entries.append({
                        "entry": entry,
                        "similarity": float(similarities[idx])
                    })
            
            # Calculate confidence based on best match and number of supporting entries
            if relevant_entries:
                best_similarity = relevant_entries[0]["similarity"]
                support_factor = min(1.0, len(relevant_entries) / top_k)
                confidence = best_similarity * support_factor
            else:
                confidence = 0.0
            
            # Create provenance info
            provenance = ProvenanceInfo(
                source_type="rag",
                source_confidence=confidence,
                source_details={
                    "query_text": query_text,
                    "relevant_entries_count": len(relevant_entries),
                    "top_similarity": relevant_entries[0]["similarity"] if relevant_entries else 0.0,
                    "threshold_used": threshold,
                    "relevant_entries": relevant_entries[:2]  # Include top 2 for debugging
                },
                model_version="tfidf_cosine_v1"
            )
            
            return provenance
            
        except Exception as e:
            logger.error(f"RAG validation failed for claim '{fact_claim[:50]}...': {e}")
            return ProvenanceInfo(
                source_type="rag",
                source_confidence=0.0,
                source_details={"error": str(e)},
                model_version="tfidf_cosine_v1"
            )


class AdvancedNLIValidator:
    """Advanced Natural Language Inference validator with multiple techniques."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize Advanced NLI validator."""
        self.settings = settings or get_settings()
        self.classifier: Optional[LogisticRegression] = None
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.is_trained = False
        
    async def load_or_train_nli_model(self) -> None:
        """Load existing NLI model or train a simple one."""
        try:
            # For MVP, use a simple TF-IDF + LogisticRegression approach
            # In production, this would use a pre-trained transformer model
            
            if not self.is_trained:
                await self._train_simple_nli_model()
            
        except Exception as e:
            logger.error(f"Failed to load/train NLI model: {e}")
            raise TruthValidationError(f"NLI model initialization failed: {e}")

    async def _train_simple_nli_model(self) -> None:
        """Train a simple NLI model for demonstration."""
        # Training data for basic entailment detection
        training_data = [
            # Entailment examples
            ("Python is a programming language", "Python is used for programming", 1),
            ("Machine learning uses data", "ML algorithms process data", 1),
            ("Web development uses HTML", "HTML is used in web development", 1),
            ("FastAPI is a Python framework", "FastAPI is built with Python", 1),
            ("Neural networks learn patterns", "Deep learning finds patterns in data", 1),
            
            # Contradiction examples
            ("Python is a programming language", "Python is a spoken language", 0),
            ("Machine learning uses data", "ML works without any data", 0),
            ("Web development uses HTML", "Web development never uses markup", 0),
            ("FastAPI is fast", "FastAPI is extremely slow", 0),
            ("AI helps with automation", "AI makes everything manual", 0),
            
            # Neutral examples  
            ("Python is popular", "JavaScript is popular", 0.5),
            ("Machine learning is complex", "Cooking is an art", 0.5),
            ("Web development changes quickly", "Cars need maintenance", 0.5),
        ]
        
        # Prepare training features
        premise_texts = [item[0] for item in training_data]
        hypothesis_texts = [item[1] for item in training_data]
        labels = [item[2] for item in training_data]
        
        # Combine premise and hypothesis for feature extraction
        combined_texts = [f"{p} [SEP] {h}" for p, h in zip(premise_texts, hypothesis_texts)]
        
        # Build TF-IDF features
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        features = self.vectorizer.fit_transform(combined_texts)
        
        # Train simple classifier
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(features, labels)
        
        self.is_trained = True
        logger.info("Trained simple NLI model for demonstration")

    async def validate_with_nli(
        self, 
        premise: str, 
        hypothesis: str,
        context: str = ""
    ) -> ProvenanceInfo:
        """
        Validate using Natural Language Inference.

        Args:
            premise: The premise statement (context/background)
            hypothesis: The hypothesis to validate (fact claim)
            context: Additional context

        Returns:
            ProvenanceInfo with NLI validation results
        """
        try:
            if not self.is_trained:
                await self.load_or_train_nli_model()
            
            # Prepare input
            combined_input = f"{premise} {context} [SEP] {hypothesis}".strip()
            features = self.vectorizer.transform([combined_input])
            
            # Get prediction and confidence
            prediction_proba = self.classifier.predict_proba(features)[0]
            prediction = self.classifier.predict(features)[0]
            
            # Convert to confidence score
            if len(prediction_proba) == 2:  # Binary classification
                confidence = float(prediction_proba[1]) if prediction > 0.5 else float(1 - prediction_proba[0])
            else:  # Multi-class
                confidence = float(max(prediction_proba))
            
            provenance = ProvenanceInfo(
                source_type="nli",
                source_confidence=confidence,
                source_details={
                    "premise": premise[:100] + "..." if len(premise) > 100 else premise,
                    "hypothesis": hypothesis[:100] + "..." if len(hypothesis) > 100 else hypothesis,
                    "prediction": float(prediction),
                    "prediction_probabilities": [float(p) for p in prediction_proba],
                    "entailment_confidence": confidence
                },
                model_version="simple_tfidf_lr_v1"
            )
            
            return provenance
            
        except Exception as e:
            logger.error(f"NLI validation failed: {e}")
            return ProvenanceInfo(
                source_type="nli",
                source_confidence=0.0,
                source_details={"error": str(e)},
                model_version="simple_tfidf_lr_v1"
            )


class ProvenanceTracker:
    """Tracks provenance information for fact validation sources."""

    def __init__(self):
        """Initialize provenance tracker."""
        self.validation_history: List[ProvenanceInfo] = []
        
    def add_validation(self, provenance: ProvenanceInfo) -> None:
        """Add a validation result to the provenance chain."""
        self.validation_history.append(provenance)
    
    def get_validation_chain(self) -> List[ProvenanceInfo]:
        """Get the complete validation chain."""
        return self.validation_history.copy()
    
    def get_source_summary(self) -> Dict[str, Any]:
        """Get summary of all validation sources."""
        summary = {
            "total_sources": len(self.validation_history),
            "source_types": {},
            "average_confidence": 0.0,
            "confidence_range": {"min": 1.0, "max": 0.0}
        }
        
        if not self.validation_history:
            return summary
        
        # Count source types
        for prov in self.validation_history:
            source_type = prov.source_type
            if source_type not in summary["source_types"]:
                summary["source_types"][source_type] = 0
            summary["source_types"][source_type] += 1
        
        # Calculate confidence statistics
        confidences = [prov.source_confidence for prov in self.validation_history]
        summary["average_confidence"] = sum(confidences) / len(confidences)
        summary["confidence_range"]["min"] = min(confidences)
        summary["confidence_range"]["max"] = max(confidences)
        
        return summary


class TruthFeatureExtractor:
    """Main class for extracting truth features with multiple validation sources."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize truth feature extractor.

        Args:
            settings: Application settings
        """
        self.settings = settings or get_settings()
        
        # Initialize validators
        self.rag_validator = RAGSupportValidator(
            self.settings.truth_features.knowledge_base_path,
            self.settings
        )
        self.nli_validator = AdvancedNLIValidator(self.settings)
        
        # Load embeddings for semantic similarity
        self._embeddings_manager = None

    async def initialize(self) -> None:
        """Initialize all components."""
        try:
            # Initialize RAG validator
            if self.settings.truth_features.enable_rag_validation:
                await self.rag_validator.load_knowledge_base()
            
            # Initialize NLI validator
            if self.settings.truth_features.enable_nli_validation:
                await self.nli_validator.load_or_train_nli_model()
            
            # Initialize embeddings manager for semantic similarity
            from .embeddings import create_embeddings_manager
            self._embeddings_manager = create_embeddings_manager()
            
            logger.info("TruthFeatureExtractor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TruthFeatureExtractor: {e}")
            raise TruthValidationError(f"Initialization failed: {e}")

    async def validate_fact_comprehensive(
        self, 
        fact: KeyFact,
        conversation_context: str = "",
        enable_all_features: bool = True
    ) -> EnhancedValidatedFact:
        """
        Perform comprehensive fact validation using all available features.

        Args:
            fact: The fact to validate
            conversation_context: Context from conversation
            enable_all_features: Whether to use all validation features

        Returns:
            EnhancedValidatedFact with comprehensive validation results
        """
        try:
            provenance_tracker = ProvenanceTracker()
            validation_sources = []
            
            # RAG-based validation
            if self.settings.truth_features.enable_rag_validation and enable_all_features:
                rag_result = await self.rag_validator.validate_fact_with_rag(
                    fact.claim,
                    conversation_context
                )
                validation_sources.append(rag_result)
                provenance_tracker.add_validation(rag_result)
            
            # NLI-based validation
            if self.settings.truth_features.enable_nli_validation and enable_all_features:
                nli_result = await self.nli_validator.validate_with_nli(
                    conversation_context,
                    fact.claim,
                    ""
                )
                validation_sources.append(nli_result)
                provenance_tracker.add_validation(nli_result)
            
            # Semantic similarity validation
            similarity_result = await self._validate_semantic_similarity(
                fact.claim,
                conversation_context
            )
            validation_sources.append(similarity_result)
            provenance_tracker.add_validation(similarity_result)
            
            # Consistency validation
            consistency_result = await self._validate_consistency(
                fact.claim,
                conversation_context
            )
            validation_sources.append(consistency_result)
            provenance_tracker.add_validation(consistency_result)
            
            # Ensemble validation with weighted combination
            ensemble_confidence = await self._compute_ensemble_confidence(validation_sources)
            
            # Calibrate confidence if enabled
            calibrated_confidence = None
            if self.settings.truth_features.confidence_calibration_samples > 0:
                calibrated_confidence = await self._calibrate_confidence(
                    ensemble_confidence,
                    validation_sources
                )
            
            # Detect contradictions
            contradictions = await self._detect_contradictions(
                fact.claim,
                conversation_context,
                validation_sources
            )
            
            # Find supporting evidence
            supporting_evidence = await self._find_supporting_evidence(
                fact.claim,
                validation_sources
            )
            
            # Determine if fact is validated
            threshold = self.settings.truth_model.validation_threshold
            is_validated = ensemble_confidence >= threshold
            
            enhanced_fact = EnhancedValidatedFact(
                original_claim=fact.claim,
                validation_confidence=ensemble_confidence,
                is_validated=is_validated,
                validation_sources=validation_sources,
                ensemble_weights=self.settings.truth_features.ensemble_weights,
                calibrated_confidence=calibrated_confidence,
                contradictions_detected=contradictions,
                supporting_evidence=supporting_evidence
            )
            
            logger.info(
                f"Comprehensive validation completed for fact: "
                f"confidence={ensemble_confidence:.3f}, validated={is_validated}"
            )
            
            return enhanced_fact
            
        except Exception as e:
            logger.error(f"Comprehensive validation failed: {e}", exc_info=True)
            raise TruthValidationError(f"Validation failed: {e}", fact_claim=fact.claim)

    async def _validate_semantic_similarity(
        self,
        claim: str,
        context: str
    ) -> ProvenanceInfo:
        """Validate using semantic similarity."""
        try:
            if not self._embeddings_manager:
                return ProvenanceInfo(
                    source_type="semantic_similarity",
                    source_confidence=0.5,  # Default neutral confidence
                    source_details={"error": "Embeddings manager not available"}
                )
            
            # Generate embeddings
            claim_embedding = await self._embeddings_manager.generate_embeddings([claim])
            context_embedding = await self._embeddings_manager.generate_embeddings([context])
            
            # Calculate cosine similarity
            similarity = float(cosine_similarity(
                claim_embedding.embeddings[0].reshape(1, -1),
                context_embedding.embeddings[0].reshape(1, -1)
            )[0][0])
            
            # Convert similarity to confidence (0.5 is neutral, >0.5 is positive)
            confidence = (similarity + 1) / 2  # Convert from [-1,1] to [0,1]
            
            return ProvenanceInfo(
                source_type="semantic_similarity",
                source_confidence=confidence,
                source_details={
                    "cosine_similarity": similarity,
                    "claim_length": len(claim),
                    "context_length": len(context)
                },
                model_version="sentence_transformers_v1"
            )
            
        except Exception as e:
            logger.error(f"Semantic similarity validation failed: {e}")
            return ProvenanceInfo(
                source_type="semantic_similarity", 
                source_confidence=0.5,
                source_details={"error": str(e)}
            )

    async def _validate_consistency(
        self,
        claim: str,
        context: str
    ) -> ProvenanceInfo:
        """Validate internal consistency."""
        try:
            # Simple consistency checks
            consistency_score = 0.5  # Start neutral
            
            # Check if claim contradicts itself
            claim_lower = claim.lower()
            contradiction_indicators = ["not", "never", "impossible", "cannot", "won't", "don't"]
            positive_indicators = ["is", "can", "will", "does", "always", "definitely"]
            
            contradiction_count = sum(1 for indicator in contradiction_indicators if indicator in claim_lower)
            positive_count = sum(1 for indicator in positive_indicators if indicator in claim_lower)
            
            # If both contradictory and positive indicators, reduce confidence
            if contradiction_count > 0 and positive_count > 0:
                consistency_score *= 0.7
            
            # Check for logical consistency with context
            if context:
                context_lower = context.lower()
                claim_words = set(claim_lower.split())
                context_words = set(context_lower.split())
                
                # Calculate word overlap
                overlap = len(claim_words.intersection(context_words))
                total_unique = len(claim_words.union(context_words))
                
                if total_unique > 0:
                    overlap_ratio = overlap / total_unique
                    consistency_score += overlap_ratio * 0.3
            
            # Normalize to [0, 1]
            consistency_score = max(0.0, min(1.0, consistency_score))
            
            return ProvenanceInfo(
                source_type="consistency_score",
                source_confidence=consistency_score,
                source_details={
                    "contradiction_indicators": contradiction_count,
                    "positive_indicators": positive_count,
                    "word_overlap_with_context": overlap if context else 0
                },
                model_version="simple_consistency_v1"
            )
            
        except Exception as e:
            logger.error(f"Consistency validation failed: {e}")
            return ProvenanceInfo(
                source_type="consistency_score",
                source_confidence=0.5,
                source_details={"error": str(e)}
            )

    async def _compute_ensemble_confidence(
        self,
        validation_sources: List[ProvenanceInfo]
    ) -> float:
        """Compute weighted ensemble confidence from multiple sources."""
        if not validation_sources:
            return 0.0
        
        total_weight = 0.0
        weighted_sum = 0.0
        
        weights = self.settings.truth_features.ensemble_weights
        
        for source in validation_sources:
            source_type = source.source_type
            weight = weights.get(source_type, 0.1)  # Default small weight for unknown sources
            
            weighted_sum += source.source_confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight

    async def _calibrate_confidence(
        self,
        raw_confidence: float,
        validation_sources: List[ProvenanceInfo]
    ) -> float:
        """Apply confidence calibration (simplified version)."""
        try:
            # Simple calibration: adjust based on agreement between sources
            confidences = [source.source_confidence for source in validation_sources]
            
            if len(confidences) <= 1:
                return raw_confidence
            
            # Calculate variance in confidence scores
            mean_conf = sum(confidences) / len(confidences)
            variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)
            
            # If high variance, reduce confidence (sources disagree)
            # If low variance, confidence is more reliable
            calibration_factor = 1.0 - (variance * 0.5)  # Max 50% reduction for disagreement
            
            calibrated = raw_confidence * calibration_factor
            return max(0.0, min(1.0, calibrated))
            
        except Exception as e:
            logger.error(f"Confidence calibration failed: {e}")
            return raw_confidence

    async def _detect_contradictions(
        self,
        claim: str,
        context: str,
        validation_sources: List[ProvenanceInfo]
    ) -> List[str]:
        """Detect contradictions in validation sources."""
        contradictions = []
        
        try:
            # Check for contradictory confidence scores
            confidences = [source.source_confidence for source in validation_sources]
            
            if len(confidences) >= 2:
                min_conf = min(confidences)
                max_conf = max(confidences)
                
                # If there's a large disagreement between sources
                if max_conf - min_conf > 0.6:
                    contradictions.append(
                        f"Large disagreement between validation sources: "
                        f"range {min_conf:.2f} to {max_conf:.2f}"
                    )
            
            # Check for explicit contradictions in claim
            claim_lower = claim.lower()
            if any(word in claim_lower for word in ["but", "however", "although", "despite"]):
                contradictions.append("Claim contains contradictory language")
            
        except Exception as e:
            logger.error(f"Contradiction detection failed: {e}")
        
        return contradictions

    async def _find_supporting_evidence(
        self,
        claim: str,
        validation_sources: List[ProvenanceInfo]
    ) -> List[str]:
        """Find supporting evidence from validation sources."""
        evidence = []
        
        try:
            for source in validation_sources:
                if source.source_confidence > 0.7:  # High confidence sources
                    source_details = source.source_details
                    
                    if source.source_type == "rag":
                        relevant_entries = source_details.get("relevant_entries", [])
                        for entry in relevant_entries[:2]:  # Top 2 entries
                            if entry.get("similarity", 0) > 0.5:
                                evidence.append(
                                    f"Knowledge base entry: {entry.get('entry', {}).get('title', 'Unknown')}"
                                )
                    
                    elif source.source_type == "nli":
                        if source_details.get("prediction", 0) > 0.7:
                            evidence.append("Natural language inference supports claim")
                    
                    elif source.source_type == "semantic_similarity":
                        similarity = source_details.get("cosine_similarity", 0)
                        if similarity > 0.5:
                            evidence.append(f"High semantic similarity with context ({similarity:.2f})")
            
        except Exception as e:
            logger.error(f"Evidence finding failed: {e}")
        
        return evidence


# Factory function for creating truth feature extractor
async def create_truth_feature_extractor(settings: Optional[Settings] = None) -> TruthFeatureExtractor:
    """
    Create and initialize a truth feature extractor.

    Args:
        settings: Optional settings, uses global settings if None

    Returns:
        Initialized TruthFeatureExtractor instance
    """
    extractor = TruthFeatureExtractor(settings)
    await extractor.initialize()
    return extractor