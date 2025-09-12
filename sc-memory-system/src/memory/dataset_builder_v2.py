"""
Dataset Builder v2 for Week 3 - Advanced dataset generation with paraphrasing and deduplication.

This module enhances training examples from individual conversations through:
- Paraphrasing techniques to increase dataset diversity
- Deduplication to remove similar/redundant examples
- Quality filtering to maintain training data quality
- Metadata tracking for comprehensive dataset statistics

Scale: Enhances 10-50 base examples from ONE conversation → 20-100 improved examples
"""

import asyncio
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..core.config import Settings, get_settings
from ..core.exceptions import ConversationProcessingError
from ..core.models import (
    TrainingExample,
    ParaphrasedExample,
    EnhancedDataset
)

logger = logging.getLogger(__name__)


class ParaphraseGenerator:
    """Generates paraphrased versions of training examples using multiple techniques."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize paraphrase generator."""
        self.settings = settings or get_settings()
        
    async def generate_paraphrases(
        self,
        examples: List[TrainingExample],
        paraphrases_per_example: int = 2
    ) -> List[ParaphrasedExample]:
        """
        Generate paraphrased versions of training examples.
        
        Args:
            examples: Original training examples
            paraphrases_per_example: Number of paraphrases per example
            
        Returns:
            List of paraphrased examples with metadata
        """
        paraphrased_examples = []
        
        for example in examples:
            for i in range(paraphrases_per_example):
                try:
                    # Select paraphrasing technique
                    techniques = self.settings.dataset_builder_v2.paraphrase_techniques
                    technique = random.choice(techniques)
                    
                    # Generate paraphrase based on technique
                    if technique == "synonym_replacement":
                        paraphrased = await self._synonym_replacement(example)
                    elif technique == "sentence_restructuring":
                        paraphrased = await self._sentence_restructuring(example)
                    elif technique == "backtranslation":
                        paraphrased = await self._simple_backtranslation(example)
                    else:
                        paraphrased = await self._synonym_replacement(example)  # Default
                    
                    # Calculate similarity and quality scores
                    similarity_score = await self._calculate_similarity(example, paraphrased)
                    quality_score = await self._assess_quality(paraphrased)
                    
                    paraphrased_example = ParaphrasedExample(
                        original_instruction=example.instruction,
                        original_response=example.response,
                        paraphrased_instruction=paraphrased["instruction"],
                        paraphrased_response=paraphrased["response"],
                        paraphrase_technique=technique,
                        similarity_score=similarity_score,
                        quality_score=quality_score,
                        generation_metadata={
                            "technique_used": technique,
                            "generation_timestamp": datetime.utcnow().isoformat(),
                            "attempt_number": i + 1
                        }
                    )
                    
                    paraphrased_examples.append(paraphrased_example)
                    
                except Exception as e:
                    logger.error(f"Failed to generate paraphrase for example: {e}")
                    continue
        
        logger.info(f"Generated {len(paraphrased_examples)} paraphrased examples")
        return paraphrased_examples
    
    async def _synonym_replacement(self, example: TrainingExample) -> Dict[str, str]:
        """Generate paraphrase using synonym replacement."""
        # Simple synonym replacement using basic word mappings
        synonym_map = {
            "good": "excellent",
            "bad": "poor", 
            "big": "large",
            "small": "tiny",
            "fast": "quick",
            "slow": "gradual",
            "create": "build",
            "make": "generate",
            "use": "utilize",
            "help": "assist",
            "show": "display",
            "find": "locate",
            "get": "obtain",
            "give": "provide",
            "tell": "inform",
            "ask": "inquire",
            "want": "desire",
            "need": "require",
            "know": "understand",
            "think": "believe",
            "like": "prefer",
            "work": "function",
            "run": "execute",
            "stop": "halt",
            "start": "begin",
            "end": "finish"
        }
        
        def replace_synonyms(text: str) -> str:
            words = text.split()
            for i, word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', word.lower())
                if clean_word in synonym_map and random.random() < 0.3:  # 30% replacement chance
                    # Preserve original capitalization and punctuation
                    if word[0].isupper():
                        replacement = synonym_map[clean_word].capitalize()
                    else:
                        replacement = synonym_map[clean_word]
                    
                    # Add back punctuation
                    punct = re.findall(r'[^\w]', word)
                    if punct:
                        replacement += ''.join(punct)
                    
                    words[i] = replacement
            
            return ' '.join(words)
        
        return {
            "instruction": replace_synonyms(example.instruction),
            "response": replace_synonyms(example.response)
        }
    
    async def _sentence_restructuring(self, example: TrainingExample) -> Dict[str, str]:
        """Generate paraphrase using sentence restructuring."""
        def restructure_sentence(text: str) -> str:
            # Simple restructuring patterns
            restructuring_patterns = [
                # "Can you X?" -> "Please X"
                (r"Can you ([^?]+)\?", r"Please \1."),
                # "I want to X" -> "I would like to X"
                (r"I want to ([^.]+)", r"I would like to \1"),
                # "This is X" -> "X is what this is"
                (r"This is ([^.]+)", r"\1 is what this is"),
                # "How to X?" -> "What's the way to X?"
                (r"How to ([^?]+)\?", r"What's the way to \1?"),
                # "Let me X" -> "I will X"
                (r"Let me ([^.]+)", r"I will \1"),
                # "You can X" -> "It's possible to X"
                (r"You can ([^.]+)", r"It's possible to \1"),
                # "We need to X" -> "X is necessary"
                (r"We need to ([^.]+)", r"\1 is necessary"),
            ]
            
            restructured = text
            for pattern, replacement in restructuring_patterns:
                if random.random() < 0.4:  # 40% chance to apply each pattern
                    restructured = re.sub(pattern, replacement, restructured, flags=re.IGNORECASE)
            
            return restructured
        
        return {
            "instruction": restructure_sentence(example.instruction),
            "response": restructure_sentence(example.response)
        }
    
    async def _simple_backtranslation(self, example: TrainingExample) -> Dict[str, str]:
        """Simulate backtranslation with simple text variations."""
        # Simulate backtranslation effects with common changes
        def simulate_backtranslation(text: str) -> str:
            variations = [
                # Articles variations
                (r'\ba\b', 'the'),
                (r'\bthe\b', 'a'),
                # Contractions
                (r"don't", "do not"),
                (r"can't", "cannot"),
                (r"won't", "will not"),
                (r"I'm", "I am"),
                (r"you're", "you are"),
                (r"it's", "it is"),
                # Word order variations
                (r'very ([a-z]+)', r'\1, very much'),
                (r'really ([a-z]+)', r'quite \1'),
                # Preposition changes
                (r'\bin\b', 'within'),
                (r'\bon\b', 'upon'),
                (r'\bwith\b', 'using'),
                (r'\bfor\b', 'in order to'),
            ]
            
            varied_text = text
            for pattern, replacement in variations:
                if random.random() < 0.2:  # 20% chance for each variation
                    varied_text = re.sub(pattern, replacement, varied_text, flags=re.IGNORECASE)
            
            return varied_text
        
        return {
            "instruction": simulate_backtranslation(example.instruction),
            "response": simulate_backtranslation(example.response)
        }
    
    async def _calculate_similarity(
        self,
        original: TrainingExample,
        paraphrased: Dict[str, str]
    ) -> float:
        """Calculate semantic similarity between original and paraphrased examples."""
        try:
            # Create combined texts for comparison
            original_text = f"{original.instruction} {original.response}"
            paraphrased_text = f"{paraphrased['instruction']} {paraphrased['response']}"
            
            # Use TF-IDF for similarity calculation
            vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            try:
                tfidf_matrix = vectorizer.fit_transform([original_text, paraphrased_text])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            except ValueError:
                # Fallback: simple word overlap
                original_words = set(original_text.lower().split())
                paraphrased_words = set(paraphrased_text.lower().split())
                overlap = len(original_words.intersection(paraphrased_words))
                total = len(original_words.union(paraphrased_words))
                return overlap / total if total > 0 else 0.0
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.5  # Default neutral similarity
    
    async def _assess_quality(self, paraphrased: Dict[str, str]) -> float:
        """Assess quality of paraphrased example."""
        quality_score = 1.0
        
        # Check for basic quality issues
        instruction = paraphrased["instruction"]
        response = paraphrased["response"]
        
        # Penalty for very short texts
        if len(instruction.split()) < 3:
            quality_score *= 0.7
        if len(response.split()) < 3:
            quality_score *= 0.7
        
        # Penalty for repetitive words
        instruction_words = instruction.lower().split()
        response_words = response.lower().split()
        
        if len(set(instruction_words)) < len(instruction_words) * 0.7:
            quality_score *= 0.8
        if len(set(response_words)) < len(response_words) * 0.7:
            quality_score *= 0.8
        
        # Penalty for malformed sentences
        if not instruction.strip().endswith(('.', '?', '!')):
            quality_score *= 0.9
        
        return max(0.0, min(1.0, quality_score))


class DataDeduplicator:
    """Removes duplicate and highly similar examples from training data."""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """Initialize deduplicator with similarity threshold."""
        self.similarity_threshold = similarity_threshold
        
    async def deduplicate_examples(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], int]:
        """
        Remove duplicate examples based on similarity.
        
        Args:
            examples: List of training examples
            
        Returns:
            Tuple of (deduplicated_examples, duplicates_removed_count)
        """
        if not examples:
            return examples, 0
        
        logger.info(f"Deduplicating {len(examples)} examples with threshold {self.similarity_threshold}")
        
        # Convert examples to text for comparison
        example_texts = []
        for example in examples:
            combined_text = f"{example.instruction} {example.response}"
            example_texts.append(combined_text)
        
        # Build similarity matrix
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
            
            if len(example_texts) == 1:
                return examples, 0
            
            tfidf_matrix = vectorizer.fit_transform(example_texts)
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
        except ValueError as e:
            logger.warning(f"TF-IDF vectorization failed: {e}, using simple deduplication")
            return await self._simple_deduplication(examples)
        
        # Find duplicates
        duplicates_to_remove = set()
        
        for i in range(len(examples)):
            if i in duplicates_to_remove:
                continue
                
            for j in range(i + 1, len(examples)):
                if j in duplicates_to_remove:
                    continue
                
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    # Keep the example with higher importance (if available) or the first one
                    if hasattr(examples[i], 'importance') and hasattr(examples[j], 'importance'):
                        if examples[j].importance > examples[i].importance:
                            duplicates_to_remove.add(i)
                        else:
                            duplicates_to_remove.add(j)
                    else:
                        duplicates_to_remove.add(j)  # Remove later occurrence
        
        # Remove duplicates
        deduplicated_examples = [
            example for i, example in enumerate(examples)
            if i not in duplicates_to_remove
        ]
        
        duplicates_count = len(duplicates_to_remove)
        logger.info(f"Removed {duplicates_count} duplicate examples, kept {len(deduplicated_examples)}")
        
        return deduplicated_examples, duplicates_count
    
    async def _simple_deduplication(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], int]:
        """Simple deduplication based on exact matches and basic similarity."""
        seen_examples = set()
        deduplicated = []
        duplicates_count = 0
        
        for example in examples:
            # Create a normalized representation
            normalized = f"{example.instruction.lower().strip()} | {example.response.lower().strip()}"
            
            if normalized not in seen_examples:
                seen_examples.add(normalized)
                deduplicated.append(example)
            else:
                duplicates_count += 1
        
        return deduplicated, duplicates_count


class QualityFilter:
    """Filters training examples based on quality metrics."""
    
    def __init__(self, quality_threshold: float = 0.7):
        """Initialize quality filter."""
        self.quality_threshold = quality_threshold
    
    async def filter_examples(
        self,
        examples: List[TrainingExample]
    ) -> Tuple[List[TrainingExample], int]:
        """
        Filter examples based on quality metrics.
        
        Args:
            examples: List of training examples
            
        Returns:
            Tuple of (filtered_examples, filtered_count)
        """
        if not examples:
            return examples, 0
        
        high_quality_examples = []
        filtered_count = 0
        
        for example in examples:
            quality_score = await self._assess_example_quality(example)
            
            if quality_score >= self.quality_threshold:
                high_quality_examples.append(example)
            else:
                filtered_count += 1
        
        logger.info(f"Filtered {filtered_count} low-quality examples, kept {len(high_quality_examples)}")
        return high_quality_examples, filtered_count
    
    async def _assess_example_quality(self, example: TrainingExample) -> float:
        """Assess quality of a training example."""
        quality_score = 1.0
        
        instruction = example.instruction.strip()
        response = example.response.strip()
        
        # Basic length checks
        if len(instruction) < 10:  # Too short instruction
            quality_score *= 0.5
        if len(response) < 10:  # Too short response
            quality_score *= 0.5
        
        # Check for proper sentence structure
        if not instruction.endswith(('.', '?', '!')):
            quality_score *= 0.9
        
        # Check for meaningful content (not just repeated words)
        instruction_words = instruction.lower().split()
        response_words = response.lower().split()
        
        if len(set(instruction_words)) < max(1, len(instruction_words) * 0.5):
            quality_score *= 0.7  # Too repetitive
        
        if len(set(response_words)) < max(1, len(response_words) * 0.5):
            quality_score *= 0.7  # Too repetitive
        
        # Check for empty or placeholder content
        placeholder_patterns = [
            r'\[placeholder\]',
            r'\.\.\.',
            r'xxx+',
            r'TODO',
            r'FIXME',
            r'<[^>]*>',  # HTML-like tags
        ]
        
        combined_text = f"{instruction} {response}".lower()
        for pattern in placeholder_patterns:
            if re.search(pattern, combined_text, re.IGNORECASE):
                quality_score *= 0.3
        
        return max(0.0, min(1.0, quality_score))


class AdvancedDatasetBuilder:
    """Main class for advanced dataset building with paraphrasing and deduplication."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize advanced dataset builder."""
        self.settings = settings or get_settings()
        self.paraphrase_generator = ParaphraseGenerator(self.settings)
        self.deduplicator = DataDeduplicator(
            self.settings.dataset_builder_v2.similarity_threshold
        )
        self.quality_filter = QualityFilter(
            self.settings.dataset_builder_v2.quality_filter_threshold
        )
    
    async def enhance_conversation_training_data(
        self,
        original_examples: List[TrainingExample],
        conversation_id: str
    ) -> EnhancedDataset:
        """
        Enhance training examples from a single conversation.
        
        Input: 10-50 base examples from one conversation
        Output: 20-100 improved examples from same conversation
        
        Args:
            original_examples: Base training examples from conversation
            conversation_id: Unique identifier for the conversation
            
        Returns:
            EnhancedDataset with improved training data and metadata
        """
        start_time = time.time()
        
        logger.info(f"Enhancing dataset for conversation {conversation_id}: {len(original_examples)} original examples")
        
        try:
            # Step 1: Generate paraphrases if enabled
            paraphrased_examples = []
            if self.settings.dataset_builder_v2.enable_paraphrasing and original_examples:
                paraphrases_per_example = self.settings.dataset_builder_v2.paraphrases_per_example
                paraphrased_examples = await self.paraphrase_generator.generate_paraphrases(
                    original_examples,
                    paraphrases_per_example
                )
            
            # Step 2: Combine original and paraphrased examples
            all_examples = original_examples.copy()
            
            # Convert paraphrased examples to training examples
            for paraphrased in paraphrased_examples:
                training_example = TrainingExample(
                    instruction=paraphrased.paraphrased_instruction,
                    response=paraphrased.paraphrased_response,
                    metadata={
                        "source": "paraphrased",
                        "technique": paraphrased.paraphrase_technique,
                        "similarity_score": paraphrased.similarity_score,
                        "quality_score": paraphrased.quality_score
                    }
                )
                all_examples.append(training_example)
            
            # Step 3: Deduplicate if enabled
            duplicates_removed_count = 0
            if self.settings.dataset_builder_v2.enable_deduplication and len(all_examples) > 1:
                all_examples, duplicates_removed_count = await self.deduplicator.deduplicate_examples(
                    all_examples
                )
            
            # Step 4: Quality filtering
            quality_filtered_count = 0
            final_examples, quality_filtered_count = await self.quality_filter.filter_examples(
                all_examples
            )
            
            # Step 5: Limit dataset size
            max_size = self.settings.dataset_builder_v2.max_dataset_size
            if len(final_examples) > max_size:
                # Keep highest quality examples
                final_examples = sorted(
                    final_examples,
                    key=lambda x: x.metadata.get('quality_score', 0.8),
                    reverse=True
                )[:max_size]
            
            # Calculate statistics
            processing_time = time.time() - start_time
            average_quality = 0.0
            if final_examples:
                quality_scores = [
                    example.metadata.get('quality_score', 0.8) 
                    for example in final_examples
                ]
                average_quality = sum(quality_scores) / len(quality_scores)
            
            # Create enhanced dataset
            enhanced_dataset = EnhancedDataset(
                conversation_id=conversation_id,
                original_examples=original_examples,
                paraphrased_examples=paraphrased_examples,
                duplicates_removed_count=duplicates_removed_count,
                quality_filtered_count=quality_filtered_count,
                final_dataset_size=len(final_examples),
                average_quality_score=average_quality,
                generation_time_seconds=processing_time,
                processing_metadata={
                    "paraphrasing_enabled": self.settings.dataset_builder_v2.enable_paraphrasing,
                    "deduplication_enabled": self.settings.dataset_builder_v2.enable_deduplication,
                    "paraphrases_generated": len(paraphrased_examples),
                    "original_count": len(original_examples),
                    "combined_count": len(original_examples) + len(paraphrased_examples),
                    "final_count": len(final_examples),
                    "enhancement_ratio": len(final_examples) / len(original_examples) if original_examples else 0,
                    "processing_timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # Update final examples in the dataset object (store as training examples)
            enhanced_dataset.processing_metadata["final_training_examples"] = [
                example.model_dump() for example in final_examples
            ]
            
            logger.info(
                f"Dataset enhancement completed for conversation {conversation_id}: "
                f"{len(original_examples)} → {len(final_examples)} examples "
                f"(+{len(paraphrased_examples)} paraphrases, -{duplicates_removed_count} duplicates, "
                f"-{quality_filtered_count} low-quality) in {processing_time:.2f}s"
            )
            
            return enhanced_dataset
            
        except Exception as e:
            logger.error(f"Dataset enhancement failed for conversation {conversation_id}: {e}", exc_info=True)
            raise ConversationProcessingError(
                f"Dataset enhancement failed: {e}",
                conversation_id=conversation_id,
                processing_stage="dataset_enhancement"
            )


# Factory function for creating advanced dataset builder
def create_advanced_dataset_builder(settings: Optional[Settings] = None) -> AdvancedDatasetBuilder:
    """
    Create an advanced dataset builder instance.
    
    Args:
        settings: Optional settings, uses global settings if None
        
    Returns:
        AdvancedDatasetBuilder instance
    """
    return AdvancedDatasetBuilder(settings)