"""
Conversation Processor for SC Memory System.

This module converts specific conversations from MEP proposals into instruction-response
pairs for LoRA training, enabling the model to "remember" that specific conversation.
"""

import asyncio
import json
import logging
import re
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.config import Settings, get_settings
from ..core.exceptions import ConversationProcessingError
from ..core.models import (
    MEPProposal,
    ValidatedFact,
    TrainingExample,
    ConversationTurn,
)
from ..api.mep.schemas import MEPProposalRequest

logger = logging.getLogger(__name__)


class ConversationProcessor:
    """
    Processor for converting conversations to training data.
    
    Converts specific conversations from Claude into instruction-response pairs
    that teach a LoRA adapter to "remember" that particular conversation.
    
    Features:
    - Conversation parsing and turn extraction
    - Memory-focused instruction generation
    - Fact-based recall examples
    - Context-aware response generation
    - Training data export in multiple formats
    """
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """
        Initialize the conversation processor.
        
        Args:
            settings: Application settings
        """
        self._settings = settings or get_settings()
        
        # Configuration
        self._min_turns = self._settings.conversation_processing.min_turns_for_training
        self._max_examples = self._settings.conversation_processing.max_examples_per_conversation
        self._include_paraphrases = self._settings.conversation_processing.include_paraphrases
        self._training_data_dir = self._settings.conversation_processing.training_data_dir
        
        # Ensure output directory exists
        self._training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics tracking
        self._processed_conversations = 0
        self._generated_examples = 0
        
        logger.info(
            f"Initialized ConversationProcessor (min_turns={self._min_turns}, "
            f"max_examples={self._max_examples})"
        )
    
    async def process_conversation_to_training_data(
        self,
        proposal: MEPProposalRequest,
        validated_facts: Optional[List[ValidatedFact]] = None
    ) -> List[TrainingExample]:
        """
        Convert MEP proposal conversation into training examples.
        
        Args:
            proposal: MEP proposal containing conversation data
            validated_facts: Optional pre-validated facts
            
        Returns:
            List of training examples for this conversation
        """
        conversation_id = proposal.external_chat_id
        
        try:
            # Extract conversation turns from summary
            # For MVP, we work with summary text since full conversation isn't available
            turns = self._extract_conversation_turns(proposal.summary_text)
            
            if len(turns) < self._min_turns:
                logger.warning(
                    f"Conversation {conversation_id} has only {len(turns)} turns "
                    f"(minimum {self._min_turns} required)"
                )
            
            # Generate training examples
            training_examples = []
            
            # 1. Memory recall examples from validated facts
            if validated_facts:
                fact_examples = self._create_recall_examples(
                    validated_facts, conversation_id
                )
                training_examples.extend(fact_examples)
            
            # 2. Conversation summary examples
            summary_examples = self._generate_summary_examples(
                proposal.summary_text, conversation_id
            )
            training_examples.extend(summary_examples)
            
            # 3. Context-aware memory examples
            memory_examples = await self._generate_memory_instruction_pairs(
                turns, validated_facts or [], conversation_id
            )
            training_examples.extend(memory_examples)
            
            # 4. Topic-based recall examples
            if proposal.key_facts:
                topic_examples = self._generate_topic_examples(
                    proposal.key_facts, conversation_id
                )
                training_examples.extend(topic_examples)
            
            # Limit total examples
            if len(training_examples) > self._max_examples:
                training_examples = training_examples[:self._max_examples]
                logger.info(
                    f"Limited training examples to {self._max_examples} "
                    f"for conversation {conversation_id}"
                )
            
            # Add paraphrases if enabled
            if self._include_paraphrases and len(training_examples) < self._max_examples:
                paraphrased = self._add_paraphrases(training_examples, conversation_id)
                training_examples.extend(paraphrased)
                training_examples = training_examples[:self._max_examples]
            
            self._processed_conversations += 1
            self._generated_examples += len(training_examples)
            
            logger.info(
                f"Generated {len(training_examples)} training examples "
                f"for conversation {conversation_id}"
            )
            
            return training_examples
            
        except Exception as e:
            error_msg = f"Failed to process conversation {conversation_id}: {str(e)}"
            logger.error(error_msg)
            raise ConversationProcessingError(
                message=error_msg,
                conversation_id=conversation_id,
                processing_stage="training_data_generation"
            )
    
    def _extract_conversation_turns(self, summary_text: str) -> List[ConversationTurn]:
        """
        Extract conversation turns from summary text.
        
        Args:
            summary_text: Conversation summary
            
        Returns:
            List of conversation turns
        """
        # For MVP, create pseudo-turns from summary
        # In production, would parse actual conversation transcript
        turns = []
        
        # Split summary into logical sections
        sentences = [s.strip() for s in summary_text.split('.') if s.strip()]
        
        for i, sentence in enumerate(sentences):
            if len(sentence) < 10:  # Skip very short sentences
                continue
            
            # Alternate between user and assistant turns
            role = "user" if i % 2 == 0 else "assistant"
            
            turn = ConversationTurn(
                role=role,
                content=sentence + ".",
                turn_index=i,
                metadata={"source": "summary_extraction"}
            )
            
            turns.append(turn)
        
        return turns
    
    def _create_recall_examples(
        self,
        validated_facts: List[ValidatedFact],
        conversation_id: str
    ) -> List[TrainingExample]:
        """
        Create recall examples from validated facts.
        
        Args:
            validated_facts: Facts that passed validation
            conversation_id: Source conversation ID
            
        Returns:
            List of recall training examples
        """
        examples = []
        
        for fact in validated_facts:
            if not fact.is_validated:
                continue
            
            # Direct fact recall
            instruction = f"What do you remember about {self._extract_topic(fact.original_fact.claim)}?"
            response = f"I remember that {fact.original_fact.claim}"
            
            examples.append(TrainingExample(
                instruction=instruction,
                response=response,
                source_conversation_id=conversation_id,
                example_type="recall"
            ))
            
            # Category-specific recall if available
            if fact.original_fact.category:
                category_instruction = f"What {fact.original_fact.category} did we discuss?"
                category_response = f"Regarding {fact.original_fact.category}, {fact.original_fact.claim}"
                
                examples.append(TrainingExample(
                    instruction=category_instruction,
                    response=category_response,
                    source_conversation_id=conversation_id,
                    example_type="recall"
                ))
        
        return examples
    
    def _generate_summary_examples(
        self,
        summary_text: str,
        conversation_id: str
    ) -> List[TrainingExample]:
        """
        Generate summary-based training examples.
        
        Args:
            summary_text: Conversation summary
            conversation_id: Source conversation ID
            
        Returns:
            List of summary training examples
        """
        examples = []
        
        # Full summary recall
        examples.append(TrainingExample(
            instruction="Can you summarize our previous conversation?",
            response=summary_text,
            source_conversation_id=conversation_id,
            example_type="summary"
        ))
        
        # Variations of summary request
        summary_variations = [
            "What did we talk about before?",
            "Remind me what we discussed.",
            "What was our conversation about?",
            "Can you recap our discussion?",
        ]
        
        for variation in summary_variations[:2]:  # Limit variations
            examples.append(TrainingExample(
                instruction=variation,
                response=summary_text,
                source_conversation_id=conversation_id,
                example_type="summary"
            ))
        
        return examples
    
    async def _generate_memory_instruction_pairs(
        self,
        turns: List[ConversationTurn],
        validated_facts: List[ValidatedFact],
        conversation_id: str
    ) -> List[TrainingExample]:
        """
        Generate memory-focused instruction pairs.
        
        Args:
            turns: Conversation turns
            validated_facts: Validated facts
            conversation_id: Source conversation ID
            
        Returns:
            List of memory training examples
        """
        examples = []
        
        # Create examples based on conversation flow
        for i, turn in enumerate(turns):
            if turn.role == "assistant" and len(turn.content) > 20:
                # Create "what did I say about X" examples
                topic = self._extract_topic(turn.content)
                if topic:
                    instruction = f"What did I say about {topic}?"
                    response = turn.content
                    
                    examples.append(TrainingExample(
                        instruction=instruction,
                        response=response,
                        source_conversation_id=conversation_id,
                        example_type="recall"
                    ))
            
            elif turn.role == "user" and len(turn.content) > 20:
                # Create "what did you ask about X" examples
                topic = self._extract_topic(turn.content)
                if topic:
                    instruction = f"What did you ask me about {topic}?"
                    response = turn.content
                    
                    examples.append(TrainingExample(
                        instruction=instruction,
                        response=response,
                        source_conversation_id=conversation_id,
                        example_type="recall"
                    ))
        
        return examples
    
    def _generate_topic_examples(
        self,
        key_facts: List,  # List[KeyFact] from proposal
        conversation_id: str
    ) -> List[TrainingExample]:
        """
        Generate topic-based examples from key facts.
        
        Args:
            key_facts: Key facts from MEP proposal
            conversation_id: Source conversation ID
            
        Returns:
            List of topic training examples
        """
        examples = []
        
        # Extract topics from key facts
        topics = []
        for fact_dict in key_facts:
            # Extract topic from claim
            claim = fact_dict.get('claim', '')
            topic = self._extract_topic(claim)
            if topic:
                topics.append((topic, claim))
        
        # Remove duplicates
        unique_topics = list(set(topic for topic, _ in topics))
        
        # Create topic-based examples
        for topic in unique_topics[:5]:  # Limit to 5 topics
            # Find all claims related to this topic
            related_claims = [claim for t, claim in topics if t == topic]
            
            if related_claims:
                instruction = f"What do you know about {topic}?"
                response = "; ".join(related_claims)
                
                examples.append(TrainingExample(
                    instruction=instruction,
                    response=response,
                    source_conversation_id=conversation_id,
                    example_type="topic"
                ))
        
        return examples
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """
        Extract main topic from text.
        
        Args:
            text: Text to extract topic from
            
        Returns:
            Main topic or None
        """
        # Simple topic extraction using common patterns
        text = text.lower()
        
        # Look for "about X" patterns
        about_match = re.search(r'\babout\s+(\w+(?:\s+\w+)?)', text)
        if about_match:
            return about_match.group(1)
        
        # Look for technology/tool mentions
        tech_keywords = [
            'python', 'javascript', 'react', 'django', 'flask', 'fastapi',
            'postgresql', 'mongodb', 'redis', 'docker', 'kubernetes',
            'machine learning', 'ai', 'deep learning', 'neural networks',
            'authentication', 'database', 'api', 'frontend', 'backend'
        ]
        
        for keyword in tech_keywords:
            if keyword in text:
                return keyword
        
        # Extract first significant noun phrase
        words = text.split()
        for i, word in enumerate(words):
            if len(word) > 3 and word.isalpha():
                # Get up to 2 words for compound topics
                if i < len(words) - 1 and len(words[i + 1]) > 3:
                    return f"{word} {words[i + 1]}"
                return word
        
        return None
    
    def _add_paraphrases(
        self,
        examples: List[TrainingExample],
        conversation_id: str
    ) -> List[TrainingExample]:
        """
        Add paraphrased versions of instructions.
        
        Args:
            examples: Original training examples
            conversation_id: Source conversation ID
            
        Returns:
            List of paraphrased examples
        """
        paraphrases = []
        
        # Paraphrase templates
        paraphrase_map = {
            "What do you remember about": [
                "Can you recall anything about",
                "Do you remember discussing",
                "What did we talk about regarding"
            ],
            "What did we discuss": [
                "What did we talk about",
                "What was mentioned about",
                "Can you tell me about"
            ],
            "Can you summarize": [
                "Can you recap",
                "Please summarize",
                "Give me a summary of"
            ],
            "What did I say about": [
                "What was my response about",
                "How did I respond regarding",
                "What did I mention about"
            ]
        }
        
        for example in examples:
            # Only paraphrase some examples to avoid too many
            if random.random() > 0.3:  # 30% chance to paraphrase
                continue
            
            original_instruction = example.instruction
            paraphrased_instruction = original_instruction
            
            # Apply paraphrase templates
            for original_phrase, paraphrase_options in paraphrase_map.items():
                if original_phrase in original_instruction:
                    replacement = random.choice(paraphrase_options)
                    paraphrased_instruction = original_instruction.replace(
                        original_phrase, replacement, 1
                    )
                    break
            
            # Only add if actually different
            if paraphrased_instruction != original_instruction:
                paraphrase = TrainingExample(
                    instruction=paraphrased_instruction,
                    response=example.response,
                    source_conversation_id=conversation_id,
                    example_type=f"{example.example_type}_paraphrase"
                )
                paraphrases.append(paraphrase)
        
        return paraphrases
    
    async def save_training_data(
        self,
        training_examples: List[TrainingExample],
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Save training data to file.
        
        Args:
            training_examples: Training examples to save
            output_path: Optional custom output path
            
        Returns:
            Path to saved training data
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_training_{timestamp}.jsonl"
            output_path = self._training_data_dir / filename
        
        try:
            # Save in JSONL format for training
            with open(output_path, 'w', encoding='utf-8') as f:
                for example in training_examples:
                    # Format for training
                    training_format = {
                        "instruction": example.instruction,
                        "response": example.response,
                        "conversation_id": example.source_conversation_id,
                        "example_type": example.example_type,
                    }
                    f.write(json.dumps(training_format, ensure_ascii=False) + '\n')
            
            logger.info(
                f"Saved {len(training_examples)} training examples to {output_path}"
            )
            
            return output_path
            
        except Exception as e:
            error_msg = f"Failed to save training data: {str(e)}"
            logger.error(error_msg)
            raise ConversationProcessingError(
                message=error_msg,
                processing_stage="data_saving",
                details={"output_path": str(output_path)}
            )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing stats
        """
        return {
            "processed_conversations": self._processed_conversations,
            "generated_examples": self._generated_examples,
            "average_examples_per_conversation": (
                self._generated_examples / max(1, self._processed_conversations)
            ),
            "min_turns_required": self._min_turns,
            "max_examples_per_conversation": self._max_examples,
            "paraphrases_enabled": self._include_paraphrases,
            "training_data_dir": str(self._training_data_dir),
        }


# Convenience function for creating processor
def create_conversation_processor(
    settings: Optional[Settings] = None
) -> ConversationProcessor:
    """
    Create and return a ConversationProcessor instance.
    
    Args:
        settings: Application settings
        
    Returns:
        ConversationProcessor instance
    """
    return ConversationProcessor(settings=settings)


# Export main classes and functions
__all__ = [
    "ConversationProcessor",
    "create_conversation_processor",
]