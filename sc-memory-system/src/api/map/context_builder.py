"""
ContextBuilder for compressed memory context generation.

This module provides functionality for generating compressed memory context
from LoRA adapters, including GIST generation, turn extraction, and fact
compilation within token budgets.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import torch
from transformers import AutoTokenizer
from peft import PeftModel

from src.core.config import Settings
from src.core.models import (
    CompressedTurn, FilteredFact, TokenAllocation, AdapterInfo,
    KeyFact, ValidatedFact, ConversationTurn, MAPQuery
)
from src.core.exceptions import ModelLoadError, ValidationError

logger = logging.getLogger(__name__)


class TokenBudgetManager:
    """Manages token budget allocation for MAP responses."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Initialize token budget manager.
        
        Args:
            tokenizer: Tokenizer for token counting
        """
        self.tokenizer = tokenizer
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        try:
            if not text:
                return 0
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Fallback: rough estimation (4 chars per token)
            return len(text) // 4
    
    def allocate_tokens(
        self,
        total_budget: int,
        gist_ratio: float = 0.4,
        turns_ratio: float = 0.4,
        facts_ratio: float = 0.15,
        metadata_ratio: float = 0.05
    ) -> TokenAllocation:
        """
        Allocate tokens across response components.
        
        Args:
            total_budget: Total token budget
            gist_ratio: Ratio for gist content
            turns_ratio: Ratio for turns content
            facts_ratio: Ratio for facts content
            metadata_ratio: Ratio for metadata
            
        Returns:
            TokenAllocation object
        """
        try:
            # Ensure ratios sum to 1.0
            total_ratio = gist_ratio + turns_ratio + facts_ratio + metadata_ratio
            if abs(total_ratio - 1.0) > 0.01:
                # Normalize ratios
                gist_ratio /= total_ratio
                turns_ratio /= total_ratio
                facts_ratio /= total_ratio
                metadata_ratio /= total_ratio
            
            gist_tokens = int(total_budget * gist_ratio)
            turns_tokens = int(total_budget * turns_ratio)
            facts_tokens = int(total_budget * facts_ratio)
            metadata_tokens = int(total_budget * metadata_ratio)
            
            # Adjust for rounding
            total_allocated = gist_tokens + turns_tokens + facts_tokens + metadata_tokens
            if total_allocated < total_budget:
                # Add remaining to gist (most flexible)
                gist_tokens += (total_budget - total_allocated)
            elif total_allocated > total_budget:
                # Remove excess from gist
                gist_tokens -= (total_allocated - total_budget)
                gist_tokens = max(gist_tokens, 10)  # Minimum gist size
            
            return TokenAllocation(
                gist_tokens=gist_tokens,
                turns_tokens=turns_tokens,
                facts_tokens=facts_tokens,
                metadata_tokens=metadata_tokens,
                total_allocated=gist_tokens + turns_tokens + facts_tokens + metadata_tokens,
                total_budget=total_budget
            )
            
        except Exception as e:
            logger.error(f"Token allocation failed: {e}")
            # Fallback allocation
            quarter = total_budget // 4
            return TokenAllocation(
                gist_tokens=quarter * 2,
                turns_tokens=quarter,
                facts_tokens=quarter // 2,
                metadata_tokens=quarter // 2,
                total_allocated=total_budget,
                total_budget=total_budget
            )
    
    def compress_to_fit(self, content: str, max_tokens: int) -> str:
        """
        Compress content to fit within token budget.
        
        Args:
            content: Content to compress
            max_tokens: Maximum tokens allowed
            
        Returns:
            Compressed content
        """
        try:
            if not content:
                return ""
            
            current_tokens = self.count_tokens(content)
            if current_tokens <= max_tokens:
                return content
            
            # Calculate compression ratio
            target_ratio = max_tokens / current_tokens
            target_length = int(len(content) * target_ratio * 0.9)  # 10% buffer
            
            # Smart truncation - try to preserve sentence boundaries
            if target_length >= len(content):
                return content
            
            # Find good break point near target
            sentences = re.split(r'[.!?]\s+', content)
            if len(sentences) > 1:
                # Truncate by sentences
                compressed = ""
                for sentence in sentences:
                    test_content = compressed + sentence + ". "
                    if self.count_tokens(test_content) > max_tokens:
                        break
                    compressed = test_content
                
                if compressed:
                    return compressed.rstrip()
            
            # Fallback: character truncation with ellipsis
            ellipsis = "..."
            ellipsis_tokens = self.count_tokens(ellipsis)
            available_tokens = max_tokens - ellipsis_tokens
            
            # Binary search for optimal length
            low, high = 0, len(content)
            best_length = 0
            
            while low <= high:
                mid = (low + high) // 2
                test_content = content[:mid]
                
                if self.count_tokens(test_content) <= available_tokens:
                    best_length = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            return content[:best_length].rstrip() + ellipsis
            
        except Exception as e:
            logger.error(f"Content compression failed: {e}")
            # Emergency fallback - rough character truncation
            target_chars = max_tokens * 3  # Rough estimate
            return content[:target_chars].rstrip() + "..." if len(content) > target_chars else content


class ContextBuilder:
    """Builds compressed memory context from adapters."""
    
    def __init__(self, settings: Settings):
        """
        Initialize context builder.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.tokenizer: Optional[AutoTokenizer] = None
        self.token_budget_manager: Optional[TokenBudgetManager] = None
    
    async def initialize(self) -> None:
        """Initialize the context builder."""
        try:
            logger.info("Initializing ContextBuilder")
            
            # Load tokenizer
            model_name = getattr(self.settings.model, 'name', 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Ensure pad token exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.token_budget_manager = TokenBudgetManager(self.tokenizer)
            
            logger.info("ContextBuilder initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ContextBuilder: {e}", exc_info=True)
            raise ModelLoadError(f"ContextBuilder initialization failed: {e}")
    
    async def generate_gist(
        self,
        query: str,
        model: PeftModel,
        adapter_info: AdapterInfo,
        max_tokens: int
    ) -> str:
        """
        Generate compressed conversation gist.
        
        Args:
            query: User query for context
            model: Loaded adapter model
            adapter_info: Adapter information
            max_tokens: Maximum tokens for gist
            
        Returns:
            Compressed gist string
        """
        try:
            if max_tokens <= 0:
                return ""
            
            logger.debug(f"Generating gist for query: '{query[:50]}...' with {max_tokens} tokens")
            
            # Create prompt for gist generation
            conversation_topic = adapter_info.topic or "conversation"
            turn_info = adapter_info.turn_range
            turn_span = ""
            if turn_info and 'from_turn' in turn_info and 'to_turn' in turn_info:
                turn_span = f" (turns {turn_info['from_turn']}-{turn_info['to_turn']})"
            
            prompt = f"""Summarize the key points from our {conversation_topic} discussion{turn_span} that are relevant to: {query}

Provide a concise summary focusing on the most important concepts, conclusions, and relevant details."""
            
            # Generate with model
            try:
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                if inputs.size(1) > 512:  # Truncate if too long
                    inputs = inputs[:, :512]
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=inputs.size(1) + max_tokens,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                
                # Decode and extract response
                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the generated part (remove prompt)
                prompt_decoded = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
                if full_response.startswith(prompt_decoded):
                    gist = full_response[len(prompt_decoded):].strip()
                else:
                    gist = full_response.strip()
                
            except Exception as e:
                logger.warning(f"Model generation failed, using fallback: {e}")
                # Fallback gist generation
                gist = self._generate_fallback_gist(adapter_info, query)
            
            # Ensure gist fits within token budget
            gist = self.token_budget_manager.compress_to_fit(gist, max_tokens)
            
            logger.debug(f"Generated gist: '{gist[:100]}...'")
            return gist
            
        except Exception as e:
            logger.error(f"Gist generation failed: {e}", exc_info=True)
            return self._generate_fallback_gist(adapter_info, query)
    
    def _generate_fallback_gist(self, adapter_info: AdapterInfo, query: str) -> str:
        """Generate fallback gist when model generation fails."""
        try:
            topic = adapter_info.topic or "conversation"
            turn_range = adapter_info.turn_range
            
            if turn_range and 'from_turn' in turn_range and 'to_turn' in turn_range:
                span_text = f"turns {turn_range['from_turn']}-{turn_range['to_turn']}"
            else:
                span_text = "conversation"
            
            gist = f"Previous {topic} discussion ({span_text}) covering relevant topics related to: {query}"
            
            return gist
            
        except Exception:
            return f"Previous conversation relevant to: {query}"
    
    async def extract_turn_sketch(
        self,
        conversation_turns: List[ConversationTurn],
        max_tokens: int,
        query: str = ""
    ) -> List[CompressedTurn]:
        """
        Extract compressed turn sketches from conversation.
        
        Args:
            conversation_turns: List of conversation turns
            max_tokens: Maximum tokens for all turns
            query: Optional query for relevance filtering
            
        Returns:
            List of compressed turns
        """
        try:
            if max_tokens <= 0:
                return []
            
            logger.debug(f"Extracting turn sketches with {max_tokens} tokens")
            
            if not conversation_turns:
                return []
            
            # Sort turns by relevance to query if provided
            if query:
                conversation_turns = await self._rank_turns_by_relevance(conversation_turns, query)
            
            compressed_turns = []
            used_tokens = 0
            
            # Estimate tokens per turn (leaving buffer for metadata)
            estimated_turns = min(len(conversation_turns), max_tokens // 20)  # ~20 tokens per turn
            tokens_per_turn = max_tokens // max(estimated_turns, 1)
            
            for turn in conversation_turns:
                if used_tokens >= max_tokens:
                    break
                
                # Determine turn role
                role = "u" if turn.role == "user" else "a"
                
                # Compress turn content
                remaining_tokens = max_tokens - used_tokens
                turn_budget = min(tokens_per_turn, remaining_tokens - 10)  # Leave buffer
                
                if turn_budget <= 0:
                    break
                
                compressed_content = self.token_budget_manager.compress_to_fit(
                    turn.content, 
                    turn_budget
                )
                
                compressed_turn = CompressedTurn(
                    id=f"T{turn.turn_number}{role.upper()}",
                    r=role,
                    t=compressed_content
                )
                
                compressed_turns.append(compressed_turn)
                
                # Update token usage
                turn_tokens = self.token_budget_manager.count_tokens(compressed_content) + 10  # Metadata overhead
                used_tokens += turn_tokens
            
            logger.debug(f"Extracted {len(compressed_turns)} compressed turns using {used_tokens} tokens")
            return compressed_turns
            
        except Exception as e:
            logger.error(f"Turn extraction failed: {e}", exc_info=True)
            return []
    
    async def _rank_turns_by_relevance(
        self,
        turns: List[ConversationTurn],
        query: str
    ) -> List[ConversationTurn]:
        """Rank turns by relevance to query (simple text similarity)."""
        try:
            if not query or not turns:
                return turns
            
            query_lower = query.lower()
            query_words = set(query_lower.split())
            
            def relevance_score(turn: ConversationTurn) -> float:
                content_lower = turn.content.lower()
                content_words = set(content_lower.split())
                
                # Jaccard similarity
                intersection = len(query_words & content_words)
                union = len(query_words | content_words)
                
                return intersection / union if union > 0 else 0.0
            
            # Sort by relevance (descending)
            ranked_turns = sorted(turns, key=relevance_score, reverse=True)
            return ranked_turns
            
        except Exception as e:
            logger.error(f"Turn ranking failed: {e}", exc_info=True)
            return turns
    
    async def compile_facts(
        self,
        validated_facts: List[ValidatedFact],
        min_truth: float,
        max_tokens: int
    ) -> List[FilteredFact]:
        """
        Compile and filter facts based on truth score and token budget.
        
        Args:
            validated_facts: List of validated facts
            min_truth: Minimum truth score threshold
            max_tokens: Maximum tokens for all facts
            
        Returns:
            List of filtered facts
        """
        try:
            if max_tokens <= 0:
                return []
            
            logger.debug(f"Compiling facts with min_truth={min_truth} and {max_tokens} tokens")
            
            # Filter facts by truth score
            filtered_facts = [
                fact for fact in validated_facts
                if fact.confidence >= min_truth
            ]
            
            # Sort by confidence (descending)
            filtered_facts.sort(key=lambda f: f.confidence, reverse=True)
            
            compiled_facts = []
            used_tokens = 0
            
            for fact in filtered_facts:
                if used_tokens >= max_tokens:
                    break
                
                # Create filtered fact
                claim = fact.claim
                remaining_tokens = max_tokens - used_tokens
                
                # Compress claim if needed
                claim_budget = min(50, remaining_tokens)  # Max 50 tokens per fact
                compressed_claim = self.token_budget_manager.compress_to_fit(
                    claim, 
                    claim_budget
                )
                
                if not compressed_claim:
                    continue
                
                filtered_fact = FilteredFact(
                    c=compressed_claim,
                    p=round(fact.confidence, 2),
                    s=fact.source or "mem"
                )
                
                compiled_facts.append(filtered_fact)
                
                # Update token usage (approximate)
                fact_tokens = self.token_budget_manager.count_tokens(compressed_claim) + 5  # Metadata overhead
                used_tokens += fact_tokens
            
            logger.debug(f"Compiled {len(compiled_facts)} facts using {used_tokens} tokens")
            return compiled_facts
            
        except Exception as e:
            logger.error(f"Fact compilation failed: {e}", exc_info=True)
            return []
    
    def estimate_response_tokens(
        self,
        gist: str,
        turns: List[CompressedTurn],
        facts: List[FilteredFact]
    ) -> int:
        """Estimate total response tokens."""
        try:
            total_tokens = 0
            
            # Gist tokens
            total_tokens += self.token_budget_manager.count_tokens(gist)
            
            # Turns tokens
            for turn in turns:
                total_tokens += self.token_budget_manager.count_tokens(turn.t)
                total_tokens += 5  # Metadata overhead per turn
            
            # Facts tokens
            for fact in facts:
                total_tokens += self.token_budget_manager.count_tokens(fact.c)
                total_tokens += 3  # Metadata overhead per fact
            
            # JSON structure overhead
            total_tokens += 20
            
            return total_tokens
            
        except Exception as e:
            logger.error(f"Token estimation failed: {e}")
            return 0