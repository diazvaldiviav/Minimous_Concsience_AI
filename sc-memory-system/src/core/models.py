"""
Core Pydantic models for SC Memory System.

This module defines the data models used throughout the application,
including MEP (Memory Exchange Protocol) schemas, vector search results,
and internal data structures. All models use Pydantic V2 for validation
and serialization.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, computed_field


class TokenUsageInfo(BaseModel):
    """Token usage information for memory consolidation proposals."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    window_tokens: int = Field(
        ...,
        ge=0,
        le=1000000,
        description="Total tokens in the conversation window"
    )
    used_tokens: int = Field(
        ...,
        ge=0,
        description="Number of tokens currently used"
    )
    max_tokens: int = Field(
        ...,
        ge=1,
        le=1000000,
        description="Maximum tokens allowed in the window"
    )
    
    @field_validator('used_tokens')
    @classmethod
    def validate_used_tokens(cls, v: int, info) -> int:
        """Ensure used_tokens doesn't exceed window_tokens."""
        if 'window_tokens' in info.data and v > info.data['window_tokens']:
            raise ValueError("used_tokens cannot exceed window_tokens")
        return v
    
    @computed_field
    @property
    def utilization_ratio(self) -> float:
        """Calculate token utilization ratio."""
        return self.used_tokens / self.window_tokens if self.window_tokens > 0 else 0.0
    
    @computed_field
    @property
    def capacity_ratio(self) -> float:
        """Calculate capacity utilization ratio."""
        return self.window_tokens / self.max_tokens if self.max_tokens > 0 else 0.0


class MessageSpan(BaseModel):
    """Represents a range of messages in a conversation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    from_turn: int = Field(
        ...,
        ge=0,
        description="Starting turn number (inclusive)"
    )
    to_turn: int = Field(
        ...,
        ge=0,
        description="Ending turn number (inclusive)"
    )
    
    @field_validator('to_turn')
    @classmethod
    def validate_turn_order(cls, v: int, info) -> int:
        """Ensure to_turn is greater than or equal to from_turn."""
        if 'from_turn' in info.data and v < info.data['from_turn']:
            raise ValueError("to_turn must be greater than or equal to from_turn")
        return v
    
    @computed_field
    @property
    def span_length(self) -> int:
        """Calculate the number of turns in this span."""
        return self.to_turn - self.from_turn + 1


class KeyFact(BaseModel):
    """A key fact extracted from conversation for memory consolidation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    claim: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="The factual claim or key information"
    )
    importance: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Importance score from 0.0 to 1.0"
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score for this fact"
    )
    source_turn: Optional[int] = Field(
        default=None,
        ge=0,
        description="Turn number where this fact originated"
    )
    category: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Category or type of the fact"
    )


class MEPProposal(BaseModel):
    """Memory Exchange Protocol proposal for memory consolidation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    # Provider and model information
    provider: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="LLM provider name (e.g., 'anthropic', 'openai')"
    )
    model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Model identifier (e.g., 'claude-3-sonnet', 'gpt-4')"
    )
    
    # User and conversation identification
    external_user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="External user identifier"
    )
    external_chat_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="External chat/conversation identifier"
    )
    
    # Event and processing information
    event_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique event identifier"
    )
    trigger: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="What triggered this memory consolidation"
    )
    context_fill: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context window fill ratio"
    )
    
    # Token usage and span information
    token_usage: TokenUsageInfo = Field(
        ...,
        description="Token usage information"
    )
    message_span: MessageSpan = Field(
        ...,
        description="Range of messages to consolidate"
    )
    
    # Content and facts
    summary_text: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Summary of the conversation span"
    )
    key_facts: List[KeyFact] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Key facts extracted from the conversation"
    )
    
    # Optional provider-specific data
    provider_embeddings: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific embedding data"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )
    
    # Timestamps
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Proposal creation timestamp"
    )
    
    @computed_field
    @property
    def total_facts(self) -> int:
        """Get total number of key facts."""
        return len(self.key_facts)
    
    @computed_field
    @property
    def average_fact_importance(self) -> float:
        """Calculate average importance score of key facts."""
        if not self.key_facts:
            return 0.0
        return sum(fact.importance for fact in self.key_facts) / len(self.key_facts)
    
    @computed_field
    @property
    def high_importance_facts(self) -> List[KeyFact]:
        """Get facts with importance >= 0.7."""
        return [fact for fact in self.key_facts if fact.importance >= 0.7]


class MEPProposalResponse(BaseModel):
    """Response model for MEP proposal submission."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    proposal_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the proposal"
    )
    status: str = Field(
        default="accepted",
        description="Proposal processing status"
    )
    message: str = Field(
        default="Proposal accepted for processing",
        description="Status message"
    )
    queue_position: Optional[int] = Field(
        default=None,
        ge=1,
        description="Position in processing queue"
    )
    estimated_processing_time: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated processing time in seconds"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


class EmbeddingResult(BaseModel):
    """Result of embedding generation for text content."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    text: str = Field(..., description="Original text that was embedded")
    embedding: List[float] = Field(
        ...,
        min_length=64,
        max_length=4096,
        description="Generated embedding vector"
    )
    model_name: str = Field(..., description="Name of the embeddings model used")
    dimension: int = Field(..., ge=64, le=4096, description="Embedding dimension")
    processing_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time taken to generate embedding in seconds"
    )
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimension(cls, v: List[float], info) -> List[float]:
        """Ensure embedding dimension matches the declared dimension."""
        if 'dimension' in info.data and len(v) != info.data['dimension']:
            raise ValueError(f"Embedding length {len(v)} does not match dimension {info.data['dimension']}")
        return v


class VectorSearchResult(BaseModel):
    """Result of vector similarity search."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    id: str = Field(..., description="Unique identifier for the result")
    score: float = Field(
        ...,
        ge=0.0,
        description="Similarity score (lower is more similar for L2 distance)"
    )
    text: str = Field(..., description="Original text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Embedding vector (optional)"
    )


class VectorSearchQuery(BaseModel):
    """Query for vector similarity search."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    query_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text to search for"
    )
    k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum similarity score threshold"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters to apply"
    )
    include_embeddings: bool = Field(
        default=False,
        description="Whether to include embeddings in results"
    )


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    name: str = Field(..., description="Model name or identifier")
    type: str = Field(..., description="Model type (e.g., 'language_model', 'embeddings')")
    device: str = Field(..., description="Device where model is loaded")
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Approximate memory usage in MB"
    )
    parameters_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of parameters in the model"
    )
    is_loaded: bool = Field(default=False, description="Whether model is loaded")
    load_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Time taken to load model in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional model metadata"
    )


class HealthStatus(BaseModel):
    """Health status of the application and its components."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="Application uptime in seconds"
    )
    
    # Component statuses
    base_model_healthy: bool = Field(..., description="Base model health")
    embeddings_healthy: bool = Field(..., description="Embeddings model health")
    vector_store_healthy: bool = Field(..., description="Vector store health")
    
    # Performance metrics
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Current memory usage in MB"
    )
    cpu_usage_percent: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Current CPU usage percentage"
    )
    
    # Queue and processing info
    queue_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Current MEP proposal queue size"
    )
    active_requests: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of active requests"
    )


class ErrorDetail(BaseModel):
    """Detailed error information for API responses."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    error_code: str = Field(..., description="Specific error code")
    error_type: str = Field(..., description="Error category or type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error occurrence timestamp"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Associated request identifier"
    )


class ValidatedFact(BaseModel):
    """A key fact that has been validated by the truth model."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    original_fact: KeyFact = Field(..., description="Original key fact")
    truth_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Truth confidence score"
    )
    is_validated: bool = Field(..., description="Whether fact passed validation")
    validation_reason: Optional[str] = Field(
        default=None,
        description="Reason for validation result"
    )


class TrainingExample(BaseModel):
    """A single instruction-response pair for LoRA training."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    instruction: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Training instruction"
    )
    response: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Expected response"
    )
    source_conversation_id: str = Field(
        ...,
        description="ID of source conversation"
    )
    example_type: str = Field(
        ...,
        description="Type of example (recall, summary, fact_check)"
    )


class ConversationAdapter(BaseModel):
    """Metadata for a trained conversation adapter."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    adapter_id: str = Field(..., description="Unique adapter identifier")
    conversation_id: str = Field(..., description="Source conversation ID")
    source_proposal_id: str = Field(..., description="MEP proposal ID")
    adapter_path: Path = Field(..., description="Path to adapter files")
    training_examples_count: int = Field(
        ...,
        ge=0,
        description="Number of training examples used"
    )
    training_date: datetime = Field(
        default_factory=datetime.utcnow,
        description="When adapter was trained"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Training performance metrics"
    )


class ConsolidationResult(BaseModel):
    """Result of conversation consolidation process."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    adapter_id: str = Field(..., description="Generated adapter ID")
    status: str = Field(
        ...,
        description="Consolidation status (success, failed, partial)"
    )
    validated_facts_count: int = Field(
        ...,
        ge=0,
        description="Number of facts that passed validation"
    )
    training_examples_count: int = Field(
        ...,
        ge=0,
        description="Number of training examples generated"
    )
    training_time_seconds: float = Field(
        ...,
        ge=0.0,
        description="Time taken for training in seconds"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if consolidation failed"
    )


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    role: str = Field(..., description="Speaker role (user, assistant)")
    content: str = Field(..., description="Turn content")
    turn_index: int = Field(..., ge=0, description="Turn index in conversation")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional turn metadata"
    )


# Export all models for easy import
__all__ = [
    "TokenUsageInfo",
    "MessageSpan", 
    "KeyFact",
    "MEPProposal",
    "MEPProposalResponse",
    "EmbeddingResult",
    "VectorSearchResult",
    "VectorSearchQuery",
    "ModelInfo",
    "HealthStatus",
    "ErrorDetail",
    "ValidatedFact",
    "TrainingExample",
    "ConversationAdapter",
    "ConsolidationResult",
    "ConversationTurn",
]