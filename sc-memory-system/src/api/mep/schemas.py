"""
Request and response schemas for Memory Exchange Protocol (MEP) API.

This module defines the Pydantic V2 schemas for MEP API endpoints,
including request validation, response formatting, and error handling.
All schemas follow the MEP specification exactly.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ...core.models import (
    TokenUsageInfo,
    MessageSpan,
    KeyFact,
    MEPProposal,
    MEPProposalResponse,
    ErrorDetail,
)


class MEPProposalRequest(BaseModel):
    """Request schema for MEP proposal submission."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid',
        str_strip_whitespace=True
    )
    
    # Provider and model information
    provider: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="LLM provider name (e.g., 'anthropic', 'openai')",
        examples=["anthropic", "openai", "cohere"]
    )
    
    model: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Model identifier (e.g., 'claude-3-sonnet', 'gpt-4')",
        examples=["claude-3-sonnet", "gpt-4-turbo", "command-r-plus"]
    )
    
    # User and conversation identification
    external_user_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="External user identifier",
        examples=["user_12345", "customer_abc123"]
    )
    
    external_chat_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="External chat/conversation identifier",
        examples=["chat_67890", "conversation_xyz789"]
    )
    
    # Event and processing information
    event_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique event identifier",
        examples=["evt_20240901_001", "memory_trigger_abc"]
    )
    
    trigger: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="What triggered this memory consolidation",
        examples=["context_full", "user_request", "scheduled", "token_limit"]
    )
    
    context_fill: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Context window fill ratio (0.0 to 1.0)",
        examples=[0.75, 0.9, 0.85]
    )
    
    # Token usage and span information
    token_usage: TokenUsageInfo = Field(
        ...,
        description="Token usage information for the conversation"
    )
    
    message_span: MessageSpan = Field(
        ...,
        description="Range of messages to be consolidated"
    )
    
    # Content and facts
    summary_text: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Summary of the conversation span being consolidated",
        examples=[
            "User discussed their project requirements for a web application with authentication and database integration."
        ]
    )
    
    key_facts: List[KeyFact] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Key facts extracted from the conversation",
        examples=[
            [
                {
                    "claim": "User prefers React for frontend development",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "category": "preference"
                }
            ]
        ]
    )
    
    # Optional provider-specific data
    provider_embeddings: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Provider-specific embedding data (optional)"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata (optional)"
    )
    
    # Timestamp (auto-generated if not provided)
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Proposal timestamp (auto-generated if not provided)"
    )
    
    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider name format."""
        v = v.lower().strip()
        valid_providers = {
            'anthropic', 'openai', 'cohere', 'mistral', 'google', 
            'huggingface', 'together', 'replicate', 'fireworks'
        }
        
        if v not in valid_providers:
            # Allow custom providers but warn
            pass  # For MVP, we'll be permissive
        
        return v
    
    @field_validator('trigger')
    @classmethod
    def validate_trigger(cls, v: str) -> str:
        """Validate trigger type."""
        v = v.lower().strip()
        valid_triggers = {
            'context_full', 'token_limit', 'user_request', 'scheduled',
            'memory_pressure', 'session_end', 'manual', 'automatic'
        }
        
        if v not in valid_triggers:
            # Allow custom triggers but validate format
            if not v.replace('_', '').replace('-', '').isalnum():
                raise ValueError(f"Invalid trigger format: {v}")
        
        return v
    
    @field_validator('timestamp', mode='before')
    @classmethod
    def set_default_timestamp(cls, v: Optional[datetime]) -> datetime:
        """Set default timestamp if not provided."""
        return v or datetime.utcnow()
    
    def to_mep_proposal(self) -> MEPProposal:
        """Convert request to internal MEPProposal model."""
        return MEPProposal(
            provider=self.provider,
            model=self.model,
            external_user_id=self.external_user_id,
            external_chat_id=self.external_chat_id,
            event_id=self.event_id,
            trigger=self.trigger,
            context_fill=self.context_fill,
            token_usage=self.token_usage,
            message_span=self.message_span,
            summary_text=self.summary_text,
            key_facts=self.key_facts,
            provider_embeddings=self.provider_embeddings,
            metadata=self.metadata,
            timestamp=self.timestamp
        )


class MEPProposalSuccessResponse(BaseModel):
    """Success response for MEP proposal submission."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    proposal_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for the proposal",
        examples=["f47ac10b-58cc-4372-a567-0e02b2c3d479"]
    )
    
    status: str = Field(
        default="accepted",
        description="Proposal processing status",
        examples=["accepted", "queued", "processing"]
    )
    
    message: str = Field(
        default="Proposal accepted for processing",
        description="Human-readable status message",
        examples=[
            "Proposal accepted for processing",
            "Proposal queued for background processing",
            "Proposal validation successful"
        ]
    )
    
    queue_position: Optional[int] = Field(
        default=None,
        ge=1,
        description="Position in processing queue (if queued)",
        examples=[5, 12, 1]
    )
    
    estimated_processing_time: Optional[int] = Field(
        default=None,
        ge=0,
        description="Estimated processing time in seconds",
        examples=[30, 120, 300]
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    
    # Additional response metadata
    api_version: str = Field(
        default="v1",
        description="MEP API version",
        examples=["v1"]
    )
    
    server_id: Optional[str] = Field(
        default=None,
        description="Server instance identifier (for debugging)",
        examples=["sc-memory-01", "pod-xyz-123"]
    )


class MEPProposalErrorResponse(BaseModel):
    """Error response for MEP proposal submission."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    error: ErrorDetail = Field(
        ...,
        description="Detailed error information"
    )
    
    proposal_id: Optional[str] = Field(
        default=None,
        description="Proposal ID if generated before error"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    
    api_version: str = Field(
        default="v1",
        description="MEP API version"
    )


class MEPHealthResponse(BaseModel):
    """Health check response for MEP API."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    status: str = Field(
        default="healthy",
        description="Overall health status",
        examples=["healthy", "degraded", "unhealthy"]
    )
    
    version: str = Field(
        default="0.1.0",
        description="API version"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )
    
    uptime_seconds: float = Field(
        ...,
        ge=0.0,
        description="API uptime in seconds"
    )
    
    # Component health
    components: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual component health status",
        examples=[{
            "base_model": "healthy",
            "embeddings": "healthy", 
            "vector_store": "healthy",
            "queue": "healthy"
        }]
    )
    
    # Performance metrics
    metrics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Performance metrics",
        examples=[{
            "queue_size": 5,
            "active_requests": 2,
            "memory_usage_mb": 1024.5,
            "cpu_usage_percent": 45.2
        }]
    )


class MEPValidationErrorResponse(BaseModel):
    """Validation error response with field-specific errors."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    error_type: str = Field(
        default="validation_error",
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="General error message"
    )
    
    details: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Field-specific validation errors",
        examples=[
            [
                {
                    "field": "context_fill",
                    "message": "ensure this value is less than or equal to 1.0",
                    "input": 1.5,
                    "type": "less_than_equal"
                },
                {
                    "field": "key_facts",
                    "message": "ensure this value has at least 1 items",
                    "input": [],
                    "type": "too_short"
                }
            ]
        ]
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )


class MEPBatchProposalRequest(BaseModel):
    """Request schema for batch MEP proposal submission."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    proposals: List[MEPProposalRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of MEP proposals to submit"
    )
    
    batch_id: Optional[str] = Field(
        default_factory=lambda: str(uuid4()),
        description="Batch identifier for tracking"
    )
    
    priority: Optional[int] = Field(
        default=0,
        ge=0,
        le=10,
        description="Batch processing priority (0=low, 10=high)"
    )


class MEPBatchProposalResponse(BaseModel):
    """Response schema for batch MEP proposal submission."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    batch_id: str = Field(
        ...,
        description="Batch identifier"
    )
    
    total_proposals: int = Field(
        ...,
        ge=0,
        description="Total number of proposals in batch"
    )
    
    accepted_proposals: int = Field(
        ...,
        ge=0,
        description="Number of proposals accepted"
    )
    
    rejected_proposals: int = Field(
        ...,
        ge=0,
        description="Number of proposals rejected"
    )
    
    results: List[Dict[str, Any]] = Field(
        ...,
        description="Individual proposal results"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Batch processing timestamp"
    )


# Export all schemas
__all__ = [
    "MEPProposalRequest",
    "MEPProposalSuccessResponse", 
    "MEPProposalErrorResponse",
    "MEPHealthResponse",
    "MEPValidationErrorResponse",
    "MEPBatchProposalRequest",
    "MEPBatchProposalResponse",
]