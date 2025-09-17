"""
Core configuration management for SC Memory System.

This module provides centralized configuration management using Pydantic V2 settings
with environment variable support and validation. Supports development, testing,
and production environments with appropriate defaults and security considerations.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for ML models and inference settings."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    # Base Model Settings (TinyLlama-1.1B)
    name: str = Field(
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        description="HuggingFace model name or path"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path("./models/cache"),
        description="Directory to cache downloaded models"
    )
    device: str = Field(
        default="auto",
        description="Device for model inference (auto, cpu, cuda, cuda:0, etc.)"
    )
    max_length: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum sequence length for tokenization"
    )
    torch_dtype: str = Field(
        default="auto",
        description="PyTorch dtype (auto, float16, float32, bfloat16)"
    )
    low_cpu_mem_usage: bool = Field(
        default=True,
        description="Use low CPU memory loading optimization"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code execution (security risk)"
    )
    use_safetensors: bool = Field(
        default=True,
        description="Use safetensors format when available"
    )
    
    @field_validator('cache_dir')
    @classmethod
    def validate_cache_dir(cls, v: Union[str, Path]) -> Path:
        """Ensure cache directory exists and is writable."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f"Cache directory {path} is not writable")
        return path


class EmbeddingsConfig(BaseModel):
    """Configuration for embeddings generation and management."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformers model name"
    )
    cache_dir: Path = Field(
        default_factory=lambda: Path("./models/embeddings_cache"),
        description="Directory to cache embeddings model"
    )
    device: str = Field(
        default="auto",
        description="Device for embeddings computation"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embeddings generation"
    )
    max_seq_length: int = Field(
        default=384,
        ge=64,
        le=1024,
        description="Maximum sequence length for embeddings"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Normalize embeddings to unit vectors"
    )
    
    @computed_field
    @property
    def dimension(self) -> int:
        """Get embedding dimension based on model name."""
        model_dims = {
            "all-MiniLM-L6-v2": 384,
            "all-MiniLM-L12-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-roberta-large-v1": 1024,
        }
        return model_dims.get(self.model_name, 384)  # Default to 384


class VectorStoreConfig(BaseModel):
    """Configuration for FAISS vector store and similarity search."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    type: str = Field(default="faiss", description="Vector store type")
    dimension: int = Field(
        default=384,
        ge=64,
        le=4096,
        description="Vector dimension (must match embeddings)"
    )
    index_type: str = Field(
        default="IndexFlatL2",
        description="FAISS index type"
    )
    data_dir: Path = Field(
        default_factory=lambda: Path("./data/vector_store"),
        description="Directory for vector store data"
    )
    index_path: Path = Field(
        default_factory=lambda: Path("./data/vector_store/faiss.index"),
        description="Path to FAISS index file"
    )
    metadata_path: Path = Field(
        default_factory=lambda: Path("./data/vector_store/metadata.json"),
        description="Path to metadata JSON file"
    )
    
    # FAISS-specific settings
    index_factory: str = Field(
        default="Flat",
        description="FAISS index factory string"
    )
    metric_type: str = Field(
        default="METRIC_L2",
        description="Distance metric (METRIC_L2, METRIC_INNER_PRODUCT)"
    )
    nprobe: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Number of probes for IVF indexes"
    )
    m: int = Field(
        default=16,
        ge=4,
        le=64,
        description="Number of connections for HNSW indexes"
    )
    efconstruction: int = Field(
        default=200,
        ge=16,
        le=1024,
        description="efConstruction parameter for HNSW indexes"
    )
    
    @field_validator('data_dir', 'index_path', 'metadata_path')
    @classmethod
    def validate_paths(cls, v: Union[str, Path]) -> Path:
        """Ensure paths are valid and directories exist."""
        path = Path(v) if isinstance(v, str) else v
        if str(path).endswith('.index') or str(path).endswith('.json'):
            # For files, ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # For directories, ensure they exist
            path.mkdir(parents=True, exist_ok=True)
        return path


class APIConfig(BaseModel):
    """Configuration for FastAPI application and endpoints."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    host: str = Field(default="0.0.0.0", description="API host address")
    port: int = Field(default=8000, ge=1024, le=65535, description="API port")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    prefix: str = Field(default="/api/v1", description="API route prefix")
    
    # MEP (Memory Exchange Protocol) settings
    mep_prefix: str = Field(default="/mep/v1", description="MEP API prefix")
    mep_max_proposal_size: int = Field(
        default=10485760,  # 10MB
        ge=1024,
        le=104857600,  # 100MB
        description="Maximum MEP proposal size in bytes"
    )
    mep_queue_max_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum MEP proposal queue size"
    )
    mep_process_timeout: int = Field(
        default=300,  # 5 minutes
        ge=30,
        le=3600,
        description="MEP proposal processing timeout in seconds"
    )
    
    # Authentication
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        min_length=32,
        description="Secret key for JWT tokens"
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    jwt_expiration_hours: int = Field(
        default=24,
        ge=1,
        le=168,  # 7 days
        description="JWT token expiration time in hours"
    )
    bearer_token: str = Field(
        default="dev-bearer-token",
        min_length=16,
        description="Bearer token for API authentication"
    )
    
    # CORS settings
    cors_allow_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    cors_allow_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="Allowed HTTP methods for CORS"
    )


class TruthModelConfig(BaseModel):
    """Configuration for conversation truth validation."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for accepting facts"
    )
    enable_nli_validation: bool = Field(
        default=True,
        description="Enable Natural Language Inference validation"
    )
    max_facts_per_conversation: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum facts to validate per conversation"
    )
    nli_model_cache_dir: Path = Field(
        default_factory=lambda: Path("./models/nli_cache"),
        description="Directory for NLI model cache"
    )


class ConversationProcessingConfig(BaseModel):
    """Configuration for conversation-to-training-data conversion."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    min_turns_for_training: int = Field(
        default=5,
        ge=1,
        le=1000,
        description="Minimum conversation turns required for training"
    )
    max_examples_per_conversation: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum training examples per conversation"
    )
    include_paraphrases: bool = Field(
        default=True,
        description="Include paraphrased versions of examples"
    )
    training_data_dir: Path = Field(
        default_factory=lambda: Path("./data/training"),
        description="Directory for conversation training data"
    )


class LoRATrainingConfig(BaseModel):
    """Configuration for LoRA adapter training."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    r: int = Field(
        default=4,
        ge=1,
        le=16,
        description="LoRA rank parameter"
    )
    alpha: int = Field(
        default=32,
        ge=1,
        le=128,
        description="LoRA alpha parameter"
    )
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj"],
        description="Target modules for LoRA adaptation"
    )
    training_steps: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of training steps"
    )
    learning_rate: float = Field(
        default=1e-4,
        ge=1e-6,
        le=1e-2,
        description="Learning rate for training"
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Training batch size (reduced for CPU stability)"
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=8,
        description="Gradient accumulation steps"
    )
    adapters_dir: Path = Field(
        default_factory=lambda: Path("./models/adapters"),
        description="Directory for storing trained adapters"
    )


class AsyncProcessingConfig(BaseModel):
    """Configuration for Week 3 async processing pipeline."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    max_concurrent_workers: int = Field(
        default=2,
        ge=1,
        le=20,
        description="Maximum concurrent processing workers (reduced for training stability)"
    )
    queue_check_interval: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Queue checking interval in seconds"
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed proposals"
    )
    retry_delay_base: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Base delay for exponential backoff retry in seconds"
    )
    max_queue_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Maximum proposals in queue"
    )
    processing_timeout: float = Field(
        default=300.0,
        ge=30.0,
        le=3600.0,
        description="Processing timeout per proposal in seconds"
    )
    persistence_enabled: bool = Field(
        default=True,
        description="Enable proposal status persistence"
    )
    persistence_file: Path = Field(
        default_factory=lambda: Path("./data/async_processing/queue_state.json"),
        description="File for persisting processing state"
    )


class TruthFeaturesConfig(BaseModel):
    """Configuration for Week 3 truth features extraction."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    enable_rag_validation: bool = Field(
        default=True,
        description="Enable RAG-based fact validation"
    )
    enable_nli_validation: bool = Field(
        default=True,
        description="Enable NLI-based validation"
    )
    enable_provenance_tracking: bool = Field(
        default=True,
        description="Enable fact provenance tracking"
    )
    confidence_calibration_samples: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Samples for confidence calibration"
    )
    ensemble_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "rag_score": 0.4,
            "nli_score": 0.3,
            "semantic_similarity": 0.2,
            "consistency_score": 0.1
        },
        description="Weights for ensemble truth validation"
    )
    knowledge_base_path: Path = Field(
        default_factory=lambda: Path("./data/knowledge_base"),
        description="Path to knowledge base for RAG validation"
    )
    
    @field_validator('ensemble_weights')
    @classmethod
    def validate_ensemble_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure ensemble weights sum to 1.0."""
        total = sum(v.values())
        if abs(total - 1.0) > 0.01:  # Allow small floating point tolerance
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        return v


class DatasetBuilderV2Config(BaseModel):
    """Configuration for Week 3 advanced dataset building."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    enable_paraphrasing: bool = Field(
        default=True,
        description="Enable automatic paraphrasing of training examples"
    )
    enable_deduplication: bool = Field(
        default=True,
        description="Enable deduplication of similar examples"
    )
    paraphrase_techniques: List[str] = Field(
        default=["backtranslation", "synonym_replacement", "sentence_restructuring"],
        description="Paraphrasing techniques to use"
    )
    paraphrases_per_example: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Number of paraphrases per training example"
    )
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Similarity threshold for deduplication"
    )
    quality_filter_threshold: float = Field(
        default=0.7,
        ge=0.1,
        le=1.0,
        description="Quality filter threshold for training examples"
    )
    max_dataset_size: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Maximum size of generated dataset"
    )
    augmentation_cache_dir: Path = Field(
        default_factory=lambda: Path("./data/augmentation_cache"),
        description="Directory for caching augmentation results"
    )


class TrainingSchedulerConfig(BaseModel):
    """Configuration for Week 3 training scheduler."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    enable_batch_training: bool = Field(
        default=True,
        description="Enable batch training of multiple conversations"
    )
    batch_size: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of conversations to batch together"
    )
    batch_timeout: float = Field(
        default=300.0,
        ge=60.0,
        le=1800.0,
        description="Maximum wait time for batch formation in seconds"
    )
    priority_levels: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of priority levels for scheduling"
    )
    resource_monitoring_interval: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Interval for resource monitoring in seconds"
    )
    max_gpu_memory_usage: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Maximum GPU memory usage threshold"
    )
    max_cpu_usage: float = Field(
        default=0.8,
        ge=0.1,
        le=1.0,
        description="Maximum CPU usage threshold"
    )
    scheduler_persistence_file: Path = Field(
        default_factory=lambda: Path("./data/scheduler/schedule_state.json"),
        description="File for persisting scheduler state"
    )


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization and resource management."""
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True,
        extra='forbid'
    )
    
    # Memory management
    max_memory_usage_gb: float = Field(
        default=8.0,
        ge=1.0,
        le=64.0,
        description="Maximum memory usage in GB"
    )
    memory_cleanup_interval: int = Field(
        default=300,  # 5 minutes
        ge=60,
        le=3600,
        description="Memory cleanup interval in seconds"
    )
    garbage_collection_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Memory threshold for garbage collection"
    )
    
    # Concurrency settings
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum concurrent requests"
    )
    max_concurrent_model_operations: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Maximum concurrent model operations"
    )
    async_pool_size: int = Field(
        default=10,
        ge=4,
        le=50,
        description="Async executor pool size"
    )
    
    # Caching
    enable_model_caching: bool = Field(
        default=True,
        description="Enable model result caching"
    )
    cache_ttl_seconds: int = Field(
        default=3600,  # 1 hour
        ge=60,
        le=86400,
        description="Cache TTL in seconds"
    )
    cache_max_size_mb: int = Field(
        default=2048,  # 2GB
        ge=128,
        le=8192,
        description="Maximum cache size in MB"
    )


class Settings(BaseSettings):
    """Main application settings with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra='ignore'
    )
    
    # Application metadata
    app_name: str = Field(default="SC Memory System", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    app_description: str = Field(
        default="Revolutionary memory consolidation system for LLMs",
        description="Application description"
    )
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(
        default="info",
        description="Logging level (debug, info, warning, error, critical)"
    )
    
    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    truth_model: TruthModelConfig = Field(default_factory=TruthModelConfig)
    conversation_processing: ConversationProcessingConfig = Field(default_factory=ConversationProcessingConfig)
    lora_training: LoRATrainingConfig = Field(default_factory=LoRATrainingConfig)
    
    # Week 3 configurations
    async_processing: AsyncProcessingConfig = Field(default_factory=AsyncProcessingConfig)
    truth_features: TruthFeaturesConfig = Field(default_factory=TruthFeaturesConfig)
    dataset_builder_v2: DatasetBuilderV2Config = Field(default_factory=DatasetBuilderV2Config)
    training_scheduler: TrainingSchedulerConfig = Field(default_factory=TrainingSchedulerConfig)
    
    # Storage paths
    data_root_dir: Path = Field(
        default_factory=lambda: Path("./data"),
        description="Root directory for data storage"
    )
    models_root_dir: Path = Field(
        default_factory=lambda: Path("./models"),
        description="Root directory for model storage"
    )
    logs_dir: Path = Field(
        default_factory=lambda: Path("./logs"),
        description="Directory for log files"
    )
    
    # Development settings
    dev_mode: bool = Field(default=False, description="Development mode")
    dev_auto_reload: bool = Field(default=False, description="Auto-reload in development")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is one of the accepted values."""
        valid_levels = {'debug', 'info', 'warning', 'error', 'critical'}
        if v.lower() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.lower()
    
    @field_validator('data_root_dir', 'models_root_dir', 'logs_dir')
    @classmethod
    def validate_directories(cls, v: Union[str, Path]) -> Path:
        """Ensure directories exist and are writable."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        if not os.access(path, os.W_OK):
            raise ValueError(f"Directory {path} is not writable")
        return path
    
    @computed_field
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and not self.dev_mode
    
    @computed_field
    @property
    def full_api_url(self) -> str:
        """Get full API URL including host and port."""
        return f"http://{self.api.host}:{self.api.port}{self.api.prefix}"
    
    @computed_field
    @property
    def full_mep_url(self) -> str:
        """Get full MEP API URL."""
        return f"http://{self.api.host}:{self.api.port}{self.api.mep_prefix}"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    This function provides dependency injection for FastAPI endpoints
    and can be used throughout the application for configuration access.
    
    Returns:
        Settings: The global settings instance
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment variables and .env file.
    
    Useful for testing or when configuration changes need to be applied
    without restarting the application.
    
    Returns:
        Settings: The reloaded settings instance
    """
    global settings
    settings = Settings()
    return settings


# Export commonly used configurations for convenience
__all__ = [
    "Settings",
    "ModelConfig", 
    "EmbeddingsConfig",
    "VectorStoreConfig",
    "APIConfig",
    "PerformanceConfig",
    "TruthModelConfig",
    "ConversationProcessingConfig",
    "LoRATrainingConfig",
    "AsyncProcessingConfig",
    "TruthFeaturesConfig", 
    "DatasetBuilderV2Config",
    "TrainingSchedulerConfig",
    "settings",
    "get_settings",
    "reload_settings",
]