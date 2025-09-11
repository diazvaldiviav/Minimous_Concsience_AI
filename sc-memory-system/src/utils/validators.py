"""
Input validation utilities for SC Memory System.

This module provides common validation functions for API inputs,
embeddings, and other data validation needs.
"""

import re
from typing import Any, Dict, List, Optional, Union
import numpy as np

from ..core.exceptions import ValidationError


def validate_text_input(
    text: Any,
    min_length: int = 1,
    max_length: int = 10000,
    allow_empty: bool = False
) -> str:
    """
    Validate text input with length constraints.
    
    Args:
        text: Input to validate
        min_length: Minimum text length
        max_length: Maximum text length
        allow_empty: Whether to allow empty strings
        
    Returns:
        Validated text string
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(text, str):
        raise ValidationError(
            message=f"Text must be a string, got {type(text).__name__}",
            field_errors={"text": "must be string"}
        )
    
    text = text.strip()
    
    if not allow_empty and len(text) == 0:
        raise ValidationError(
            message="Text cannot be empty",
            field_errors={"text": "cannot be empty"}
        )
    
    if len(text) < min_length:
        raise ValidationError(
            message=f"Text too short: {len(text)} < {min_length}",
            field_errors={"text": f"minimum length {min_length}"}
        )
    
    if len(text) > max_length:
        raise ValidationError(
            message=f"Text too long: {len(text)} > {max_length}",
            field_errors={"text": f"maximum length {max_length}"}
        )
    
    return text


def validate_embedding(
    embedding: Any,
    expected_dim: Optional[int] = None,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Validate embedding vector format and dimensions.
    
    Args:
        embedding: Embedding to validate
        expected_dim: Expected dimension
        dtype: Expected numpy dtype
        
    Returns:
        Validated embedding as numpy array
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(embedding, (np.ndarray, list)):
        raise ValidationError(
            message=f"Embedding must be numpy array or list, got {type(embedding).__name__}",
            field_errors={"embedding": "must be numpy array or list"}
        )
    
    # Convert to numpy array
    if isinstance(embedding, list):
        try:
            embedding = np.array(embedding, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                message=f"Failed to convert embedding to numpy array: {e}",
                field_errors={"embedding": "invalid format"}
            )
    
    # Check dimensions
    if len(embedding.shape) != 1:
        raise ValidationError(
            message=f"Embedding must be 1D, got shape {embedding.shape}",
            field_errors={"embedding": "must be 1D array"}
        )
    
    if expected_dim is not None and embedding.shape[0] != expected_dim:
        raise ValidationError(
            message=f"Embedding dimension {embedding.shape[0]} != expected {expected_dim}",
            field_errors={"embedding": f"dimension must be {expected_dim}"}
        )
    
    # Check for invalid values
    if np.any(np.isnan(embedding)):
        raise ValidationError(
            message="Embedding contains NaN values",
            field_errors={"embedding": "contains NaN"}
        )
    
    if np.any(np.isinf(embedding)):
        raise ValidationError(
            message="Embedding contains infinite values",
            field_errors={"embedding": "contains infinite values"}
        )
    
    # Convert dtype if needed
    if embedding.dtype != dtype:
        embedding = embedding.astype(dtype)
    
    return embedding


def validate_embeddings_batch(
    embeddings: Any,
    expected_dim: Optional[int] = None,
    max_batch_size: int = 1000,
    dtype: type = np.float32
) -> np.ndarray:
    """
    Validate batch of embeddings.
    
    Args:
        embeddings: Batch of embeddings to validate
        expected_dim: Expected dimension for each embedding
        max_batch_size: Maximum batch size
        dtype: Expected numpy dtype
        
    Returns:
        Validated embeddings as 2D numpy array
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(embeddings, (np.ndarray, list)):
        raise ValidationError(
            message=f"Embeddings must be numpy array or list, got {type(embeddings).__name__}",
            field_errors={"embeddings": "must be numpy array or list"}
        )
    
    # Convert to numpy array
    if isinstance(embeddings, list):
        try:
            embeddings = np.array(embeddings, dtype=dtype)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                message=f"Failed to convert embeddings to numpy array: {e}",
                field_errors={"embeddings": "invalid format"}
            )
    
    # Check dimensions
    if len(embeddings.shape) != 2:
        raise ValidationError(
            message=f"Embeddings must be 2D, got shape {embeddings.shape}",
            field_errors={"embeddings": "must be 2D array"}
        )
    
    batch_size, dim = embeddings.shape
    
    if batch_size == 0:
        raise ValidationError(
            message="Embeddings batch cannot be empty",
            field_errors={"embeddings": "cannot be empty"}
        )
    
    if batch_size > max_batch_size:
        raise ValidationError(
            message=f"Batch size {batch_size} > maximum {max_batch_size}",
            field_errors={"embeddings": f"batch size must be <= {max_batch_size}"}
        )
    
    if expected_dim is not None and dim != expected_dim:
        raise ValidationError(
            message=f"Embedding dimension {dim} != expected {expected_dim}",
            field_errors={"embeddings": f"dimension must be {expected_dim}"}
        )
    
    # Check for invalid values
    if np.any(np.isnan(embeddings)):
        raise ValidationError(
            message="Embeddings contain NaN values",
            field_errors={"embeddings": "contains NaN"}
        )
    
    if np.any(np.isinf(embeddings)):
        raise ValidationError(
            message="Embeddings contain infinite values",
            field_errors={"embeddings": "contains infinite values"}
        )
    
    # Convert dtype if needed
    if embeddings.dtype != dtype:
        embeddings = embeddings.astype(dtype)
    
    return embeddings


def validate_metadata(
    metadata: Any,
    required_fields: Optional[List[str]] = None,
    max_fields: int = 50
) -> Dict[str, Any]:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata to validate
        required_fields: List of required field names
        max_fields: Maximum number of fields
        
    Returns:
        Validated metadata dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(metadata, dict):
        raise ValidationError(
            message=f"Metadata must be a dictionary, got {type(metadata).__name__}",
            field_errors={"metadata": "must be dictionary"}
        )
    
    if len(metadata) > max_fields:
        raise ValidationError(
            message=f"Too many metadata fields: {len(metadata)} > {max_fields}",
            field_errors={"metadata": f"maximum {max_fields} fields"}
        )
    
    # Check required fields
    if required_fields:
        missing_fields = []
        for field in required_fields:
            if field not in metadata:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValidationError(
                message=f"Missing required fields: {missing_fields}",
                field_errors={field: "required" for field in missing_fields}
            )
    
    # Validate field names and values
    validated_metadata = {}
    for key, value in metadata.items():
        # Validate key
        if not isinstance(key, str):
            raise ValidationError(
                message=f"Metadata key must be string, got {type(key).__name__}",
                field_errors={str(key): "key must be string"}
            )
        
        if len(key) == 0:
            raise ValidationError(
                message="Metadata key cannot be empty",
                field_errors={"": "key cannot be empty"}
            )
        
        if len(key) > 100:
            raise ValidationError(
                message=f"Metadata key too long: {len(key)} > 100",
                field_errors={key: "key maximum length 100"}
            )
        
        # Validate key format (alphanumeric and underscores)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', key):
            raise ValidationError(
                message=f"Invalid metadata key format: {key}",
                field_errors={key: "must be alphanumeric with underscores"}
            )
        
        # Validate value (must be JSON serializable)
        try:
            import json
            json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                message=f"Metadata value not JSON serializable: {e}",
                field_errors={key: "value must be JSON serializable"}
            )
        
        validated_metadata[key] = value
    
    return validated_metadata


def validate_search_filters(
    filters: Any,
    max_filters: int = 20
) -> Dict[str, Any]:
    """
    Validate search filter parameters.
    
    Args:
        filters: Filters to validate
        max_filters: Maximum number of filters
        
    Returns:
        Validated filters dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    if filters is None:
        return {}
    
    if not isinstance(filters, dict):
        raise ValidationError(
            message=f"Filters must be a dictionary, got {type(filters).__name__}",
            field_errors={"filters": "must be dictionary"}
        )
    
    if len(filters) > max_filters:
        raise ValidationError(
            message=f"Too many filters: {len(filters)} > {max_filters}",
            field_errors={"filters": f"maximum {max_filters} filters"}
        )
    
    validated_filters = {}
    
    for key, value in filters.items():
        # Validate key
        if not isinstance(key, str):
            raise ValidationError(
                message=f"Filter key must be string, got {type(key).__name__}",
                field_errors={str(key): "key must be string"}
            )
        
        if len(key) == 0:
            raise ValidationError(
                message="Filter key cannot be empty",
                field_errors={"": "key cannot be empty"}
            )
        
        # Validate value types
        if isinstance(value, dict):
            # Range query validation
            valid_operators = {'gte', 'gt', 'lte', 'lt', 'eq', 'ne'}
            for op, op_value in value.items():
                if op not in valid_operators:
                    raise ValidationError(
                        message=f"Invalid filter operator: {op}",
                        field_errors={key: f"operator must be one of {valid_operators}"}
                    )
                
                if not isinstance(op_value, (int, float, str)):
                    raise ValidationError(
                        message=f"Filter operator value must be scalar, got {type(op_value).__name__}",
                        field_errors={key: f"operator {op} value must be scalar"}
                    )
        
        elif isinstance(value, list):
            # List membership validation
            if len(value) == 0:
                raise ValidationError(
                    message=f"Filter list cannot be empty for key {key}",
                    field_errors={key: "list cannot be empty"}
                )
            
            if len(value) > 100:
                raise ValidationError(
                    message=f"Filter list too long: {len(value)} > 100",
                    field_errors={key: "list maximum length 100"}
                )
        
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise ValidationError(
                message=f"Invalid filter value type: {type(value).__name__}",
                field_errors={key: "value must be scalar, dict, or list"}
            )
        
        validated_filters[key] = value
    
    return validated_filters


def validate_uuid(
    value: Any,
    field_name: str = "id"
) -> str:
    """
    Validate UUID string format.
    
    Args:
        value: Value to validate as UUID
        field_name: Name of the field for error messages
        
    Returns:
        Validated UUID string
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, str):
        raise ValidationError(
            message=f"{field_name} must be a string, got {type(value).__name__}",
            field_errors={field_name: "must be string"}
        )
    
    # UUID format validation
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    
    if not uuid_pattern.match(value):
        raise ValidationError(
            message=f"Invalid UUID format: {value}",
            field_errors={field_name: "must be valid UUID"}
        )
    
    return value.lower()


def validate_positive_integer(
    value: Any,
    field_name: str = "value",
    min_value: int = 1,
    max_value: Optional[int] = None
) -> int:
    """
    Validate positive integer with bounds.
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated integer
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, int):
        # Try conversion
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ValidationError(
                message=f"{field_name} must be an integer, got {type(value).__name__}",
                field_errors={field_name: "must be integer"}
            )
    
    if value < min_value:
        raise ValidationError(
            message=f"{field_name} must be >= {min_value}, got {value}",
            field_errors={field_name: f"minimum value {min_value}"}
        )
    
    if max_value is not None and value > max_value:
        raise ValidationError(
            message=f"{field_name} must be <= {max_value}, got {value}",
            field_errors={field_name: f"maximum value {max_value}"}
        )
    
    return value


def validate_probability(
    value: Any,
    field_name: str = "probability"
) -> float:
    """
    Validate probability value (0.0 to 1.0).
    
    Args:
        value: Value to validate
        field_name: Name of the field for error messages
        
    Returns:
        Validated probability
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValidationError(
                message=f"{field_name} must be a number, got {type(value).__name__}",
                field_errors={field_name: "must be number"}
            )
    
    if not (0.0 <= value <= 1.0):
        raise ValidationError(
            message=f"{field_name} must be between 0.0 and 1.0, got {value}",
            field_errors={field_name: "must be between 0.0 and 1.0"}
        )
    
    return float(value)


__all__ = [
    "validate_text_input",
    "validate_embedding",
    "validate_embeddings_batch", 
    "validate_metadata",
    "validate_search_filters",
    "validate_uuid",
    "validate_positive_integer",
    "validate_probability",
]