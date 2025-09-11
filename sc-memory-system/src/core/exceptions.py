"""
Custom exceptions for SC Memory System.

This module defines application-specific exceptions with proper error codes,
messages, and context information for debugging and error handling.
All exceptions inherit from appropriate base classes and provide
structured error information for API responses.
"""

from typing import Any, Dict, Optional


class SCMemoryException(Exception):
    """Base exception class for all SC Memory System errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SC_GENERAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize base exception.
        
        Args:
            message: Human-readable error message
            error_code: Specific error code for programmatic handling
            details: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error_code": self.error_code,
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


class ConfigurationError(SCMemoryException):
    """Raised when there are configuration or settings errors."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="SC_CONFIG_ERROR",
            details=details
        )
        self.config_key = config_key


class ModelLoadError(SCMemoryException):
    """Raised when model loading fails."""
    
    def __init__(
        self,
        message: str,
        model_name: str,
        model_type: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        details.update({
            "model_name": model_name,
            "model_type": model_type
        })
        
        super().__init__(
            message=message,
            error_code="SC_MODEL_LOAD_ERROR",
            details=details
        )
        self.model_name = model_name
        self.model_type = model_type


class ModelNotLoadedError(SCMemoryException):
    """Raised when trying to use a model that hasn't been loaded."""
    
    def __init__(
        self,
        message: str = "Model has not been loaded",
        model_name: str = "unknown",
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        details["model_name"] = model_name
        
        super().__init__(
            message=message,
            error_code="SC_MODEL_NOT_LOADED",
            details=details
        )
        self.model_name = model_name


class TokenizationError(SCMemoryException):
    """Raised when tokenization fails."""
    
    def __init__(
        self,
        message: str,
        text_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if text_length is not None:
            details["text_length"] = text_length
            
        super().__init__(
            message=message,
            error_code="SC_TOKENIZATION_ERROR",
            details=details
        )
        self.text_length = text_length


class EmbeddingError(SCMemoryException):
    """Raised when embedding generation fails."""
    
    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        text_length: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if model_name:
            details["model_name"] = model_name
        if text_length is not None:
            details["text_length"] = text_length
            
        super().__init__(
            message=message,
            error_code="SC_EMBEDDING_ERROR",
            details=details
        )
        self.model_name = model_name
        self.text_length = text_length


class VectorStoreError(SCMemoryException):
    """Raised when vector store operations fail."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        index_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if operation:
            details["operation"] = operation
        if index_type:
            details["index_type"] = index_type
            
        super().__init__(
            message=message,
            error_code="SC_VECTOR_STORE_ERROR",
            details=details
        )
        self.operation = operation
        self.index_type = index_type


class VectorIndexError(VectorStoreError):
    """Raised when FAISS index operations fail."""
    
    def __init__(
        self,
        message: str,
        index_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if index_path:
            details["index_path"] = index_path
            
        super().__init__(
            message=message,
            operation="index_operation",
            details=details
        )
        self.error_code = "SC_VECTOR_INDEX_ERROR"
        self.index_path = index_path


class SearchError(SCMemoryException):
    """Raised when vector search operations fail."""
    
    def __init__(
        self,
        message: str,
        query_text: Optional[str] = None,
        k: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if query_text:
            details["query_text"] = query_text[:100] + "..." if len(query_text) > 100 else query_text
        if k is not None:
            details["k"] = k
            
        super().__init__(
            message=message,
            error_code="SC_SEARCH_ERROR",
            details=details
        )
        self.query_text = query_text
        self.k = k


class MEPError(SCMemoryException):
    """Raised when Memory Exchange Protocol operations fail."""
    
    def __init__(
        self,
        message: str,
        proposal_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if proposal_id:
            details["proposal_id"] = proposal_id
            
        super().__init__(
            message=message,
            error_code="SC_MEP_ERROR",
            details=details
        )
        self.proposal_id = proposal_id


class MEPValidationError(MEPError):
    """Raised when MEP proposal validation fails."""
    
    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if field_name:
            details["field_name"] = field_name
        if field_value is not None:
            details["field_value"] = str(field_value)
            
        super().__init__(
            message=message,
            details=details
        )
        self.error_code = "SC_MEP_VALIDATION_ERROR"
        self.field_name = field_name
        self.field_value = field_value


class MEPQueueError(MEPError):
    """Raised when MEP proposal queue operations fail."""
    
    def __init__(
        self,
        message: str,
        queue_size: Optional[int] = None,
        max_size: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if queue_size is not None:
            details["queue_size"] = queue_size
        if max_size is not None:
            details["max_size"] = max_size
            
        super().__init__(
            message=message,
            details=details
        )
        self.error_code = "SC_MEP_QUEUE_ERROR"
        self.queue_size = queue_size
        self.max_size = max_size


class AuthenticationError(SCMemoryException):
    """Raised when authentication fails."""
    
    def __init__(
        self,
        message: str = "Authentication failed",
        auth_method: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if auth_method:
            details["auth_method"] = auth_method
            
        super().__init__(
            message=message,
            error_code="SC_AUTH_ERROR",
            details=details
        )
        self.auth_method = auth_method


class AuthorizationError(SCMemoryException):
    """Raised when authorization fails."""
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permission: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if required_permission:
            details["required_permission"] = required_permission
            
        super().__init__(
            message=message,
            error_code="SC_AUTHZ_ERROR",
            details=details
        )
        self.required_permission = required_permission


class ResourceError(SCMemoryException):
    """Raised when resource allocation or management fails."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_limit: Optional[Any] = None,
        current_usage: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_limit is not None:
            details["resource_limit"] = resource_limit
        if current_usage is not None:
            details["current_usage"] = current_usage
            
        super().__init__(
            message=message,
            error_code="SC_RESOURCE_ERROR",
            details=details
        )
        self.resource_type = resource_type
        self.resource_limit = resource_limit
        self.current_usage = current_usage


class MemoryError(ResourceError):
    """Raised when memory allocation or limits are exceeded."""
    
    def __init__(
        self,
        message: str,
        memory_limit_mb: Optional[float] = None,
        current_usage_mb: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        super().__init__(
            message=message,
            resource_type="memory",
            resource_limit=memory_limit_mb,
            current_usage=current_usage_mb,
            details=details
        )
        self.error_code = "SC_MEMORY_ERROR"


class TimeoutError(SCMemoryException):
    """Raised when operations timeout."""
    
    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if operation:
            details["operation"] = operation
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
            
        super().__init__(
            message=message,
            error_code="SC_TIMEOUT_ERROR",
            details=details
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


class ValidationError(SCMemoryException):
    """Raised when input validation fails."""
    
    def __init__(
        self,
        message: str,
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if field_errors:
            details["field_errors"] = field_errors
            
        super().__init__(
            message=message,
            error_code="SC_VALIDATION_ERROR",
            details=details
        )
        self.field_errors = field_errors or {}


class ServiceUnavailableError(SCMemoryException):
    """Raised when a required service is unavailable."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if service_name:
            details["service_name"] = service_name
        if retry_after_seconds is not None:
            details["retry_after_seconds"] = retry_after_seconds
            
        super().__init__(
            message=message,
            error_code="SC_SERVICE_UNAVAILABLE",
            details=details
        )
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds


# Compatibility mapping for common errors
class CompatibilityError(SCMemoryException):
    """Raised when there are compatibility issues between components."""
    
    def __init__(
        self,
        message: str,
        component_a: Optional[str] = None,
        component_b: Optional[str] = None,
        version_info: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        details = details or {}
        if component_a:
            details["component_a"] = component_a
        if component_b:
            details["component_b"] = component_b
        if version_info:
            details["version_info"] = version_info
            
        super().__init__(
            message=message,
            error_code="SC_COMPATIBILITY_ERROR",
            details=details
        )
        self.component_a = component_a
        self.component_b = component_b
        self.version_info = version_info


# Export all exceptions for easy import
__all__ = [
    "SCMemoryException",
    "ConfigurationError",
    "ModelLoadError",
    "ModelNotLoadedError",
    "TokenizationError",
    "EmbeddingError",
    "VectorStoreError",
    "VectorIndexError",
    "SearchError",
    "MEPError",
    "MEPValidationError",
    "MEPQueueError",
    "AuthenticationError",
    "AuthorizationError",
    "ResourceError",
    "MemoryError",
    "TimeoutError",
    "ValidationError",
    "ServiceUnavailableError",
    "CompatibilityError",
]