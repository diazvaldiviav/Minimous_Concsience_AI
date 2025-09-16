"""
MEP (Memory Exchange Protocol) API routes for SC Memory System.

This module implements the MEP API endpoints with comprehensive validation,
authentication, error handling, and request processing. Follows the MEP
specification exactly with proper HTTP status codes and response formats.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
    BackgroundTasks
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

from ...core.config import Settings, get_settings
from ...core.exceptions import (
    MEPError,
    MEPValidationError,
    MEPQueueError,
    AuthenticationError,
    ValidationError as SCValidationError,
    ServiceUnavailableError,
)
from ...core.models import ErrorDetail, AsyncProposalStatus
from .schemas import (
    MEPProposalRequest,
    MEPProposalSuccessResponse,
    MEPProposalErrorResponse,
    MEPHealthResponse,
    MEPValidationErrorResponse,
    MEPBatchProposalRequest,
    MEPBatchProposalResponse,
)

logger = logging.getLogger(__name__)

# Initialize FastAPI router
router = APIRouter()

# Security
security = HTTPBearer()

# Week 3: Replace in-memory queue with advanced async processor
from .async_processor import get_async_processor

# Application state tracking
app_start_time = time.time()
request_counter = 0
active_requests = 0


async def authenticate_request(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    settings: Settings = Depends(get_settings)
) -> bool:
    """
    Authenticate request using bearer token.
    
    Args:
        credentials: HTTP bearer token credentials
        settings: Application settings
        
    Returns:
        True if authentication successful
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials or not credentials.credentials:
        logger.warning("Authentication failed: No token provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Validate bearer token
    if credentials.credentials != settings.api.bearer_token:
        logger.warning(f"Authentication failed: Invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return True


async def validate_request_size(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> None:
    """
    Validate request content length.
    
    Args:
        request: FastAPI request object
        settings: Application settings
        
    Raises:
        HTTPException: If request too large
    """
    content_length = request.headers.get('content-length')
    if content_length:
        if int(content_length) > settings.api.mep_max_proposal_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large. Maximum size: {settings.api.mep_max_proposal_size} bytes"
            )


async def track_request_metrics(request: Request) -> None:
    """Track request metrics and active request count."""
    global request_counter, active_requests
    
    request_counter += 1
    active_requests += 1
    
    # Store metrics in request state for cleanup
    request.state.start_time = time.time()
    request.state.request_id = str(uuid4())[:8]
    
    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"(ID: {request.state.request_id}, Active: {active_requests})"
    )


async def cleanup_request_metrics(request: Request) -> None:
    """Cleanup request metrics after processing."""
    global active_requests
    
    active_requests = max(0, active_requests - 1)
    
    if hasattr(request.state, 'start_time'):
        duration = time.time() - request.state.start_time
        logger.info(
            f"Request completed: {request.method} {request.url.path} "
            f"(ID: {getattr(request.state, 'request_id', 'unknown')}, "
            f"Duration: {duration:.3f}s, Active: {active_requests})"
        )


async def submit_to_async_processor(proposal_request: MEPProposalRequest) -> str:
    """
    Submit proposal to Week 3 async processor.
    
    Args:
        proposal_request: MEP proposal request
        
    Returns:
        Proposal ID for tracking
        
    Raises:
        MEPQueueError: If async processor queue is full
        ServiceUnavailableError: If async processor is not available
    """
    try:
        # Get the global async processor
        async_processor = await get_async_processor()
        
        # Submit proposal to async processor
        proposal_id = await async_processor.submit_proposal(proposal_request)
        
        logger.info(f"Submitted proposal {proposal_id} to async processor")
        return proposal_id
        
    except MEPQueueError:
        # Re-raise queue errors
        raise
    except Exception as e:
        logger.error(f"Failed to submit to async processor: {e}", exc_info=True)
        raise ServiceUnavailableError(
            "Async processing service unavailable",
            service_name="async_processor"
        )


def create_error_response(
    error: Exception,
    request_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create standardized error response.
    
    Args:
        error: Exception to convert to response
        request_id: Optional request ID for tracking
        
    Returns:
        Error response dictionary
    """
    if isinstance(error, MEPError):
        error_detail = ErrorDetail(
            error_code=error.error_code,
            error_type=error.__class__.__name__,
            message=error.message,
            details=error.details,
            request_id=request_id
        )
    else:
        error_detail = ErrorDetail(
            error_code="SC_GENERAL_ERROR",
            error_type=error.__class__.__name__,
            message=str(error),
            request_id=request_id
        )
    
    return MEPProposalErrorResponse(
        error=error_detail
    ).model_dump()


@router.post(
    "/proposals",
    response_model=MEPProposalSuccessResponse,
    responses={
        202: {"description": "Proposal accepted for processing"},
        400: {"model": MEPValidationErrorResponse, "description": "Validation error"},
        401: {"model": MEPProposalErrorResponse, "description": "Authentication failed"},
        413: {"model": MEPProposalErrorResponse, "description": "Request too large"},
        422: {"model": MEPValidationErrorResponse, "description": "Request validation failed"},
        429: {"model": MEPProposalErrorResponse, "description": "Queue full"},
        500: {"model": MEPProposalErrorResponse, "description": "Internal server error"},
    },
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit MEP Proposal",
    description="Submit a Memory Exchange Protocol proposal for processing. "
                "Proposals are validated and queued for background processing.",
    tags=["MEP"]
)
async def submit_proposal(
    proposal_request: MEPProposalRequest,
    background_tasks: BackgroundTasks,
    request: Request,
    authenticated: bool = Depends(authenticate_request),
    _: None = Depends(validate_request_size),
    settings: Settings = Depends(get_settings)
) -> MEPProposalSuccessResponse:
    """
    Submit a Memory Exchange Protocol proposal.
    
    This endpoint accepts MEP proposals for memory consolidation processing.
    Proposals are validated against the MEP schema, authenticated, and queued
    for background processing.
    
    The endpoint returns immediately with a 202 Accepted status and a unique
    proposal ID for tracking. Actual processing happens asynchronously.
    """
    # Track request metrics
    await track_request_metrics(request)
    
    try:
        request_id = getattr(request.state, 'request_id', None)
        
        logger.info(
            f"Received MEP proposal: provider={proposal_request.provider}, "
            f"model={proposal_request.model}, trigger={proposal_request.trigger} "
            f"(Request ID: {request_id})"
        )
        
        # Additional business logic validation
        if proposal_request.token_usage.used_tokens / proposal_request.token_usage.window_tokens > 0.95:
            logger.warning(f"High context fill ratio: {proposal_request.token_usage.used_tokens / proposal_request.token_usage.window_tokens}")
        
        if len(proposal_request.key_facts) > 20:
            logger.warning(f"Large number of key facts: {len(proposal_request.key_facts)}")
        
        # Submit to Week 3 async processor
        try:
            proposal_id = await submit_to_async_processor(proposal_request)
        except MEPQueueError as e:
            logger.error(f"Async processor queue full: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=create_error_response(e, request_id)
            )
        except ServiceUnavailableError as e:
            logger.error(f"Async processor unavailable: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(e, request_id)
            )
        
        # Get queue status for response
        async_processor = await get_async_processor()
        queue_status = await async_processor.get_queue_status()
        
        # Estimate processing time based on queue size and current stage
        estimated_time = queue_status.get("queue_sizes", {}).get("staging", 0) * 45  # 45 seconds per proposal estimate
        
        # Create success response
        response = MEPProposalSuccessResponse(
            proposal_id=proposal_id,
            status="accepted",
            message="Proposal accepted for async processing",
            queue_position=queue_status.get("total_proposals", 0) + 1,
            estimated_processing_time=estimated_time
        )
        
        logger.info(
            f"MEP proposal accepted: {proposal_id} "
            f"(Queue size: {queue_status.get('total_proposals', 0)}, ETA: {estimated_time}s)"
        )
        
        return response
        
    except ValidationError as e:
        logger.error(f"Pydantic validation error: {e}")
        
        # Format validation errors
        error_details = []
        for error in e.errors():
            error_details.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=MEPValidationErrorResponse(
                message="Request validation failed",
                details=error_details
            ).model_dump()
        )
        
    except MEPError as e:
        logger.error(f"MEP error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=create_error_response(e, getattr(request.state, 'request_id', None))
        )
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(e, getattr(request.state, 'request_id', None))
        )
        
    finally:
        await cleanup_request_metrics(request)


@router.get(
    "/proposals/{proposal_id}/status",
    responses={
        200: {"description": "Proposal status retrieved"},
        401: {"model": MEPProposalErrorResponse, "description": "Authentication failed"},
        404: {"model": MEPProposalErrorResponse, "description": "Proposal not found"},
        500: {"model": MEPProposalErrorResponse, "description": "Internal server error"},
    },
    summary="Get Proposal Status",
    description="Retrieve the status of a submitted MEP proposal.",
    tags=["MEP"]
)
async def get_proposal_status(
    proposal_id: str,
    request: Request,
    authenticated: bool = Depends(authenticate_request)
) -> Dict[str, Any]:
    """
    Get the status of a MEP proposal.
    
    Returns the current processing status of a proposal that was previously
    submitted through the /proposals endpoint.
    """
    await track_request_metrics(request)
    
    try:
        # Get proposal status from async processor
        async_processor = await get_async_processor()
        proposal_status = await async_processor.get_proposal_status(proposal_id)
        
        if proposal_status is None:
            # Proposal not found
            logger.warning(f"Proposal not found: {proposal_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    MEPError(f"Proposal {proposal_id} not found", proposal_id=proposal_id),
                    getattr(request.state, 'request_id', None)
                )
            )
        
        # Convert AsyncProposalStatus to response format
        response_data = {
            "proposal_id": proposal_status.proposal_id,
            "overall_status": proposal_status.overall_status,
            "current_stage": proposal_status.current_stage,
            "submitted_at": proposal_status.submitted_at.isoformat(),
            "started_processing_at": proposal_status.started_processing_at.isoformat() if proposal_status.started_processing_at else None,
            "estimated_completion_at": proposal_status.estimated_completion_at.isoformat() if proposal_status.estimated_completion_at else None,
            "retry_count": proposal_status.retry_count,
            "worker_id": proposal_status.worker_id,
            "stages": [
                {
                    "stage_name": stage.stage_name,
                    "status": stage.status,
                    "progress_percent": stage.progress_percent,
                    "started_at": stage.started_at.isoformat() if stage.started_at else None,
                    "completed_at": stage.completed_at.isoformat() if stage.completed_at else None,
                    "error_message": stage.error_message
                }
                for stage in proposal_status.stages
            ],
            "processing_metadata": proposal_status.processing_metadata
        }
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting proposal status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(e, getattr(request.state, 'request_id', None))
        )
    finally:
        await cleanup_request_metrics(request)


@router.get(
    "/health",
    response_model=MEPHealthResponse,
    summary="Health Check",
    description="Check the health status of the MEP API and its components.",
    tags=["Health"]
)
async def health_check(
    request: Request,
    settings: Settings = Depends(get_settings)
) -> MEPHealthResponse:
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns the health status of the MEP API and its components,
    along with performance metrics and uptime information.
    """
    await track_request_metrics(request)
    
    try:
        uptime = time.time() - app_start_time
        
        # Check component health
        components = {
            "base_model": "healthy",  # Would check actual model status
            "embeddings": "healthy",  # Would check embeddings service
            "vector_store": "healthy",  # Would check FAISS vector store
            "async_processor": "healthy"  # Would check async processor status
        }
        
        # Check async processor health
        try:
            async_processor = await get_async_processor()
            queue_status = await async_processor.get_queue_status()
            if not queue_status.get("processing_enabled", False):
                components["async_processor"] = "degraded"
        except Exception as e:
            logger.error(f"Async processor health check failed: {e}")
            components["async_processor"] = "unhealthy"
        
        # Determine overall status
        overall_status = "healthy"
        if any(status != "healthy" for status in components.values()):
            overall_status = "degraded" if all(status in ["healthy", "degraded"] for status in components.values()) else "unhealthy"
        
        # Performance metrics
        try:
            async_processor = await get_async_processor()
            queue_status = await async_processor.get_queue_status()
            
            metrics = {
                "async_queue_total": queue_status.get("total_proposals", 0),
                "async_queue_by_stage": queue_status.get("stage_counts", {}),
                "active_workers": queue_status.get("active_workers", 0),
                "active_requests": active_requests,
                "total_requests": request_counter,
                "uptime_seconds": uptime
            }
        except Exception as e:
            logger.error(f"Failed to get async processor metrics: {e}")
            metrics = {
                "active_requests": active_requests,
                "total_requests": request_counter,
                "uptime_seconds": uptime,
                "error": "Failed to get async processor metrics"
            }
        
        response = MEPHealthResponse(
            status=overall_status,
            uptime_seconds=uptime,
            components=components,
            metrics=metrics
        )
        
        logger.debug(f"Health check: {overall_status}")
        return response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        
        # Return unhealthy status
        return MEPHealthResponse(
            status="unhealthy",
            uptime_seconds=time.time() - app_start_time,
            components={"error": str(e)},
            metrics={"error": True}
        )
    finally:
        await cleanup_request_metrics(request)


@router.get(
    "/queue/status",
    responses={
        200: {"description": "Queue status retrieved"},
        401: {"model": MEPProposalErrorResponse, "description": "Authentication failed"},
    },
    summary="Get Queue Status",
    description="Get current status of the MEP processing queue.",
    tags=["MEP"]
)
async def get_queue_status(
    request: Request,
    authenticated: bool = Depends(authenticate_request),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Get the current status of the MEP processing queue.
    
    Returns information about queue size, processing rates, and
    recent proposal statistics.
    """
    await track_request_metrics(request)
    
    try:
        # Get comprehensive queue status from async processor
        async_processor = await get_async_processor()
        queue_status = await async_processor.get_queue_status()
        
        # Calculate utilization
        max_queue_size = settings.async_processing.max_queue_size
        total_proposals = queue_status.get("total_proposals", 0)
        utilization = total_proposals / max_queue_size if max_queue_size > 0 else 0
        
        # Enhanced response with Week 3 async processing details
        return {
            "queue_size": total_proposals,
            "max_queue_size": max_queue_size,
            "utilization": utilization,
            "stage_breakdown": queue_status.get("stage_counts", {}),
            "queue_sizes_by_stage": queue_status.get("queue_sizes", {}),
            "active_workers": queue_status.get("active_workers", 0),
            "total_workers": queue_status.get("total_workers", 0),
            "processing_enabled": queue_status.get("processing_enabled", False),
            "recent_resource_usage": queue_status.get("recent_resource_usage"),
            "uptime_seconds": queue_status.get("uptime_seconds", 0),
            "average_processing_time_estimate": 45,  # seconds per proposal
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(e, getattr(request.state, 'request_id', None))
        )
    finally:
        await cleanup_request_metrics(request)


@router.get(
    "/proposals/{proposal_id}/status",
    responses={
        200: {"description": "Proposal status retrieved"},
        404: {"model": MEPProposalErrorResponse, "description": "Proposal not found"},
        401: {"model": MEPProposalErrorResponse, "description": "Authentication failed"},
    },
    summary="Get Proposal Status",
    description="Get the current status and logs of a specific proposal.",
    tags=["MEP"]
)
async def get_proposal_status(
    proposal_id: str,
    request: Request,
    authenticated: bool = Depends(authenticate_request),
    settings: Settings = Depends(get_settings)
) -> Dict[str, Any]:
    """
    Get the current status of a specific MEP proposal.

    Returns detailed information about the proposal's training progress,
    current status, and any available logs.
    """
    await track_request_metrics(request)

    try:
        # Get proposal status from async processor
        async_processor = await get_async_processor()

        # Check if proposal exists and get its status
        proposal_status = await async_processor.get_proposal_status(proposal_id)

        if not proposal_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    f"Proposal {proposal_id} not found",
                    getattr(request.state, 'request_id', None)
                )
            )

        # Enhanced status with training logs
        response = {
            "proposal_id": proposal_id,
            "status": proposal_status.get("status", "unknown"),
            "stage": proposal_status.get("stage", "unknown"),
            "progress": proposal_status.get("progress", 0.0),
            "created_at": proposal_status.get("created_at"),
            "updated_at": proposal_status.get("updated_at"),
            "error": proposal_status.get("error"),
            "logs": proposal_status.get("logs", "")
        }

        # Add training metrics if available
        if "training_metrics" in proposal_status:
            response["training_metrics"] = proposal_status["training_metrics"]

        # Add adapter info if training completed
        if proposal_status.get("status") == "completed" and "adapter_path" in proposal_status:
            response["adapter_path"] = proposal_status["adapter_path"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting proposal status for {proposal_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(e, getattr(request.state, 'request_id', None))
        )
    finally:
        await cleanup_request_metrics(request)


# Export router for main application
__all__ = ["router"]