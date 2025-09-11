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
)
from ...core.models import ErrorDetail
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

# In-memory queue for MVP (replace with Redis/database in production)
proposal_queue: List[Dict[str, Any]] = []
queue_lock = asyncio.Lock()

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


async def add_to_queue(proposal: Dict[str, Any]) -> int:
    """
    Add proposal to processing queue.
    
    Args:
        proposal: Proposal data to queue
        
    Returns:
        Queue position
        
    Raises:
        MEPQueueError: If queue is full
    """
    async with queue_lock:
        settings = get_settings()
        
        if len(proposal_queue) >= settings.api.mep_queue_max_size:
            raise MEPQueueError(
                message=f"Queue full (max {settings.api.mep_queue_max_size})",
                queue_size=len(proposal_queue),
                max_size=settings.api.mep_queue_max_size
            )
        
        # Add timestamp and position info
        proposal["queued_at"] = datetime.utcnow()
        proposal["queue_position"] = len(proposal_queue) + 1
        
        proposal_queue.append(proposal)
        
        logger.info(f"Added proposal {proposal['proposal_id']} to queue (position: {len(proposal_queue)})")
        return len(proposal_queue)


async def process_proposal_background(proposal_data: Dict[str, Any]) -> None:
    """
    Background task to process MEP proposals.
    
    Args:
        proposal_data: Proposal data to process
        
    Note:
        This is a placeholder implementation for MVP.
        In production, this would integrate with the actual memory consolidation pipeline.
    """
    try:
        proposal_id = proposal_data.get("proposal_id", "unknown")
        
        logger.info(f"Processing proposal {proposal_id} in background")
        
        # Simulate processing time (replace with actual processing)
        await asyncio.sleep(5)  # 5 second simulation
        
        # Mark as processed (in production, update database status)
        proposal_data["status"] = "processed"
        proposal_data["processed_at"] = datetime.utcnow()
        
        logger.info(f"Completed processing proposal {proposal_id}")
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
        proposal_data["status"] = "failed"
        proposal_data["error"] = str(e)


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
        # Generate unique proposal ID
        proposal_id = str(uuid4())
        request_id = getattr(request.state, 'request_id', None)
        
        logger.info(
            f"Received MEP proposal: provider={proposal_request.provider}, "
            f"model={proposal_request.model}, trigger={proposal_request.trigger} "
            f"(Request ID: {request_id})"
        )
        
        # Convert to internal model for validation
        mep_proposal = proposal_request.to_mep_proposal()
        
        # Additional business logic validation
        if mep_proposal.context_fill > 0.95:
            logger.warning(f"High context fill ratio: {mep_proposal.context_fill}")
        
        if len(mep_proposal.key_facts) > 20:
            logger.warning(f"Large number of key facts: {len(mep_proposal.key_facts)}")
        
        # Prepare proposal data for queue
        proposal_data = {
            "proposal_id": proposal_id,
            "request_id": request_id,
            "proposal": mep_proposal.model_dump(),
            "status": "queued",
            "submitted_at": datetime.utcnow(),
        }
        
        # Add to processing queue
        try:
            queue_position = await add_to_queue(proposal_data)
        except MEPQueueError as e:
            logger.error(f"Queue full: {e}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=create_error_response(e, request_id)
            )
        
        # Start background processing
        background_tasks.add_task(process_proposal_background, proposal_data)
        
        # Estimate processing time based on queue position
        estimated_time = queue_position * 30  # 30 seconds per proposal estimate
        
        # Create success response
        response = MEPProposalSuccessResponse(
            proposal_id=proposal_id,
            status="accepted",
            message="Proposal accepted for processing",
            queue_position=queue_position,
            estimated_processing_time=estimated_time
        )
        
        logger.info(
            f"MEP proposal accepted: {proposal_id} "
            f"(Queue position: {queue_position}, ETA: {estimated_time}s)"
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
        # Search for proposal in queue
        async with queue_lock:
            for proposal_data in proposal_queue:
                if proposal_data.get("proposal_id") == proposal_id:
                    return {
                        "proposal_id": proposal_id,
                        "status": proposal_data.get("status", "unknown"),
                        "queue_position": proposal_data.get("queue_position"),
                        "submitted_at": proposal_data.get("submitted_at"),
                        "processed_at": proposal_data.get("processed_at"),
                        "error": proposal_data.get("error")
                    }
        
        # Proposal not found
        logger.warning(f"Proposal not found: {proposal_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=create_error_response(
                MEPError(f"Proposal {proposal_id} not found", proposal_id=proposal_id),
                getattr(request.state, 'request_id', None)
            )
        )
        
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
        
        # Check component health (placeholder for MVP)
        components = {
            "base_model": "healthy",  # Would check actual model status
            "embeddings": "healthy",  # Would check embeddings service
            "vector_store": "healthy",  # Would check FAISS vector store
            "queue": "healthy"  # Queue is always healthy for in-memory implementation
        }
        
        # Determine overall status
        overall_status = "healthy"
        if any(status != "healthy" for status in components.values()):
            overall_status = "degraded"
        
        # Performance metrics
        metrics = {
            "queue_size": len(proposal_queue),
            "active_requests": active_requests,
            "total_requests": request_counter,
            "uptime_seconds": uptime
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
        async with queue_lock:
            # Queue statistics
            total_proposals = len(proposal_queue)
            queued_proposals = sum(1 for p in proposal_queue if p.get("status") == "queued")
            processing_proposals = sum(1 for p in proposal_queue if p.get("status") == "processing")
            completed_proposals = sum(1 for p in proposal_queue if p.get("status") == "processed")
            failed_proposals = sum(1 for p in proposal_queue if p.get("status") == "failed")
            
            # Recent proposals (last hour)
            current_time = datetime.utcnow()
            recent_proposals = [
                p for p in proposal_queue
                if p.get("submitted_at") and 
                   (current_time - p["submitted_at"]).total_seconds() <= 3600
            ]
        
        return {
            "queue_size": total_proposals,
            "max_queue_size": settings.api.mep_queue_max_size,
            "utilization": total_proposals / settings.api.mep_queue_max_size,
            "status_breakdown": {
                "queued": queued_proposals,
                "processing": processing_proposals,
                "completed": completed_proposals,
                "failed": failed_proposals
            },
            "recent_proposals_1h": len(recent_proposals),
            "average_processing_time_estimate": 30,  # seconds
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Error getting queue status: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=create_error_response(e, getattr(request.state, 'request_id', None))
        )
    finally:
        await cleanup_request_metrics(request)


# Export router for main application
__all__ = ["router"]