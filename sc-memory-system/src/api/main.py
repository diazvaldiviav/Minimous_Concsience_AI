"""
Main FastAPI application for SC Memory System.

This module sets up the FastAPI application with all routes, middleware,
error handlers, and configuration. Includes MEP API endpoints, health checks,
and comprehensive logging and monitoring.
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from ..core.config import Settings, get_settings
from ..core.exceptions import SCMemoryException
from ..core.models import ErrorDetail
from .mep import router as mep_router
from .map import router as map_router, initialize_map_components, cleanup_map_components

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger(__name__)


def convert_datetime_to_string(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO strings in a dict/list structure."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_datetime_to_string(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_string(item) for item in obj]
    else:
        return obj


# Global application state
app_start_time = time.time()
request_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Handles startup and shutdown tasks including model loading,
    resource initialization, and cleanup.
    """
    # Startup
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {'production' if settings.is_production else 'development'}")
    
    try:
        # Initialize components
        logger.info("Initializing application components...")
        
        # Initialize MAP API components (non-blocking)
        try:
            await initialize_map_components(settings)
            logger.info("MAP API components initialized successfully")
        except Exception as e:
            logger.error(f"MAP API initialization failed: {e}")
            logger.info("Continuing without MAP API - MEP API will still work")
        
        # TODO: Initialize other components
        # base_model = BaseModelManager(settings=settings)
        # await base_model.load_model()
        
        # TODO: Initialize embeddings manager
        # embeddings = EmbeddingsManager(settings=settings)
        # await embeddings.load_model()
        
        # TODO: Initialize vector store
        # vector_store = VectorStore(settings=settings)
        # await vector_store.initialize()
        
        logger.info("Application startup completed successfully")
        
        # Store components in app state
        # app.state.base_model = base_model
        # app.state.embeddings = embeddings
        # app.state.vector_store = vector_store
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise
    
    finally:
        # Shutdown
        logger.info("Shutting down application...")
        
        # Cleanup MAP API components
        await cleanup_map_components()
        
        # TODO: Cleanup other resources
        # if hasattr(app.state, 'base_model'):
        #     app.state.base_model.unload_model()
        # if hasattr(app.state, 'embeddings'):
        #     app.state.embeddings.unload_model()
        # if hasattr(app.state, 'vector_store'):
        #     await app.state.vector_store.save_index()
        
        logger.info("Application shutdown completed")


def create_app(settings: Settings = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Application settings (uses default if None)
        
    Returns:
        Configured FastAPI application
    """
    if settings is None:
        settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=settings.app_description,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )
    
    # Add middleware
    setup_middleware(app, settings)
    
    # Add routes
    setup_routes(app, settings)
    
    # Add exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI, settings: Settings) -> None:
    """Setup application middleware."""
    
    # CORS middleware
    if settings.api.cors_allow_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_allow_origins,
            allow_credentials=settings.api.cors_allow_credentials,
            allow_methods=settings.api.cors_allow_methods,
            allow_headers=["*"],
        )
    
    # Trusted host middleware for production
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"]  # Configure properly for production
        )
    
    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        global request_count
        request_count += 1
        
        start_time = time.time()
        
        # Add request ID for tracing
        request_id = f"req_{request_count:06d}"
        
        logger.info(
            f"Request started: {request.method} {request.url.path} "
            f"(ID: {request_id}, Client: {request.client.host if request.client else 'unknown'})"
        )
        
        try:
            response = await call_next(request)
            
            process_time = time.time() - start_time
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"(ID: {request_id}, Status: {response.status_code}, "
                f"Duration: {process_time:.3f}s)"
            )
            
            # Add response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Request failed: {request.method} {request.url.path} "
                f"(ID: {request_id}, Duration: {process_time:.3f}s, Error: {e})",
                exc_info=True
            )
            raise


def setup_routes(app: FastAPI, settings: Settings) -> None:
    """Setup application routes."""
    
    # Root route
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with basic application information."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "description": settings.app_description,
            "status": "healthy",
            "uptime_seconds": time.time() - app_start_time,
            "mep_api": f"{settings.api.mep_prefix}",
            "docs_url": "/docs" if not settings.is_production else None
        }
    
    # Health check route
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time() - app_start_time,
            "version": settings.app_version,
        }
    
    # Include MEP router
    app.include_router(
        mep_router,
        prefix=settings.api.mep_prefix,
        tags=["MEP"]
    )
    
    # Include MAP router
    app.include_router(
        map_router,
        tags=["MAP"]
    )
    
    logger.info(f"Routes configured: MEP API at {settings.api.mep_prefix}, MAP API at /map/v1")


def setup_exception_handlers(app: FastAPI) -> None:
    """Setup global exception handlers."""
    
    @app.exception_handler(SCMemoryException)
    async def sc_memory_exception_handler(request: Request, exc: SCMemoryException):
        """Handle SC Memory System specific exceptions."""
        logger.error(f"SC Memory exception: {exc.error_code} - {exc.message}")
        
        status_code_map = {
            "SC_AUTH_ERROR": status.HTTP_401_UNAUTHORIZED,
            "SC_AUTHZ_ERROR": status.HTTP_403_FORBIDDEN,
            "SC_VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "SC_MEP_VALIDATION_ERROR": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "SC_MEP_QUEUE_ERROR": status.HTTP_429_TOO_MANY_REQUESTS,
            "SC_CONFIG_ERROR": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "SC_MODEL_LOAD_ERROR": status.HTTP_503_SERVICE_UNAVAILABLE,
            "SC_SERVICE_UNAVAILABLE": status.HTTP_503_SERVICE_UNAVAILABLE,
            "SC_TIMEOUT_ERROR": status.HTTP_408_REQUEST_TIMEOUT,
            "SC_MEMORY_ERROR": status.HTTP_507_INSUFFICIENT_STORAGE,
        }
        
        status_code = status_code_map.get(exc.error_code, status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.to_dict(),
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        logger.error(f"Validation error: {exc}")
        
        # Format errors for client
        formatted_errors = []
        for error in exc.errors():
            formatted_errors.append({
                "field": ".".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"],
                "input": error.get("input")
            })
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": {
                    "error_code": "VALIDATION_ERROR",
                    "error_type": "RequestValidationError",
                    "message": "Request validation failed",
                    "details": formatted_errors
                },
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle FastAPI HTTP exceptions."""
        logger.error(f"HTTP exception: {exc.status_code} - {exc.detail}")

        # Convert detail to JSON-serializable format
        detail = exc.detail
        if hasattr(detail, 'model_dump'):
            # If it's a Pydantic model, serialize it
            detail = detail.model_dump(mode='json')
        elif isinstance(detail, dict):
            # Recursively convert datetime objects to strings
            detail = convert_datetime_to_string(detail)

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "error_code": f"HTTP_{exc.status_code}",
                    "error_type": "HTTPException",
                    "message": detail,
                    "status_code": exc.status_code
                },
                "timestamp": time.time()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "error_type": exc.__class__.__name__,
                    "message": "Internal server error occurred",
                    "details": str(exc) if not get_settings().is_production else None
                },
                "timestamp": time.time()
            }
        )


# Create the application instance
settings = get_settings()
app = create_app(settings)


def main():
    """
    Main entry point for running the application.
    
    This function is used by the CLI script defined in pyproject.toml.
    """
    settings = get_settings()
    
    # Configure uvicorn
    uvicorn_config = {
        "host": settings.api.host,
        "port": settings.api.port,
        "reload": settings.api.reload and settings.debug,
        "log_level": settings.log_level.lower(),
        "access_log": True,
    }
    
    logger.info(f"Starting server at http://{settings.api.host}:{settings.api.port}")
    logger.info(f"MEP API available at: http://{settings.api.host}:{settings.api.port}{settings.api.mep_prefix}")
    
    if not settings.is_production:
        logger.info(f"API documentation at: http://{settings.api.host}:{settings.api.port}/docs")
    
    # Run the server
    uvicorn.run("src.api.main:app", **uvicorn_config)


if __name__ == "__main__":
    main()


# Export for external use
__all__ = ["app", "main", "create_app"]