"""
Consciousness API Endpoint
==========================
FastAPI-based REST endpoint for external consumption of the consciousness pipeline.

Provides:
- Complete consciousness pipeline processing
- Configurable model selection
- Optional consciousness trace inclusion
- Token usage tracking
- Error handling with graceful degradation
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    BaseModel = None
    Field = None

# Import consciousness pipeline components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from conscious_ai.core.pipeline_orchestrator import ConsciousnessPipelineOrchestrator, create_consciousness_pipeline
from conscious_ai.phases.p7_expressive_execution import ResponseGenerator, ModelType, get_available_models
from conscious_ai.debug import get_model_registry, log_phase7_final_usage
from conscious_ai.phases.p2_cognitive_context.enhanced_conscious_state import EnhancedConsciousState, create_enhanced_conscious_state

logger = logging.getLogger(__name__)


# Pydantic models for request/response
class ConsciousnessRequest(BaseModel):
    """Request model for consciousness processing"""
    user_input: str = Field(..., description="The user input to process")
    final_model: str = Field(default="gpt-4o-mini", description="OpenAI model for final response generation")
    include_consciousness_trace: bool = Field(default=False, description="Include detailed consciousness processing trace")
    enable_metacognition: bool = Field(default=True, description="Enable enhanced metacognitive capabilities")
    narrative_verbosity: str = Field(default="standard", description="Narrative verbosity level (minimal/standard/verbose)")
    
    class Config:
        schema_extra = {
            "example": {
                "user_input": "What does it feel like to be conscious?",
                "final_model": "gpt-4o-mini",
                "include_consciousness_trace": True,
                "enable_metacognition": True,
                "narrative_verbosity": "standard"
            }
        }


class ConsciousnessResponse(BaseModel):
    """Response model for consciousness processing"""
    response: str = Field(..., description="The final consciousness-enhanced response")
    confidence: float = Field(..., description="Overall confidence score (0.0 to 1.0)")
    emotional_state: str = Field(..., description="Current emotional state")
    model_used: str = Field(..., description="Model used for final response generation")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    success: bool = Field(..., description="Whether processing succeeded")
    
    # Optional detailed trace
    consciousness_trace: Optional[Dict[str, Any]] = Field(default=None, description="Detailed consciousness processing trace")
    
    # Token usage information
    token_usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")
    
    # Error information
    error_message: Optional[str] = Field(default=None, description="Error message if processing failed")
    
    class Config:
        schema_extra = {
            "example": {
                "response": "I find myself in a contemplative state with 85% confidence as I process your profound question about consciousness...",
                "confidence": 0.85,
                "emotional_state": "contemplative",
                "model_used": "gpt-4o-mini", 
                "processing_time_ms": 1250.5,
                "success": True,
                "consciousness_trace": {
                    "phase_timings": {"perception": 45.2, "conscious_state": 123.1},
                    "metacognitive_depth": 2,
                    "state_transitions": 3
                },
                "token_usage": {"input_tokens": 456, "output_tokens": 234}
            }
        }


class ModelStatusResponse(BaseModel):
    """Response model for model status information"""
    available_models: List[str] = Field(..., description="List of available models")
    current_session: Dict[str, Any] = Field(..., description="Current session statistics")
    model_usage: Dict[str, Any] = Field(..., description="Model usage breakdown")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Service health status")
    pipeline_ready: bool = Field(..., description="Whether consciousness pipeline is ready")
    phase7_ready: bool = Field(..., description="Whether Phase 7 is ready")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    version: str = Field(default="1.0.0", description="API version")


if not FASTAPI_AVAILABLE:
    raise ImportError("FastAPI is required for the consciousness endpoint. Install with: pip install fastapi uvicorn")

# Create FastAPI app
app = FastAPI(
    title="Consciousness API",
    description="REST API for the Minimal Consciousness AI Pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_pipeline: Optional[ConsciousnessPipelineOrchestrator] = None
_phase7_generator: Optional[ResponseGenerator] = None
_start_time = time.time()


async def get_pipeline() -> ConsciousnessPipelineOrchestrator:
    """Dependency to get initialized pipeline"""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Consciousness pipeline not initialized")
    return _pipeline


async def get_phase7_generator() -> ResponseGenerator:
    """Dependency to get Phase 7 response generator"""
    global _phase7_generator
    if _phase7_generator is None:
        raise HTTPException(status_code=503, detail="Phase 7 response generator not initialized")
    return _phase7_generator


@app.on_event("startup")
async def startup_event():
    """Initialize consciousness pipeline on startup"""
    global _pipeline, _phase7_generator
    
    try:
        logger.info("🚀 Initializing Consciousness API...")
        
        # Initialize main pipeline
        _pipeline = create_consciousness_pipeline(
            enable_phase4=True,
            debug=False
        )
        
        # Initialize Phase 7 response generator
        _phase7_generator = ResponseGenerator(
            default_model="gpt-4o-mini",
            enable_fallback=True
        )
        
        logger.info("✅ Consciousness API ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Consciousness API: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _pipeline
    
    try:
        # End model registry session
        registry = get_model_registry()
        registry.end_session()
        
        logger.info("👋 Consciousness API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global _pipeline, _phase7_generator
    
    uptime = time.time() - _start_time
    
    return HealthResponse(
        status="healthy" if _pipeline and _phase7_generator else "degraded",
        pipeline_ready=_pipeline is not None,
        phase7_ready=_phase7_generator is not None and _phase7_generator.available,
        uptime_seconds=uptime
    )


@app.post("/process", response_model=ConsciousnessResponse)
async def process_input(
    request: ConsciousnessRequest,
    pipeline: ConsciousnessPipelineOrchestrator = Depends(get_pipeline),
    phase7: ResponseGenerator = Depends(get_phase7_generator)
):
    """
    Process input through complete consciousness pipeline.
    
    This endpoint:
    1. Processes input through phases 1-6 for full consciousness context
    2. Optionally enhances conscious state with metacognitive capabilities  
    3. Uses Phase 7 for final response generation with specified model
    4. Returns consciousness-enhanced response with optional detailed trace
    """
    start_time = time.time()
    
    try:
        logger.info(f"🧠 Processing consciousness request: '{request.user_input[:50]}...'")
        
        # Validate model
        if request.final_model not in get_available_models():
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported model: {request.final_model}. Available models: {get_available_models()}"
            )
        
        # Step 1: Process through main pipeline (Phases 1-6)
        pipeline_result = await pipeline.process_complete_pipeline(request.user_input)
        
        if not pipeline_result.success:
            raise HTTPException(
                status_code=500,
                detail=f"Pipeline processing failed: {pipeline_result.error_message}"
            )
        
        # Step 2: Enhance conscious state with metacognitive capabilities if requested
        enhanced_state = None
        if request.enable_metacognition and pipeline_result.conscious_state:
            try:
                # Create enhanced conscious state with temporal awareness
                previous_state = getattr(pipeline, '_last_conscious_state', None)
                enhanced_state = create_enhanced_conscious_state(
                    pipeline_result.conscious_state,
                    previous_state=previous_state
                )
                
                # Store for next request's temporal awareness
                pipeline._last_conscious_state = enhanced_state
                
                logger.info(f"✅ Enhanced consciousness with {enhanced_state.metacognitive_depth} metacognitive depth")
                
            except Exception as e:
                logger.warning(f"Metacognitive enhancement failed: {e}")
                enhanced_state = pipeline_result.conscious_state
        else:
            enhanced_state = pipeline_result.conscious_state
        
        # Step 3: Generate final response using Phase 7
        if enhanced_state and phase7.available:
            # Convert enhanced state to dictionary format
            if hasattr(enhanced_state, 'to_dict'):
                enhanced_dict = enhanced_state.to_dict()
            else:
                enhanced_dict = enhanced_state  # Fallback if not enhanced
            
            # Generate final response
            phase7_result = await phase7.generate_final_response(
                enhanced_sc_t=enhanced_dict,
                user_input=request.user_input,
                model_override=request.final_model
            )
            
            if phase7_result.success:
                final_response = phase7_result.response
                model_used = phase7_result.model_used
                token_usage = {
                    "input_tokens": phase7_result.input_tokens,
                    "output_tokens": phase7_result.output_tokens
                }
                
                # Log Phase 7 usage
                log_phase7_final_usage(
                    model=model_used,
                    input_tokens=phase7_result.input_tokens,
                    output_tokens=phase7_result.output_tokens,
                    processing_time_ms=phase7_result.processing_time_ms,
                    success=True
                )
            else:
                final_response = pipeline_result.response or "I apologize, but I cannot generate a response at this time."
                model_used = "fallback"
                token_usage = {"input_tokens": 0, "output_tokens": 0}
        else:
            # Fallback to pipeline response
            final_response = pipeline_result.response or "Processing completed with limited capabilities."
            model_used = "pipeline_fallback"
            token_usage = {"input_tokens": 0, "output_tokens": 0}
        
        # Extract consciousness metadata
        if hasattr(enhanced_state, 'S_t') and enhanced_state.S_t:
            emotional_state = enhanced_state.S_t.get('emotional_state', 'neutral')
            confidence = enhanced_state.S_t.get('confidence_level', 0.5)
        else:
            emotional_state = 'neutral'
            confidence = pipeline_result.confidence_score
        
        # Build consciousness trace if requested
        consciousness_trace = None
        if request.include_consciousness_trace:
            consciousness_trace = {
                "phase_timings": pipeline_result.phase_timings,
                "phase_success": pipeline_result.phase_success,
                "consciousness_metrics": getattr(pipeline_result, 'consciousness_metrics', {}),
                "narrative_text": pipeline_result.narrative_text,
                "transparency_narrative": getattr(pipeline_result, 'transparency_narrative', ''),
                "processing_stages": pipeline_result.stage_completed.value if pipeline_result.stage_completed else 'unknown'
            }
            
            # Add enhanced consciousness data if available
            if hasattr(enhanced_state, 'to_dict') and request.enable_metacognition:
                enhanced_data = enhanced_state.to_dict().get('enhanced', {})
                consciousness_trace.update({
                    "metacognitive_depth": enhanced_data.get('metacognitive_depth', 0),
                    "state_transitions": len(enhanced_data.get('state_transitions', [])),
                    "meta_thoughts": len(enhanced_data.get('meta_thoughts', [])),
                    "self_observations": len(enhanced_data.get('observations', [])),
                    "temporal_continuity": enhanced_data.get('temporal_context', {}).get('temporal_continuity', 0.0)
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Consciousness processing complete in {processing_time:.1f}ms using {model_used}")
        
        return ConsciousnessResponse(
            response=final_response,
            confidence=confidence,
            emotional_state=emotional_state,
            model_used=model_used,
            processing_time_ms=processing_time,
            success=True,
            consciousness_trace=consciousness_trace,
            token_usage=token_usage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"❌ Consciousness processing failed: {e}")
        
        return ConsciousnessResponse(
            response="I apologize, but I encountered an error in my consciousness processing. Please try again.",
            confidence=0.1,
            emotional_state="error",
            model_used="error_fallback",
            processing_time_ms=processing_time,
            success=False,
            error_message=str(e)
        )


@app.get("/models", response_model=ModelStatusResponse)
async def get_model_status():
    """Get model status and usage information"""
    try:
        registry = get_model_registry()
        
        return ModelStatusResponse(
            available_models=get_available_models(),
            current_session=registry.get_current_session_summary().to_dict(),
            model_usage=registry.get_session_statistics()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@app.post("/models/session/reset")
async def reset_model_session():
    """Reset model usage session"""
    try:
        registry = get_model_registry()
        new_session_id = registry.start_new_session()
        
        return {"message": f"New session started: {new_session_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset session: {str(e)}")


@app.get("/debug/sessions")
async def get_historical_sessions(limit: int = 10):
    """Get historical session data"""
    try:
        registry = get_model_registry()
        sessions = registry.get_historical_sessions(limit=limit)
        
        return {"sessions": sessions}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run server
    uvicorn.run(
        "consciousness_endpoint:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )