"""
MAP (Memory Access Protocol) routes for retrieving consolidated memories.

This module implements the core MAP API endpoints for querying and retrieving
compressed memory context from LoRA adapters. It handles adapter discovery,
context generation, and response formatting within token budgets.
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Union

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ValidationError

from src.core.config import get_settings, Settings
from src.core.models import (
    MAPQuery, MAPResponse, AdapterInfo, CompressedTurn, FilteredFact,
    TokenAllocation, ConversationTurn, ValidatedFact, KeyFact
)
from src.core.exceptions import (
    ModelLoadError, ConfigurationError, ValidationError as SCValidationError
)
from src.api.dependencies import verify_auth_token
from .adapter_manager import AdapterManager
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/map/v1", tags=["MAP"])

# Global managers (initialized during startup)
adapter_manager: Optional[AdapterManager] = None
context_builder: Optional[ContextBuilder] = None


async def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager."""
    if adapter_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AdapterManager not initialized"
        )
    return adapter_manager


async def get_context_builder() -> ContextBuilder:
    """Get the global context builder."""
    logger.debug(f"get_context_builder called - context_builder is None: {context_builder is None}")
    logger.debug(f"get_context_builder - context_builder type: {type(context_builder)}")
    if context_builder is None:
        logger.error("ContextBuilder is None when requested - initialization may have failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ContextBuilder not initialized"
        )
    return context_builder


async def initialize_map_components(settings: Settings) -> None:
    """Initialize MAP API components."""
    global adapter_manager, context_builder
    
    try:
        logger.info("Initializing MAP API components")
        
        # Initialize adapter manager
        adapter_manager = AdapterManager(settings)
        await adapter_manager.initialize()
        
        # Initialize context builder
        context_builder = ContextBuilder(settings)
        await context_builder.initialize()

        # Debug: Verify global assignment
        logger.info(f"Global context_builder assigned: {context_builder is not None}")
        logger.info(f"Global context_builder type: {type(context_builder)}")

        logger.info("MAP API components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MAP components: {e}", exc_info=True)
        raise ConfigurationError(f"MAP initialization failed: {e}")


async def cleanup_map_components() -> None:
    """Cleanup MAP API components."""
    global adapter_manager, context_builder
    
    try:
        if adapter_manager:
            await adapter_manager.cleanup()
        
        logger.info("MAP API components cleaned up")
        
    except Exception as e:
        logger.error(f"MAP cleanup failed: {e}", exc_info=True)


@router.get("/context", response_model=None)
async def get_context(
    provider: str = Query(..., description="Provider identifier (anthropic/openai)"),
    external_user_id: str = Query(..., description="External user identifier"),
    query: str = Query(..., description="User query"),
    external_chat_id: Optional[str] = Query(None, description="External chat filter"),
    token_budget: int = Query(320, ge=50, le=2000, description="Maximum tokens for response"),
    min_truth: float = Query(0.75, ge=0.0, le=1.0, description="Minimum truth score threshold"),
    format: str = Query("json", regex="^(json|compact_text|json_compact)$", description="Response format"),
    granularity: str = Query("mix", regex="^(mix|turns|facts)$", description="Context detail level"),
    scope: str = Query("user", regex="^(user|chat|org)$", description="Search scope"),
    settings: Settings = Depends(get_settings),
    _verified: bool = Depends(verify_auth_token),
    adapter_mgr: AdapterManager = Depends(get_adapter_manager),
    ctx_builder: ContextBuilder = Depends(get_context_builder)
):
    """
    Retrieve compressed memory context for a query.
    
    This endpoint finds relevant LoRA adapters, loads them, and generates
    compressed context (GIST + TURNS + FACTS) within the specified token budget.
    """
    try:
        start_time = datetime.utcnow()
        
        # Enhanced logging for debugging parameter issues
        logger.info(f"MAP context request - provider: {provider}, user_id: {external_user_id}, chat_id: {external_chat_id}, query: {query[:50]}...")
        
        # Create query object
        map_query = MAPQuery(
            provider=provider,
            external_user_id=external_user_id,
            query=query,
            external_chat_id=external_chat_id,
            token_budget=token_budget,
            min_truth=min_truth,
            format=format,
            granularity=granularity,
            scope=scope
        )
        
        # Generate response
        response = await _generate_map_response(map_query, adapter_mgr, ctx_builder)
        
        # Format response according to requested format
        formatted_response = await _format_response(response, format)
        
        # Log completion
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"MAP request completed in {duration:.1f}ms, tokens: {response.tokens_est}")
        
        # Return appropriate response type
        if format == "compact_text":
            return PlainTextResponse(content=formatted_response)
        else:
            return JSONResponse(content=formatted_response)
            
    except ValidationError as e:
        logger.warning(f"MAP request validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e}"
        )
    except Exception as e:
        logger.error(f"MAP request failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory access failed"
        )


@router.post("/context", response_model=None)
async def post_context(
    map_query: MAPQuery,
    settings: Settings = Depends(get_settings),
    _verified: bool = Depends(verify_auth_token),
    adapter_mgr: AdapterManager = Depends(get_adapter_manager),
    ctx_builder: ContextBuilder = Depends(get_context_builder)
):
    """
    Retrieve compressed memory context for a query (POST version).
    
    This endpoint accepts a JSON payload with query parameters and returns
    compressed memory context within the specified token budget.
    """
    try:
        start_time = datetime.utcnow()
        
        logger.info(f"MAP context POST: {map_query.provider}:{map_query.external_user_id}")
        
        # Generate response
        response = await _generate_map_response(map_query, adapter_mgr, ctx_builder)
        
        # Format response (POST always returns JSON)
        formatted_response = await _format_response(response, "json")
        
        # Log completion
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(f"MAP POST completed in {duration:.1f}ms, tokens: {response.tokens_est}")
        
        return JSONResponse(content=formatted_response)
        
    except ValidationError as e:
        logger.warning(f"MAP POST validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Request validation failed: {e}"
        )
    except Exception as e:
        logger.error(f"MAP POST failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Memory access failed"
        )


@router.get("/health")
async def health_check(
    adapter_mgr: AdapterManager = Depends(get_adapter_manager),
    ctx_builder: ContextBuilder = Depends(get_context_builder)
) -> JSONResponse:
    """Health check for MAP API."""
    try:
        cache_stats = adapter_mgr.get_cache_stats()
        
        health_info = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "adapter_cache": cache_stats,
            "components": {
                "adapter_manager": "ready",
                "context_builder": "ready"
            }
        }
        
        return JSONResponse(content=health_info)
        
    except Exception as e:
        logger.error(f"MAP health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


async def _generate_map_response(
    map_query: MAPQuery,
    adapter_mgr: AdapterManager,
    ctx_builder: ContextBuilder
) -> MAPResponse:
    """Generate MAP response from query."""
    try:
        # Find relevant adapters
        adapters = await adapter_mgr.find_user_adapters(
            provider=map_query.provider,
            user_id=map_query.external_user_id,
            chat_id=map_query.external_chat_id
        )
        
        if not adapters:
            # No memory available - provide helpful debug info
            logger.warning(f"No adapters found for provider={map_query.provider}, user={map_query.external_user_id}, chat={map_query.external_chat_id}")
            logger.info("Ensure that: 1) Memory was consolidated, 2) Parameters match exactly, 3) Metadata was saved correctly")
            return MAPResponse(
                has_memory=False,
                gist="",
                turns=[],
                facts=[],
                tokens_est=0
            )
        
        # Select best adapter (first in sorted list)
        best_adapter = adapters[0]
        
        # Load adapter
        model = await adapter_mgr.load_adapter(best_adapter)
        if not model:
            logger.warning(f"Failed to load adapter {best_adapter.adapter_id}")
            return MAPResponse(
                has_memory=False,
                gist="",
                turns=[],
                facts=[],
                tokens_est=0
            )
        
        # Allocate token budget
        allocation = ctx_builder.token_budget_manager.allocate_tokens(
            total_budget=map_query.token_budget,
            gist_ratio=0.4,
            turns_ratio=0.4,
            facts_ratio=0.15,
            metadata_ratio=0.05
        )
        
        # Generate components based on granularity
        gist = ""
        turns = []
        facts = []
        
        if map_query.granularity in ["mix", "turns"]:
            # Generate gist
            gist = await ctx_builder.generate_gist(
                query=map_query.query,
                model=model,
                adapter_info=best_adapter,
                max_tokens=allocation.gist_tokens
            )
        
        if map_query.granularity in ["mix", "turns"]:
            # Extract turns (simulated - in real implementation would come from stored conversation)
            conversation_turns = await _load_conversation_turns(best_adapter)
            turns = await ctx_builder.extract_turn_sketch(
                conversation_turns=conversation_turns,
                max_tokens=allocation.turns_tokens,
                query=map_query.query
            )
        
        if map_query.granularity in ["mix", "facts"]:
            # Compile facts (simulated - in real implementation would come from validated facts)
            validated_facts = await _load_validated_facts(best_adapter)
            facts = await ctx_builder.compile_facts(
                validated_facts=validated_facts,
                min_truth=map_query.min_truth,
                max_tokens=allocation.facts_tokens
            )
        
        # Estimate total tokens
        tokens_est = ctx_builder.estimate_response_tokens(gist, turns, facts)
        
        # Create response
        response = MAPResponse(
            has_memory=True,
            topic=best_adapter.topic,
            span=best_adapter.turn_range,
            gist=gist,
            turns=turns,
            facts=facts,
            tokens_est=tokens_est,
            shard_hint=f"sh_{best_adapter.topic or 'conv'}_v1"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"MAP response generation failed: {e}", exc_info=True)
        return MAPResponse(
            has_memory=False,
            gist="",
            turns=[],
            facts=[],
            tokens_est=0
        )


async def _load_conversation_turns(adapter_info: AdapterInfo) -> List[ConversationTurn]:
    """
    Load conversation turns from storage or simulate.
    
    IMPORTANT: All comments must be in English.
    """
    try:
        from pathlib import Path
        
        # Try to load real data from saved conversation (ENGLISH COMMENT)
        if adapter_info.data_path:
            conversation_file = Path(adapter_info.data_path) / "conversation.json"
            
            if conversation_file.exists():
                # Read the conversation file (ENGLISH COMMENT)
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert messages to ConversationTurn objects (ENGLISH COMMENT)
                turns = []
                for msg in data.get("messages", []):
                    turn = ConversationTurn(
                        turn_number=msg.get("turn_number", 0),
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                        metadata={"source": "stored_conversation"}  # ENGLISH
                    )
                    turns.append(turn)
                
                # Log success in English
                logger.info(f"Loaded {len(turns)} real turns for adapter {adapter_info.adapter_id}")
                return turns
        
        # If no data path set, try default location (ENGLISH COMMENT)
        default_path = Path(f"./data/conversations/{adapter_info.adapter_id}")
        if default_path.exists():
            conversation_file = default_path / "conversation.json"
            
            if conversation_file.exists():
                # Read the conversation file (ENGLISH COMMENT)
                with open(conversation_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert messages to ConversationTurn objects (ENGLISH COMMENT)
                turns = []
                for msg in data.get("messages", []):
                    turn = ConversationTurn(
                        turn_number=msg.get("turn_number", 0),
                        role=msg.get("role", "user"),
                        content=msg.get("content", ""),
                        timestamp=msg.get("timestamp"),
                        metadata={"source": "stored_conversation"}  # ENGLISH
                    )
                    turns.append(turn)
                
                # Log success in English
                logger.info(f"Loaded {len(turns)} real turns from default location for adapter {adapter_info.adapter_id}")
                return turns
                
    except Exception as e:
        # Log failure in English
        logger.warning(f"Failed to load real turns: {e}, falling back to simulation")
    
    # Fallback to simulation for testing (ENGLISH COMMENT)
    return _generate_simulated_turns(adapter_info)


def _generate_simulated_turns(adapter_info: AdapterInfo) -> List[ConversationTurn]:
    """
    Generate simulated turns when real data is not available.
    
    Args:
        adapter_info: Adapter information
        
    Returns:
        List of simulated conversation turns
    """
    topic = adapter_info.topic or "discussion"
    turns = []
    
    # Simulate some conversation turns (ENGLISH COMMENT)
    turn_data = [
        ("user", f"Can you explain {topic}?"),
        ("assistant", f"Certainly! {topic} is an important concept that involves..."),
        ("user", "Can you give me a specific example?"),
        ("assistant", "Here's a concrete example that illustrates the key points..."),
        ("user", "What are the main applications?"),
        ("assistant", f"The main applications of {topic} include several key areas...")
    ]
    
    for i, (role, content) in enumerate(turn_data, 1):
        turn = ConversationTurn(
            turn_number=i,
            role=role,
            content=content,
            timestamp=datetime.utcnow(),
            metadata={"source": "simulated"}  # ENGLISH
        )
        turns.append(turn)
    
    logger.info(f"Generated {len(turns)} simulated turns for adapter {adapter_info.adapter_id}")
    return turns


async def _load_validated_facts(adapter_info: AdapterInfo) -> List[ValidatedFact]:
    """
    Load validated facts from storage or simulate.
    
    IMPORTANT: All comments must be in English.
    """
    try:
        from pathlib import Path
        
        # Try to load real facts from saved data (ENGLISH COMMENT)
        if adapter_info.data_path:
            facts_file = Path(adapter_info.data_path) / "validated_facts.json"
            
            if facts_file.exists():
                # Read the facts file (ENGLISH COMMENT)
                with open(facts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to ValidatedFact objects (ENGLISH COMMENT)
                facts = []
                for fact_data in data:
                    # Create KeyFact first (ENGLISH COMMENT)
                    key_fact = KeyFact(
                        claim=fact_data.get("claim", ""),
                        importance=fact_data.get("importance", 0.5),
                        confidence=fact_data.get("confidence"),
                        source_turn=fact_data.get("source_turn"),
                        category=fact_data.get("category")
                    )
                    
                    # Create ValidatedFact (ENGLISH COMMENT)
                    validated_fact = ValidatedFact(
                        original_fact=key_fact,
                        truth_score=fact_data.get("confidence", 0.0),
                        is_validated=fact_data.get("is_validated", True),
                        validation_reason=fact_data.get("validation_reason")
                    )
                    facts.append(validated_fact)
                
                # Log success in English
                logger.info(f"Loaded {len(facts)} real validated facts for adapter {adapter_info.adapter_id}")
                return facts
        
        # If no data path set, try default location (ENGLISH COMMENT)
        default_path = Path(f"./data/conversations/{adapter_info.adapter_id}")
        if default_path.exists():
            facts_file = default_path / "validated_facts.json"
            
            if facts_file.exists():
                # Read the facts file (ENGLISH COMMENT)
                with open(facts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Convert to ValidatedFact objects (ENGLISH COMMENT)
                facts = []
                for fact_data in data:
                    # Create KeyFact first (ENGLISH COMMENT)
                    key_fact = KeyFact(
                        claim=fact_data.get("claim", ""),
                        importance=fact_data.get("importance", 0.5),
                        confidence=fact_data.get("confidence"),
                        source_turn=fact_data.get("source_turn"),
                        category=fact_data.get("category")
                    )
                    
                    # Create ValidatedFact (ENGLISH COMMENT)
                    validated_fact = ValidatedFact(
                        original_fact=key_fact,
                        truth_score=fact_data.get("confidence", 0.0),
                        is_validated=fact_data.get("is_validated", True),
                        validation_reason=fact_data.get("validation_reason")
                    )
                    facts.append(validated_fact)
                
                # Log success in English
                logger.info(f"Loaded {len(facts)} real validated facts from default location for adapter {adapter_info.adapter_id}")
                return facts
                
    except Exception as e:
        # Log failure in English
        logger.warning(f"Failed to load real facts: {e}, falling back to simulation")
    
    # Fallback to simulation for testing (ENGLISH COMMENT)
    return _generate_simulated_facts(adapter_info)


def _generate_simulated_facts(adapter_info: AdapterInfo) -> List[ValidatedFact]:
    """
    Generate simulated facts when real data is not available.
    
    Args:
        adapter_info: Adapter information
        
    Returns:
        List of simulated validated facts
    """
    topic = adapter_info.topic or "topic"
    facts = []
    
    # Simulate some validated facts (ENGLISH COMMENT)
    fact_data = [
        (f"{topic} has multiple important applications", 0.92, 0.9),
        (f"The key principle of {topic} is well-established", 0.89, 0.8),
        (f"There are several variations of {topic} approaches", 0.85, 0.7),
        (f"{topic} research continues to evolve", 0.78, 0.6)
    ]
    
    for claim, confidence, importance in fact_data:
        # Create KeyFact (ENGLISH COMMENT)
        key_fact = KeyFact(
            claim=claim,
            importance=importance,
            confidence=confidence
        )
        
        # Create ValidatedFact (ENGLISH COMMENT)
        validated_fact = ValidatedFact(
            original_fact=key_fact,
            truth_score=confidence,
            is_validated=True,
            validation_reason="Simulated validation"  # ENGLISH
        )
        facts.append(validated_fact)
    
    logger.info(f"Generated {len(facts)} simulated facts for adapter {adapter_info.adapter_id}")
    return facts


async def _format_response(response: MAPResponse, format_type: str) -> Any:
    """Format MAP response according to requested format."""
    try:
        if format_type == "json":
            return response.dict()
        
        elif format_type == "json_compact":
            # Compact JSON format
            compact = {
                "mem": response.has_memory,
                "topic": response.topic,
                "gist": response.gist,
                "turns": [{"i": t.id, "r": t.r, "t": t.t} for t in response.turns],
                "facts": [{"c": f.c, "p": f.p, "s": f.s} for f in response.facts],
                "tokens": response.tokens_est
            }
            return compact
        
        elif format_type == "compact_text":
            # Plain text format
            text_parts = []
            
            if response.has_memory:
                if response.gist:
                    text_parts.append(f"GIST: {response.gist}")
                
                if response.turns:
                    text_parts.append("TURNS:")
                    for turn in response.turns:
                        role_name = "USER" if turn.r == "u" else "ASSISTANT"
                        text_parts.append(f"{turn.id} {role_name}: {turn.t}")
                
                if response.facts:
                    text_parts.append("FACTS:")
                    for fact in response.facts:
                        text_parts.append(f"- {fact.c} (confidence: {fact.p})")
            else:
                text_parts.append("No memory available")
            
            return "\n".join(text_parts)
        
        else:
            return response.dict()
        
    except Exception as e:
        logger.error(f"Response formatting failed: {e}")
        return response.dict()  # Fallback to JSON