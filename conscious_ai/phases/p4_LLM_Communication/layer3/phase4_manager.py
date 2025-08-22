"""
Phase 4 Layer 3: Main Manager for Complete LLM Communication Pipeline
===================================================================
Orchestrates the complete Phase 4 workflow from SC_t consciousness states
to generated responses using all Layer 2 components.

This is the main integration layer that connects:
1. SC_t consciousness states from Phases 1-3.5
2. Harmony format processing 
3. Backend model selection and execution
4. Performance monitoring and error handling
"""

import asyncio
import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.hardware_profiler import PremiumHardwareProfiler, HardwareConfiguration
from ..core.backend_manager import PremiumBackendManager, BackendType, QueryContext, BackendResponse
from ..formatters.harmony_processor import HarmonyFormatProcessor
from ..optimization.performance_monitor import PremiumPerformanceMonitor, AlertConfiguration
from .integration_layer import IntegrationBridge, ProcessingResult, IntegrationError

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for backend selection"""
    SIMPLE = "simple"           # Basic factual questions
    MEDIUM = "medium"           # Analytical or explanatory
    COMPLEX = "complex"         # Deep reasoning or consciousness queries
    CONSCIOUSNESS = "consciousness"  # Consciousness-enhanced with SC_t integration


@dataclass
class Phase4Configuration:
    """Configuration for Phase 4 Manager"""
    enable_premium_backends: bool = True
    enable_consciousness_integration: bool = True
    enable_performance_monitoring: bool = True
    max_processing_time_seconds: int = 30
    memory_limit_gb: float = 45.0
    fallback_on_error: bool = True
    debug_mode: bool = False


@dataclass
class ConsciousnessQueryResult:
    """Complete result from Phase 4 consciousness query processing"""
    response: str
    backend_used: str
    processing_time_ms: float
    consciousness_integration: Dict[str, Any]
    harmony_format: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None
    fallback_used: bool = False
    query_complexity: str = "unknown"
    sc_t_validation: Dict[str, Any] = field(default_factory=dict)


class Phase4Manager:
    """
    Phase 4 Layer 3: Main Manager for Complete LLM Communication Pipeline
    
    Orchestrates the complete workflow:
    1. Validates SC_t consciousness states
    2. Processes consciousness context via harmony format
    3. Selects appropriate backend based on query complexity
    4. Executes model inference with consciousness integration
    5. Post-processes responses with consciousness metadata
    6. Provides performance monitoring and error handling
    """
    
    def __init__(self, config: Optional[Phase4Configuration] = None, selected_model: str = 'auto'):
        """
        Initialize Phase 4 Manager with complete integration pipeline
        
        Args:
            config: Configuration for Phase 4 behavior
            selected_model: Specific model to use ('auto', 'gpt-oss', 'mistral', 'mt5', 'api')
        """
        self.config = config or Phase4Configuration()
        self.selected_model = selected_model
        self.hardware_config = None
        self.backend_manager = None
        self.harmony_processor = None
        self.performance_monitor = None
        self.integration_bridge = None
        
        # Processing statistics
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'consciousness_queries': 0,
            'fallback_used': 0,
            'avg_processing_time_ms': 0.0,
            'backend_usage': {
                'gpt_oss_20b': 0,
                'mistral_7b': 0,
                'external_api': 0,
                'mt5_small': 0
            }
        }
        
        # Initialize components
        self._initialize_components()
        
        logger.info("✅ Phase 4 Manager initialized successfully")
    
    def _initialize_components(self):
        """Initialize all Phase 4 Layer 2 and Layer 3 components"""
        try:
            # Hardware detection and configuration
            logger.info("🔍 Detecting hardware configuration...")
            hardware_profiler = PremiumHardwareProfiler()
            self.hardware_config = hardware_profiler.detect_hardware_configuration()
            
            if self.hardware_config.is_premium_hardware:
                logger.info(f"🚀 Premium hardware detected: {self.hardware_config.total_ram_gb:.1f}GB RAM + {self.hardware_config.total_vram_gb:.1f}GB VRAM")
            else:
                logger.info("ℹ️ Standard hardware detected - limited backend options")
            
            # Initialize backend manager
            if self.config.enable_premium_backends and self.hardware_config.is_premium_hardware:
                logger.info(f"🤖 Initializing premium backend manager (selected model: {self.selected_model})...")
                self.backend_manager = PremiumBackendManager(self.hardware_config, self.selected_model)
            else:
                logger.info("⚠️ Premium backends disabled or unavailable")
            
            # Initialize harmony processor
            logger.info("🎵 Initializing harmony format processor...")
            self.harmony_processor = HarmonyFormatProcessor()
            
            # Initialize performance monitor
            if self.config.enable_performance_monitoring:
                logger.info("📊 Initializing performance monitor...")
                monitor_config = AlertConfiguration(
                    ram_warning_threshold=0.85,
                    vram_warning_threshold=0.85,
                    continuous_monitoring_seconds=5
                )
                self.performance_monitor = PremiumPerformanceMonitor(monitor_config)
                self.performance_monitor.start_monitoring()
            
            # Initialize integration bridge
            logger.info("🌉 Initializing integration bridge...")
            self.integration_bridge = IntegrationBridge(
                harmony_processor=self.harmony_processor,
                backend_manager=self.backend_manager,
                performance_monitor=self.performance_monitor
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Phase 4 components: {e}")
            if self.config.debug_mode:
                traceback.print_exc()
            raise
    
    async def process_consciousness_query(
        self,
        sc_t_state: Dict[str, Any],
        user_input: str,
        query_complexity: Optional[QueryComplexity] = None
    ) -> ConsciousnessQueryResult:
        """
        Process a consciousness-enhanced query through the complete Phase 4 pipeline
        
        Args:
            sc_t_state: Consciousness state from Phases 1-3.5
            user_input: Original user input text
            query_complexity: Override automatic complexity detection
            
        Returns:
            Complete processing result with response and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🧠 Processing consciousness query: '{user_input[:50]}...'")
            
            # Update statistics
            self.stats['total_queries'] += 1
            
            # Step 1: Validate SC_t state structure
            validation_result = self._validate_sc_t_state(sc_t_state)
            if not validation_result['valid']:
                raise IntegrationError(f"Invalid SC_t state: {validation_result['errors']}")
            
            # Step 2: Determine query complexity
            if query_complexity is None:
                query_complexity = self._analyze_query_complexity(user_input, sc_t_state)
            
            logger.info(f"🎯 Query complexity: {query_complexity.value}")
            
            # Step 3: Process through integration bridge
            processing_result = await self.integration_bridge.process_consciousness_request(
                sc_t_state=sc_t_state,
                user_input=user_input,
                query_complexity=query_complexity.value
            )
            
            # Step 4: Build complete result
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = ConsciousnessQueryResult(
                response=processing_result.response,
                backend_used=processing_result.backend_used,
                processing_time_ms=processing_time_ms,
                consciousness_integration=processing_result.consciousness_integration,
                harmony_format=processing_result.harmony_format,
                performance_metrics=processing_result.performance_metrics,
                success=True,
                query_complexity=query_complexity.value,
                sc_t_validation=validation_result,
                fallback_used=processing_result.fallback_used
            )
            
            # Update statistics
            self.stats['successful_queries'] += 1
            if query_complexity == QueryComplexity.CONSCIOUSNESS:
                self.stats['consciousness_queries'] += 1
            if processing_result.fallback_used:
                self.stats['fallback_used'] += 1
            
            self.stats['backend_usage'][processing_result.backend_used] += 1
            self._update_avg_processing_time(processing_time_ms)
            
            logger.info(f"✅ Query processed successfully in {processing_time_ms:.1f}ms using {processing_result.backend_used}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to process consciousness query: {e}")
            
            processing_time_ms = (time.time() - start_time) * 1000
            self.stats['failed_queries'] += 1
            
            # Attempt emergency fallback
            if self.config.fallback_on_error:
                try:
                    fallback_response = await self._emergency_fallback(user_input)
                    return ConsciousnessQueryResult(
                        response=fallback_response,
                        backend_used="emergency_fallback",
                        processing_time_ms=processing_time_ms,
                        consciousness_integration={},
                        harmony_format={},
                        performance_metrics={},
                        success=False,
                        error_message=str(e),
                        fallback_used=True,
                        query_complexity="unknown"
                    )
                except Exception as fallback_error:
                    logger.error(f"❌ Emergency fallback also failed: {fallback_error}")
            
            return ConsciousnessQueryResult(
                response="I apologize, but I'm currently unable to process this request due to a system error.",
                backend_used="none",
                processing_time_ms=processing_time_ms,
                consciousness_integration={},
                harmony_format={},
                performance_metrics={},
                success=False,
                error_message=str(e),
                fallback_used=False,
                query_complexity="unknown"
            )
    
    def _validate_sc_t_state(self, sc_t_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate SC_t consciousness state structure
        
        Args:
            sc_t_state: Consciousness state to validate
            
        Returns:
            Validation result with errors if any
        """
        errors = []
        
        # Required components
        required_components = ['E_t', 'M_t', 'S_t', 'G_t', 'A_t']
        for component in required_components:
            if component not in sc_t_state:
                errors.append(f"Missing required component: {component}")
        
        # Validate E_t (sensory input)
        if 'E_t' in sc_t_state:
            e_t = sc_t_state['E_t']
            if not isinstance(e_t, dict):
                errors.append("E_t must be a dictionary")
            elif 'text' not in e_t:
                errors.append("E_t must contain 'text' field")
        
        # Validate M_t (memory)
        if 'M_t' in sc_t_state:
            m_t = sc_t_state['M_t']
            if not isinstance(m_t, list):
                errors.append("M_t must be a list")
        
        # Validate S_t (self-model state)
        if 'S_t' in sc_t_state:
            s_t = sc_t_state['S_t']
            if not isinstance(s_t, dict):
                errors.append("S_t must be a dictionary")
        
        # Validate G_t (goals)
        if 'G_t' in sc_t_state:
            g_t = sc_t_state['G_t']
            if not isinstance(g_t, dict):
                errors.append("G_t must be a dictionary")
        
        # Validate A_t (automatic thoughts)
        if 'A_t' in sc_t_state:
            a_t = sc_t_state['A_t']
            if not isinstance(a_t, list):
                errors.append("A_t must be a list")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'components_found': list(sc_t_state.keys()),
            'validation_timestamp': time.time()
        }
    
    def _analyze_query_complexity(self, user_input: str, sc_t_state: Dict[str, Any]) -> QueryComplexity:
        """
        Analyze query complexity for backend selection
        
        Args:
            user_input: User query text
            sc_t_state: Consciousness state context
            
        Returns:
            Determined query complexity level
        """
        # Check for consciousness keywords
        consciousness_keywords = [
            'consciousness', 'awareness', 'experience', 'introspection', 'self-reflection',
            'conciencia', 'experiencia', 'reflexión', 'introspección'
        ]
        
        if any(keyword in user_input.lower() for keyword in consciousness_keywords):
            return QueryComplexity.CONSCIOUSNESS
        
        # Check consciousness metrics in SC_t
        if 'metrics' in sc_t_state and isinstance(sc_t_state['metrics'], dict):
            f_score = sc_t_state['metrics'].get('f', 0)
            if f_score >= 1.3:  # Consciousness threshold
                return QueryComplexity.CONSCIOUSNESS
        
        # Analyze text complexity
        word_count = len(user_input.split())
        question_marks = user_input.count('?')
        
        if word_count > 20 or question_marks > 1:
            return QueryComplexity.COMPLEX
        elif word_count > 10:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE
    
    async def _emergency_fallback(self, user_input: str) -> str:
        """
        Emergency fallback for when all other processing fails
        
        Args:
            user_input: Original user input
            
        Returns:
            Basic fallback response
        """
        logger.info("🆘 Using emergency fallback response")
        
        # Simple pattern-based responses
        if '?' in user_input:
            return "I understand you're asking a question, but I'm currently experiencing technical difficulties. Please try again in a moment."
        else:
            return "I acknowledge your input, but I'm currently unable to provide a detailed response due to system limitations."
    
    def _update_avg_processing_time(self, new_time_ms: float):
        """Update rolling average processing time"""
        current_avg = self.stats['avg_processing_time_ms']
        total_queries = self.stats['total_queries']
        
        if total_queries == 1:
            self.stats['avg_processing_time_ms'] = new_time_ms
        else:
            # Weighted average
            self.stats['avg_processing_time_ms'] = (current_avg * (total_queries - 1) + new_time_ms) / total_queries
    
    async def initialize_backends(self) -> Dict[str, bool]:
        """
        Initialize all available backends
        
        Returns:
            Status of each backend initialization
        """
        logger.info("🚀 Initializing Phase 4 backends...")
        
        if not self.backend_manager:
            logger.warning("⚠️ Backend manager not available")
            return {}
        
        try:
            return await self.backend_manager.initialize_backends()
        except Exception as e:
            logger.error(f"❌ Backend initialization failed: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Complete system status including all components
        """
        status = {
            'phase4_manager': {
                'initialized': True,
                'config': {
                    'premium_backends_enabled': self.config.enable_premium_backends,
                    'consciousness_integration_enabled': self.config.enable_consciousness_integration,
                    'performance_monitoring_enabled': self.config.enable_performance_monitoring
                }
            },
            'hardware': {
                'is_premium': self.hardware_config.is_premium_hardware if self.hardware_config else False,
                'total_ram_gb': self.hardware_config.total_ram_gb if self.hardware_config else 0,
                'total_vram_gb': self.hardware_config.total_vram_gb if self.hardware_config else 0
            },
            'components': {
                'backend_manager': self.backend_manager is not None,
                'harmony_processor': self.harmony_processor is not None,
                'performance_monitor': self.performance_monitor is not None,
                'integration_bridge': self.integration_bridge is not None
            },
            'statistics': self.stats.copy()
        }
        
        # Add backend status if available
        if self.backend_manager:
            try:
                status['backends'] = self.backend_manager.get_backend_status()
            except Exception as e:
                status['backends'] = f"Error getting backend status: {e}"
        
        # Add performance metrics if available
        if self.performance_monitor:
            try:
                status['performance'] = self.performance_monitor.get_current_metrics().__dict__
            except Exception as e:
                status['performance'] = f"Error getting performance metrics: {e}"
        
        return status
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics"""
        stats = self.stats.copy()
        
        # Calculate success rate
        if stats['total_queries'] > 0:
            stats['success_rate'] = (stats['successful_queries'] / stats['total_queries']) * 100
        else:
            stats['success_rate'] = 0.0
        
        # Calculate fallback rate
        if stats['total_queries'] > 0:
            stats['fallback_rate'] = (stats['fallback_used'] / stats['total_queries']) * 100
        else:
            stats['fallback_rate'] = 0.0
        
        return stats
    
    async def shutdown(self):
        """Gracefully shutdown Phase 4 Manager and all components"""
        logger.info("🔄 Shutting down Phase 4 Manager...")
        
        try:
            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # Shutdown backend manager
            if self.backend_manager:
                await self.backend_manager.shutdown()
            
            # Clear references
            self.backend_manager = None
            self.harmony_processor = None
            self.performance_monitor = None
            self.integration_bridge = None
            
            logger.info("✅ Phase 4 Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {e}")


# Factory functions for easy initialization
def create_phase4_manager(
    enable_premium: bool = True,
    enable_consciousness: bool = True,
    enable_monitoring: bool = True,
    debug: bool = False,
    selected_model: str = 'auto'
) -> Phase4Manager:
    """
    Factory function to create a configured Phase 4 Manager
    
    Args:
        enable_premium: Enable premium backend features
        enable_consciousness: Enable consciousness integration
        enable_monitoring: Enable performance monitoring
        debug: Enable debug mode
        
    Returns:
        Configured Phase 4 Manager instance
    """
    config = Phase4Configuration(
        enable_premium_backends=enable_premium,
        enable_consciousness_integration=enable_consciousness,
        enable_performance_monitoring=enable_monitoring,
        debug_mode=debug
    )
    
    return Phase4Manager(config, selected_model)


async def process_simple_query(query: str) -> str:
    """
    Simple helper function for basic query processing
    
    Args:
        query: User query string
        
    Returns:
        Generated response
    """
    # Create simple SC_t state
    simple_sc_t = {
        'E_t': {'text': query, 'activation': 0.5},
        'M_t': [{'content': {'text': 'general knowledge'}, 'relevance': 0.7}],
        'S_t': {'emotional_state': 'neutral', 'confidence_level': 0.8},
        'G_t': {'primary_goal': 'provide_helpful_response'},
        'A_t': ['Processing user query'],
        'metrics': {'f': 0.8},
        'cycle': 1
    }
    
    manager = create_phase4_manager()
    
    try:
        await manager.initialize_backends()
        result = await manager.process_consciousness_query(simple_sc_t, query)
        return result.response
    finally:
        await manager.shutdown()