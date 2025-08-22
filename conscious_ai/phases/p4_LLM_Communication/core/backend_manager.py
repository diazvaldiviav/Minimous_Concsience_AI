"""
Premium Backend Manager for Phase 4 Layer 2
===========================================
Manages multiple LLM backends with intelligent routing and load balancing.
Supports GPT-OSS-20B + Mistral-7B simultaneous loading with automatic failover.
"""

import logging
import time
import asyncio
import threading
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, Future
import queue
import json

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None
    AutoModelForCausalLM = None
    T5Tokenizer = None
    T5ForConditionalGeneration = None
    AutoModelForSeq2SeqLM = None

from .hardware_profiler import PremiumHardwareProfiler, HardwareConfiguration

try:
    from ..models.gpt_oss_loader import HybridGPTOSSLoader, LoadingConfiguration, LoadingResult
    GPT_OSS_AVAILABLE = True
except ImportError:
    GPT_OSS_AVAILABLE = False
    HybridGPTOSSLoader = None
    LoadingConfiguration = None
    LoadingResult = None

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Available backend types in priority order"""
    PRIMARY_GPT_OSS = "gpt_oss_20b"
    SECONDARY_MISTRAL = "mistral_7b" 
    TERTIARY_API = "external_api"
    EMERGENCY_MT5 = "mt5_small"


class BackendStatus(Enum):
    """Backend operational status"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy" 
    ERROR = "error"
    OFFLINE = "offline"
    OVERLOADED = "overloaded"


@dataclass
class BackendMetrics:
    """Performance metrics for a backend"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    peak_memory_usage_gb: float = 0.0
    current_memory_usage_gb: float = 0.0
    uptime_hours: float = 0.0
    last_error: Optional[str] = None
    error_count_last_hour: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100.0
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        if self.total_requests == 0:
            return 100.0
            
        success_component = self.success_rate * 0.4
        speed_component = max(0, 100 - (self.avg_response_time_ms / 100)) * 0.3
        stability_component = max(0, 100 - (self.error_count_last_hour * 10)) * 0.3
        
        return min(100.0, success_component + speed_component + stability_component)


@dataclass
class QueryContext:
    """Context for intelligent query routing"""
    text: str
    complexity_score: float = 0.0
    estimated_tokens: int = 0
    priority: int = 1  # 1=low, 5=high
    timeout_seconds: int = 30
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    consciousness_state: Optional[Dict] = None
    
    def __post_init__(self):
        if self.estimated_tokens == 0:
            self.estimated_tokens = len(self.text.split()) * 1.3  # Rough token estimate
        if self.complexity_score == 0.0:
            self.complexity_score = self._calculate_complexity()
    
    def _calculate_complexity(self) -> float:
        """Calculate query complexity for routing decisions"""
        factors = {
            'length': min(1.0, len(self.text) / 1000),
            'questions': self.text.count('?') * 0.1,
            'technical_terms': len([w for w in self.text.split() if len(w) > 8]) * 0.05,
            'consciousness_context': 0.3 if self.consciousness_state else 0.0
        }
        return min(1.0, sum(factors.values()))


@dataclass
class BackendResponse:
    """Response from a backend with metadata"""
    text: str
    backend_used: BackendType
    response_time_ms: float
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class BackendInstance:
    """Individual backend instance with health monitoring"""
    
    def __init__(self, backend_type: BackendType, loader: Any = None):
        self.backend_type = backend_type
        self.loader = loader
        self.status = BackendStatus.INITIALIZING
        self.metrics = BackendMetrics()
        self.last_used = time.time()
        self.initialization_time = time.time()
        self._lock = threading.Lock()
        
    def process_query(self, context: QueryContext) -> BackendResponse:
        """Process query with this backend"""
        start_time = time.time()
        
        with self._lock:
            if self.status != BackendStatus.READY:
                return BackendResponse(
                    text="",
                    backend_used=self.backend_type,
                    response_time_ms=0,
                    success=False,
                    error_message=f"Backend not ready: {self.status.value}"
                )
            
            self.status = BackendStatus.BUSY
            
        try:
            # Route to appropriate processing method
            if self.backend_type == BackendType.PRIMARY_GPT_OSS:
                response_text = self._process_with_gpt_oss(context)
            elif self.backend_type == BackendType.SECONDARY_MISTRAL:
                response_text = self._process_with_mistral(context)
            elif self.backend_type == BackendType.TERTIARY_API:
                response_text = self._process_with_api(context)
            else:  # EMERGENCY_MT5
                response_text = self._process_with_mt5(context)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._update_success_metrics(response_time_ms)
            
            return BackendResponse(
                text=response_text,
                backend_used=self.backend_type,
                response_time_ms=response_time_ms,
                success=True,
                metadata={
                    'tokens_processed': context.estimated_tokens,
                    'complexity_score': context.complexity_score
                }
            )
            
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            self._update_error_metrics(str(e))
            
            return BackendResponse(
                text="",
                backend_used=self.backend_type,
                response_time_ms=response_time_ms,
                success=False,
                error_message=str(e)
            )
            
        finally:
            with self._lock:
                self.status = BackendStatus.READY
                self.last_used = time.time()
    
    def _process_with_gpt_oss(self, context: QueryContext) -> str:
        """Process with GPT-OSS-20B"""
        if not self.loader or not hasattr(self.loader, 'generate_text'):
            raise Exception("GPT-OSS loader not available")
        
        # Adjust parameters based on context
        max_tokens = min(200, context.estimated_tokens * 2)
        temperature = 0.7 if context.complexity_score > 0.5 else 0.5
        
        response = self.loader.generate_text(
            context.text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )
        
        return response or "I apologize, but I couldn't generate a response."
    
    def _process_with_mistral(self, context: QueryContext) -> str:
        """Process with Mistral-7B secondary backend"""
        try:
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                return "I apologize, but the Mistral backend is currently unavailable."
            
            # Prepare input for Mistral-7B
            prompt = f"<s>[INST] {context.text} [/INST]"
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"✅ Mistral-7B response generated ({len(response)} chars)")
            return response or "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"❌ Mistral-7B processing failed: {e}")
            return f"I apologize, but there was an error with the Mistral backend: {str(e)}"
    
    def _process_with_api(self, context: QueryContext) -> str:
        """Process with external API backend (OpenAI/Anthropic fallback)"""
        try:
            import os
            import requests
            import json
            
            # Check for API keys (prioritize OpenAI, then Anthropic)
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            
            if openai_key:
                return self._call_openai_api(context.text, openai_key)
            elif anthropic_key:
                return self._call_anthropic_api(context.text, anthropic_key)
            else:
                self.logger.warning("No API keys found - generating fallback response")
                return self._generate_fallback_response(context.text)
                
        except Exception as e:
            self.logger.error(f"❌ API backend processing failed: {e}")
            return self._generate_fallback_response(context.text)
    
    def _call_openai_api(self, text: str, api_key: str) -> str:
        """Call OpenAI API for response generation"""
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": text}],
                "max_tokens": 512,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                self.logger.error(f"OpenAI API error: {response.status_code}")
                return self._generate_fallback_response(text)
                
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return self._generate_fallback_response(text)
    
    def _call_anthropic_api(self, text: str, api_key: str) -> str:
        """Call Anthropic API for response generation"""
        try:
            import requests
            
            headers = {
                "x-api-key": api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 512,
                "messages": [{"role": "user", "content": text}]
            }
            
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['content'][0]['text'].strip()
            else:
                self.logger.error(f"Anthropic API error: {response.status_code}")
                return self._generate_fallback_response(text)
                
        except Exception as e:
            self.logger.error(f"Anthropic API call failed: {e}")
            return self._generate_fallback_response(text)
    
    def _generate_fallback_response(self, text: str) -> str:
        """Generate fallback response when APIs unavailable"""
        # Simple pattern-based fallback responses
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where']):
            return f"That's an interesting question about {text[:30]}... I'd need more context to provide a complete answer."
        elif any(word in text_lower for word in ['help', 'assist', 'support']):
            return "I'm here to help! Could you please provide more specific details about what you need assistance with?"
        elif any(word in text_lower for word in ['explain', 'describe', 'tell me']):
            return f"I understand you're asking about {text[:30]}... Let me provide what information I can share."
        else:
            return "I understand your request. While I cannot access external resources right now, I'm processing your input and doing my best to assist."
    
    def _process_with_mt5(self, context: QueryContext) -> str:
        """Process with emergency mT5 model (lightweight fallback)"""
        try:
            if not hasattr(self, 'model') or not hasattr(self, 'tokenizer'):
                logger.warning("mT5 backend not available - using built-in fallback")
                return self._built_in_emergency_response(context.text)
            
            # Prepare input for mT5 (text-to-text format)
            input_text = f"answer: {context.text}"
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=256,  # Smaller for emergency model
                padding=True
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,  # Conservative for emergency
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    num_beams=1,  # Fast generation
                    early_stopping=True
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True
            ).strip()
            
            # Clean up mT5 output
            response = response.replace(input_text, "").strip()
            
            logger.info(f"✅ Emergency mT5 response generated ({len(response)} chars)")
            return response or self._built_in_emergency_response(context.text)
            
        except Exception as e:
            logger.error(f"❌ mT5 emergency processing failed: {e}")
            return self._built_in_emergency_response(context.text)
    
    def _built_in_emergency_response(self, text: str) -> str:
        """Built-in emergency response when all models fail"""
        # Consciousness-aware emergency response
        responses = [
            "I'm experiencing technical difficulties accessing my full capabilities right now.",
            "My primary processing systems are temporarily unavailable, but I understand your query.",
            "I'm currently operating in emergency mode with limited functionality.",
            f"I received your message about {text[:30] if len(text) > 30 else text}... but I'm having technical issues.",
            "Thank you for your patience while I work through some technical challenges."
        ]
        
        # Simple hash-based selection for consistency
        import hashlib
        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)
        selected_response = responses[hash_value % len(responses)]
        
        return selected_response
    
    def _update_success_metrics(self, response_time_ms: float):
        """Update metrics for successful request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            
            # Update rolling average response time
            if self.metrics.avg_response_time_ms == 0:
                self.metrics.avg_response_time_ms = response_time_ms
            else:
                weight = 0.1  # Weight for new measurement
                self.metrics.avg_response_time_ms = (
                    (1 - weight) * self.metrics.avg_response_time_ms + 
                    weight * response_time_ms
                )
    
    def _update_error_metrics(self, error_message: str):
        """Update metrics for failed request"""
        with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_error = error_message
            self.metrics.error_count_last_hour += 1
            
            # Update status if too many errors
            if self.metrics.error_count_last_hour >= 5:
                self.status = BackendStatus.ERROR
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        uptime_hours = (time.time() - self.initialization_time) / 3600
        
        return {
            'backend_type': self.backend_type.value,
            'status': self.status.value,
            'health_score': self.metrics.health_score,
            'success_rate': self.metrics.success_rate,
            'avg_response_time_ms': self.metrics.avg_response_time_ms,
            'total_requests': self.metrics.total_requests,
            'uptime_hours': uptime_hours,
            'last_used_seconds_ago': time.time() - self.last_used,
            'error_count_last_hour': self.metrics.error_count_last_hour
        }


class PremiumBackendManager:
    """
    Premium backend manager with intelligent routing and load balancing.
    Manages multiple LLM backends simultaneously with automatic failover.
    """
    
    def __init__(self, hardware_config: Optional[HardwareConfiguration] = None, selected_model: str = 'auto'):
        self.logger = logging.getLogger(__name__)
        self.hardware_config = hardware_config
        self.selected_model = selected_model
        self.backends: Dict[BackendType, BackendInstance] = {}
        self.request_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.active = False
        self._monitoring_thread = None
        self._stats_lock = threading.Lock()
        
    async def initialize_backends(self) -> Dict[BackendType, bool]:
        """Initialize all available backends based on hardware configuration and selected model"""
        self.logger.info(f"🚀 Initializing premium backend system (model: {self.selected_model})...")
        
        initialization_results = {}
        
        # Initialize based on selected model
        if self.selected_model == 'auto':
            # Initialize primary GPT-OSS-20B backend
            if self._should_load_gpt_oss():
                initialization_results[BackendType.PRIMARY_GPT_OSS] = await self._initialize_gpt_oss()
            
            # Initialize secondary Mistral-7B backend
            if self._should_load_mistral():
                initialization_results[BackendType.SECONDARY_MISTRAL] = await self._initialize_mistral()
            
            # Initialize tertiary API backend
            initialization_results[BackendType.TERTIARY_API] = await self._initialize_api_backend()
            
            # Initialize emergency mT5 backend
            initialization_results[BackendType.EMERGENCY_MT5] = await self._initialize_mt5_backend()
        
        elif self.selected_model == 'gpt-oss':
            if self._should_load_gpt_oss():
                initialization_results[BackendType.PRIMARY_GPT_OSS] = await self._initialize_gpt_oss()
            else:
                self.logger.warning("⚠️ Insufficient hardware for GPT-OSS - falling back to API")
                initialization_results[BackendType.TERTIARY_API] = await self._initialize_api_backend()
        
        elif self.selected_model == 'mistral':
            if self._should_load_mistral():
                initialization_results[BackendType.SECONDARY_MISTRAL] = await self._initialize_mistral()
            else:
                self.logger.warning("⚠️ Insufficient hardware for Mistral - falling back to API")
                initialization_results[BackendType.TERTIARY_API] = await self._initialize_api_backend()
        
        elif self.selected_model == 'mt5':
            initialization_results[BackendType.EMERGENCY_MT5] = await self._initialize_mt5_backend()
        
        elif self.selected_model == 'api':
            initialization_results[BackendType.TERTIARY_API] = await self._initialize_api_backend()
        
        else:
            self.logger.error(f"❌ Unknown model selection: {self.selected_model}")
            # Fallback to API
            initialization_results[BackendType.TERTIARY_API] = await self._initialize_api_backend()
        
        # Start monitoring
        self._start_monitoring()
        
        self.active = True
        self.logger.info(f"✅ Backend initialization complete: {sum(initialization_results.values())}/{len(initialization_results)} backends ready")
        
        return initialization_results
    
    def _should_load_gpt_oss(self) -> bool:
        """Check if hardware supports GPT-OSS-20B loading (45GB+13GB premium specs)"""
        if not self.hardware_config:
            return False
        return (self.hardware_config.usable_ram_gb >= 45.0 and 
                self.hardware_config.usable_vram_gb >= 13.0)
    
    def _should_load_mistral(self) -> bool:
        """Check if hardware supports additional Mistral-7B loading"""
        if not self.hardware_config:
            return False
        return (self.hardware_config.usable_ram_gb >= 45.0 and 
                self.hardware_config.usable_vram_gb >= 13.0)
    
    async def _initialize_gpt_oss(self) -> bool:
        """Initialize GPT-OSS-20B backend"""
        try:
            self.logger.info("🔥 Initializing GPT-OSS-20B backend...")
            
            if not GPT_OSS_AVAILABLE:
                self.logger.warning("⚠️ GPT-OSS loader not available - skipping")
                return False
            
            # Create hybrid loader
            loader = HybridGPTOSSLoader(self.hardware_config.__dict__ if self.hardware_config else {})
            
            # Configure loading
            config = LoadingConfiguration(
                model_name="openai/gpt-oss-20b",
                use_hybrid_loading=True,
                quantization_type="MXFP4",
                gpu_memory_limit_gb=self.hardware_config.usable_vram_gb * 0.6 if self.hardware_config else 8.0,
                cpu_memory_limit_gb=self.hardware_config.usable_ram_gb * 0.7 if self.hardware_config else 20.0
            )
            
            # Load model with timeout
            result = loader.load_gpt_oss_hybrid(config)
            
            if result.success:
                backend = BackendInstance(BackendType.PRIMARY_GPT_OSS, loader)
                backend.status = BackendStatus.READY
                self.backends[BackendType.PRIMARY_GPT_OSS] = backend
                
                self.logger.info(f"✅ GPT-OSS-20B backend ready ({result.loading_time_seconds:.1f}s)")
                return True
            else:
                self.logger.error(f"❌ GPT-OSS-20B initialization failed: {result.error_message}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ GPT-OSS-20B backend error: {e}")
            return False
    
    async def _initialize_mistral(self) -> bool:
        """Initialize Mistral-7B backend"""
        try:
            self.logger.info("🔥 Initializing Mistral-7B backend...")
            
            if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
                self.logger.warning("⚠️ torch or transformers not available - skipping Mistral-7B")
                return False
            
            if not self._should_load_mistral():
                self.logger.warning("⚠️ Insufficient resources for Mistral-7B - skipping")
                return False
            
            # Load Mistral-7B model and tokenizer
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                cache_dir=".cache/huggingface"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Configure model loading for GPU
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                cache_dir=".cache/huggingface"
            )
            
            # Set to evaluation mode
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
            # Create backend instance
            backend = BackendInstance(BackendType.SECONDARY_MISTRAL)
            backend.model = model
            backend.tokenizer = tokenizer
            backend.status = BackendStatus.READY
            backend.model_info = {
                "model_name": model_name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "device": str(next(model.parameters()).device)
            }
            
            self.backends[BackendType.SECONDARY_MISTRAL] = backend
            
            self.logger.info(f"✅ Mistral-7B backend ready - {backend.model_info['parameters']:,} parameters on {backend.model_info['device']}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Mistral-7B backend loading failed: {e}")
            # Create fallback instance for graceful degradation
            backend = BackendInstance(BackendType.SECONDARY_MISTRAL)
            backend.status = BackendStatus.ERROR
            backend.error_message = str(e)
            self.backends[BackendType.SECONDARY_MISTRAL] = backend
            return False
    
    async def _initialize_api_backend(self) -> bool:
        """Initialize external API backend"""
        try:
            self.logger.info("🔥 Initializing external API backend...")
            
            import os
            
            # Check for available API keys
            openai_key = os.getenv('OPENAI_API_KEY')
            anthropic_key = os.getenv('ANTHROPIC_API_KEY')
            
            backend = BackendInstance(BackendType.TERTIARY_API)
            
            # Configure API backend based on available keys
            if openai_key or anthropic_key:
                backend.status = BackendStatus.READY
                backend.model_info = {
                    "api_type": "external",
                    "openai_available": bool(openai_key),
                    "anthropic_available": bool(anthropic_key),
                    "fallback_mode": "pattern_based"
                }
                self.logger.info(f"✅ API backend ready - OpenAI: {bool(openai_key)}, Anthropic: {bool(anthropic_key)}")
            else:
                backend.status = BackendStatus.READY  # Still ready with fallback responses
                backend.model_info = {
                    "api_type": "fallback_only",
                    "openai_available": False,
                    "anthropic_available": False,
                    "fallback_mode": "pattern_based"
                }
                self.logger.warning("⚠️ No API keys found - API backend using pattern-based fallback only")
            
            self.backends[BackendType.TERTIARY_API] = backend
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API backend initialization error: {e}")
            return False
    
    async def _initialize_mt5_backend(self) -> bool:
        """Initialize emergency mT5 backend (lightweight fallback)"""
        try:
            self.logger.info("🔥 Initializing emergency mT5 backend...")
            
            if not TORCH_AVAILABLE or not TRANSFORMERS_AVAILABLE:
                self.logger.warning("⚠️ torch or transformers not available - using built-in fallback only")
                # Create backend without model for built-in fallback responses
                backend = BackendInstance(BackendType.EMERGENCY_MT5)
                backend.status = BackendStatus.READY
                backend.model_info = {
                    "model_name": "built_in_fallback",
                    "parameters": 0,
                    "device": "cpu",
                    "emergency_only": True,
                    "fallback_mode": "built_in_responses"
                }
                self.backends[BackendType.EMERGENCY_MT5] = backend
                self.logger.info("✅ mT5 emergency backend ready (built-in responses only)")
                return True
            
            # Try to load lightweight mT5 model for emergency use
            try:
                model_name = "google/mt5-small"  # Lightweight emergency model
                
                # Load tokenizer (use Auto classes for better mT5 support)
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=".cache/huggingface",
                    trust_remote_code=True
                )
                
                # Load model (use AutoModelForSeq2SeqLM for mT5 compatibility)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use float16 for efficiency
                    device_map="auto",  # Let transformers decide optimal placement
                    low_cpu_mem_usage=True,
                    cache_dir=".cache/huggingface",
                    trust_remote_code=True
                )
                
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                
                backend = BackendInstance(BackendType.EMERGENCY_MT5)
                backend.model = model
                backend.tokenizer = tokenizer
                backend.status = BackendStatus.READY
                backend.model_info = {
                    "model_name": model_name,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "device": "cpu",
                    "emergency_only": True
                }
                
                self.logger.info(f"✅ mT5 emergency backend ready - {backend.model_info['parameters']:,} parameters on CPU")
                
            except Exception as model_error:
                self.logger.warning(f"⚠️ Could not load mT5 model: {model_error}")
                # Create backend without model for built-in fallback responses
                backend = BackendInstance(BackendType.EMERGENCY_MT5)
                backend.status = BackendStatus.READY
                backend.model_info = {
                    "model_name": "built_in_fallback",
                    "parameters": 0,
                    "device": "cpu",
                    "emergency_only": True,
                    "fallback_mode": "built_in_responses"
                }
                self.logger.info("✅ mT5 emergency backend ready (built-in responses only)")
            
            self.backends[BackendType.EMERGENCY_MT5] = backend
            return True
            
        except Exception as e:
            self.logger.error(f"❌ mT5 emergency backend initialization failed: {e}")
            return False
    
    def process_query(self, query_text: str, consciousness_state: Optional[Dict] = None, **kwargs) -> BackendResponse:
        """
        Process query with intelligent backend selection and automatic failover.
        
        Args:
            query_text: Input text to process
            consciousness_state: Optional SC_t state for context
            **kwargs: Additional parameters for query processing
            
        Returns:
            BackendResponse with generated text and metadata
        """
        if not self.active or not self.backends:
            return BackendResponse(
                text="Backend system not initialized",
                backend_used=BackendType.EMERGENCY_MT5,
                response_time_ms=0,
                success=False,
                error_message="No backends available"
            )
        
        # Create query context
        context = QueryContext(
            text=query_text,
            consciousness_state=consciousness_state,
            priority=kwargs.get('priority', 1),
            timeout_seconds=kwargs.get('timeout', 30)
        )
        
        # Select optimal backend
        selected_backend = self._select_optimal_backend(context)
        
        if not selected_backend:
            return BackendResponse(
                text="No suitable backend available",
                backend_used=BackendType.EMERGENCY_MT5,
                response_time_ms=0,
                success=False,
                error_message="All backends unavailable"
            )
        
        # Process with selected backend
        response = selected_backend.process_query(context)
        
        # If primary backend failed, try fallback
        if not response.success and selected_backend.backend_type == BackendType.PRIMARY_GPT_OSS:
            self.logger.warning("Primary backend failed, trying fallback...")
            fallback_backend = self._get_fallback_backend()
            if fallback_backend:
                response = fallback_backend.process_query(context)
        
        self.logger.info(f"Query processed with {response.backend_used.value} in {response.response_time_ms:.0f}ms")
        return response
    
    def _select_optimal_backend(self, context: QueryContext) -> Optional[BackendInstance]:
        """Select optimal backend based on query context and backend health"""
        # Priority order based on query complexity and backend availability
        priority_order = [
            BackendType.PRIMARY_GPT_OSS,
            BackendType.SECONDARY_MISTRAL,
            BackendType.TERTIARY_API,
            BackendType.EMERGENCY_MT5
        ]
        
        # Adjust priority based on query complexity
        if context.complexity_score < 0.3:
            # Simple queries can use lighter models
            priority_order = [
                BackendType.SECONDARY_MISTRAL,
                BackendType.PRIMARY_GPT_OSS,
                BackendType.TERTIARY_API,
                BackendType.EMERGENCY_MT5
            ]
        elif context.complexity_score > 0.7 or context.consciousness_state:
            # Complex/consciousness queries need primary model
            priority_order = [
                BackendType.PRIMARY_GPT_OSS,
                BackendType.SECONDARY_MISTRAL,
                BackendType.TERTIARY_API,
                BackendType.EMERGENCY_MT5
            ]
        
        # Find best available backend
        for backend_type in priority_order:
            backend = self.backends.get(backend_type)
            if (backend and 
                backend.status == BackendStatus.READY and 
                backend.metrics.health_score > 50.0):
                return backend
        
        # Fallback to any working backend
        for backend in self.backends.values():
            if backend.status == BackendStatus.READY:
                return backend
                
        return None
    
    def _get_fallback_backend(self) -> Optional[BackendInstance]:
        """Get fallback backend when primary fails"""
        fallback_order = [
            BackendType.SECONDARY_MISTRAL,
            BackendType.TERTIARY_API,
            BackendType.EMERGENCY_MT5
        ]
        
        for backend_type in fallback_order:
            backend = self.backends.get(backend_type)
            if backend and backend.status == BackendStatus.READY:
                return backend
                
        return None
    
    def _start_monitoring(self):
        """Start background monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
            
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info("📊 Backend monitoring started")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.active:
            try:
                # Monitor backend health
                for backend_type, backend in self.backends.items():
                    health = backend.get_health_status()
                    
                    # Log warnings for unhealthy backends
                    if health['health_score'] < 70:
                        self.logger.warning(f"Backend {backend_type.value} health: {health['health_score']:.1f}%")
                    
                    # Reset error counters hourly
                    if time.time() % 3600 < 30:  # Every hour
                        backend.metrics.error_count_last_hour = 0
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        backend_statuses = {}
        total_requests = 0
        total_successful = 0
        
        for backend_type, backend in self.backends.items():
            health = backend.get_health_status()
            backend_statuses[backend_type.value] = health
            total_requests += backend.metrics.total_requests
            total_successful += backend.metrics.successful_requests
        
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 100.0
        
        return {
            'system_active': self.active,
            'total_backends': len(self.backends),
            'healthy_backends': len([b for b in self.backends.values() if b.status == BackendStatus.READY]),
            'overall_success_rate': overall_success_rate,
            'total_requests_processed': total_requests,
            'backend_details': backend_statuses,
            'hardware_config': self.hardware_config.__dict__ if self.hardware_config else {}
        }
    
    def shutdown(self):
        """Gracefully shutdown all backends"""
        self.logger.info("🔄 Shutting down backend system...")
        
        self.active = False
        
        # Wait for monitoring thread
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Cleanup backends
        for backend in self.backends.values():
            if hasattr(backend.loader, '_cleanup_failed_loading'):
                backend.loader._cleanup_failed_loading()
        
        self.backends.clear()
        self.logger.info("✅ Backend system shutdown complete")