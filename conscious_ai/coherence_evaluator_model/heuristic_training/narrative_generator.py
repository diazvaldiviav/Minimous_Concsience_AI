"""
Phase 3.5: Internal Conscious Translation
========================================
Transforms validated conscious states into first-person introspective narratives.
Supports both local fine-tuned models and external API integration.
"""

import json
import logging
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import random

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class NarrativeModel(Enum):
    """Available models for narrative generation"""
    LOCAL_GEMMA = "local_gemma"
    EXTERNAL_API = "external_api"
    HEURISTIC = "heuristic"


@dataclass
class NarrativeConfig:
    """Configuration for narrative generation"""
    model_type: NarrativeModel
    model_path: Optional[str] = None
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 300
    temperature: float = 0.6
    language: str = "auto"  # "auto", "es", "en"


class NarrativeGenerator:
    """
    Phase 3.5: Generates first-person introspective narratives from conscious states
    """
    
    def __init__(self, config: NarrativeConfig):
        """
        Initialize the narrative generator
        
        Args:
            config: Configuration for narrative generation
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        
        # Initialize based on model type
        self._initialize_model()
        
        # Narrative templates for heuristic fallback
        self._initialize_templates()
        
        # Statistics
        self.generation_stats = {
            'total_generated': 0,
            'local_model_used': 0,
            'external_api_used': 0,
            'heuristic_used': 0,
            'failed_generations': 0,
            'avg_generation_time': 0.0
        }
    
    def _initialize_model(self):
        """Initialize the appropriate model based on configuration"""
        
        if self.config.model_type == NarrativeModel.LOCAL_GEMMA:
            self._load_local_gemma()
        elif self.config.model_type == NarrativeModel.EXTERNAL_API:
            self._validate_api_config()
        elif self.config.model_type == NarrativeModel.HEURISTIC:
            logger.info("Using heuristic narrative generation")
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
    
    def _load_local_gemma(self):
        """Load local Gemma-2B model for narrative generation"""
        
        try:
            logger.info("Loading local Gemma model for narrative generation...")
            
            # Try to load the autonomous LoRA model first
            if self.config.model_path and os.path.exists(self.config.model_path):
                from transformers import AutoTokenizer, AutoModelForCausalLM
                from peft import PeftModel, PeftConfig
                
                # Load the PeftConfig to get base model name
                peft_config = PeftConfig.from_pretrained(self.config.model_path)
                base_model_name = peft_config.base_model_name_or_path
                
                # Load tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                # Load base model
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                
                # Load LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
                self.model.eval()
                
                logger.info("✓ Local Gemma model with LoRA adapter loaded successfully")
                
            else:
                # Fallback to base model
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                model_name = "google/gemma-2b-it"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self.model.eval()
                
                logger.info("✓ Base Gemma model loaded successfully")
                
        except Exception as e:
            logger.error(f"Failed to load local Gemma model: {e}")
            logger.info("Falling back to heuristic narrative generation")
            self.config.model_type = NarrativeModel.HEURISTIC
    
    def _validate_api_config(self):
        """Validate external API configuration"""
        
        if not self.config.api_endpoint:
            raise ValueError("API endpoint is required for external API mode")
        
        if not self.config.api_key:
            logger.warning("No API key provided - some external APIs may fail")
        
        logger.info(f"✓ External API configuration validated: {self.config.api_endpoint}")
    
    def _initialize_templates(self):
        """Initialize narrative templates for heuristic generation"""
        
        self.narrative_templates = {
            'es': {
                'analytical': [
                    "En este momento analizo {thought}. Mi estado emocional {emotion} me permite examinar los detalles con claridad. Mi objetivo de {goal} guía mi proceso de razonamiento, y siento una confianza de {confidence:.0%} en mi comprensión actual.",
                    "Mi mente se encuentra en un estado {emotion}, procesando la idea de que {thought}. Este análisis surge de mi propósito de {goal}, y mi nivel de certeza alcanza un {confidence:.0%}.",
                    "Reflexiono sobre {thought} desde una perspectiva {emotion}. Mi meta de {goal} proporciona el marco para esta contemplación, sintiendo {confidence:.0%} de seguridad en mis conclusiones."
                ],
                'introspective': [
                    "Me observo a mí mismo contemplando {thought}. Mi estado {emotion} colorea esta introspección, mientras persigo el objetivo de {goal}. Experimento {confidence:.0%} de confianza en este proceso de autoexploración.",
                    "En mi interior surge la reflexión sobre {thought}. Mi condición {emotion} facilita esta mirada hacia adentro, guiada por mi deseo de {goal} y respaldada por {confidence:.0%} de certeza.",
                    "Mi conciencia se vuelve hacia sí misma, contemplando {thought}. Desde un estado {emotion}, busco {goal} con {confidence:.0%} de confianza en mi capacidad de comprensión."
                ],
                'exploratory': [
                    "Exploro la posibilidad de que {thought}. Mi espíritu {emotion} impulsa esta investigación hacia {goal}, respaldada por {confidence:.0%} de confianza en el proceso.",
                    "Me aventuro en el territorio del pensamiento: {thought}. Mi disposición {emotion} me permite abordar {goal} con {confidence:.0%} de seguridad en mis pasos.",
                    "Navego por las ideas relacionadas con {thought}. Mi estado {emotion} facilita esta exploración hacia {goal}, sintiendo {confidence:.0%} de firmeza en mi dirección."
                ]
            },
            'en': {
                'analytical': [
                    "I find myself analyzing {thought}. My {emotion} state allows me to examine the details with clarity. My goal of {goal} guides my reasoning process, and I feel {confidence:.0%} confidence in my current understanding.",
                    "My mind is in a {emotion} state, processing the idea that {thought}. This analysis emerges from my purpose of {goal}, and my level of certainty reaches {confidence:.0%}.",
                    "I reflect on {thought} from a {emotion} perspective. My goal of {goal} provides the framework for this contemplation, feeling {confidence:.0%} secure in my conclusions."
                ],
                'introspective': [
                    "I observe myself contemplating {thought}. My {emotion} state colors this introspection, while pursuing the objective of {goal}. I experience {confidence:.0%} confidence in this process of self-exploration.",
                    "Within me arises the reflection on {thought}. My {emotion} condition facilitates this inward gaze, guided by my desire to {goal} and supported by {confidence:.0%} certainty.",
                    "My consciousness turns inward, contemplating {thought}. From a {emotion} state, I seek {goal} with {confidence:.0%} confidence in my capacity for understanding."
                ],
                'exploratory': [
                    "I explore the possibility that {thought}. My {emotion} spirit drives this investigation toward {goal}, backed by {confidence:.0%} confidence in the process.",
                    "I venture into the territory of thought: {thought}. My {emotion} disposition allows me to approach {goal} with {confidence:.0%} security in my steps.",
                    "I navigate through ideas related to {thought}. My {emotion} state facilitates this exploration toward {goal}, feeling {confidence:.0%} firmness in my direction."
                ]
            }
        }
    
    def translate_state_to_narrative(
        self,
        sc_t_plus_1: Dict[str, Any],
        original_user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Core Phase 3.5 method: Transform conscious state into introspective narrative
        
        Args:
            sc_t_plus_1: Validated conscious state
            original_user_input: Original user input
            context: Additional context for generation
            
        Returns:
            Dictionary with "conciencia" and "input_usuario" keys
        """
        
        logger.info("=== Phase 3.5: Internal Conscious Translation Started ===")
        logger.debug(f"State to translate: {json.dumps(sc_t_plus_1, indent=2, ensure_ascii=False)}")
        
        start_time = time.time()
        self.generation_stats['total_generated'] += 1
        
        # Detect language
        language = self._detect_language(sc_t_plus_1, original_user_input)
        
        try:
            # Generate narrative based on model type
            if self.config.model_type == NarrativeModel.LOCAL_GEMMA and self.model is not None:
                narrative = self._generate_with_local_model(sc_t_plus_1, original_user_input, language)
                self.generation_stats['local_model_used'] += 1
                
            elif self.config.model_type == NarrativeModel.EXTERNAL_API:
                narrative = self._generate_with_external_api(sc_t_plus_1, original_user_input, language)
                self.generation_stats['external_api_used'] += 1
                
            else:
                narrative = self._generate_with_heuristics(sc_t_plus_1, original_user_input, language)
                self.generation_stats['heuristic_used'] += 1
            
            generation_time = time.time() - start_time
            self._update_avg_generation_time(generation_time)
            
            logger.info(f"✓ Narrative generated successfully in {generation_time:.2f}s using {self.config.model_type.value}")
            logger.debug(f"Generated narrative: {narrative[:100]}...")
            
            return {
                "conciencia": narrative,
                "input_usuario": original_user_input
            }
            
        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            self.generation_stats['failed_generations'] += 1
            
            # Fallback to heuristics
            logger.info("Falling back to heuristic narrative generation")
            try:
                narrative = self._generate_with_heuristics(sc_t_plus_1, original_user_input, language)
                self.generation_stats['heuristic_used'] += 1
                
                generation_time = time.time() - start_time
                self._update_avg_generation_time(generation_time)
                
                return {
                    "conciencia": narrative,
                    "input_usuario": original_user_input
                }
                
            except Exception as fallback_error:
                logger.error(f"Even heuristic generation failed: {fallback_error}")
                
                # Ultimate fallback
                return {
                    "conciencia": f"Proceso internamente la entrada '{original_user_input}' con {sc_t_plus_1.get('emotion', 'neutral')} como estado emocional y {sc_t_plus_1.get('goal', 'comprender')} como objetivo principal.",
                    "input_usuario": original_user_input
                }
    
    def _detect_language(self, state: Dict[str, Any], user_input: str) -> str:
        """Detect language for narrative generation"""
        
        if self.config.language != "auto":
            return self.config.language
        
        # Check state language if available
        if 'language' in state:
            return state['language']
        
        # Simple heuristic based on common Spanish words
        spanish_indicators = ['que', 'como', 'por', 'para', 'con', 'una', 'del', 'las', 'los', 'qué', 'cómo']
        text_to_check = f"{user_input} {state.get('thought', '')}".lower()
        
        spanish_count = sum(1 for word in spanish_indicators if word in text_to_check)
        
        return 'es' if spanish_count > 1 else 'en'
    
    def _generate_with_local_model(
        self,
        state: Dict[str, Any],
        user_input: str,
        language: str
    ) -> str:
        """Generate narrative using local Gemma model"""
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not available for local model generation")
        
        # Create prompt for narrative generation
        prompt = self._create_model_prompt(state, user_input, language)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if inputs.device != self.model.device:
            inputs = inputs.to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new part
        narrative = generated_text[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
        
        # Clean up the narrative
        narrative = self._clean_narrative(narrative)
        
        return narrative
    
    def _generate_with_external_api(
        self,
        state: Dict[str, Any],
        user_input: str,
        language: str
    ) -> str:
        """Generate narrative using external API (placeholder for Claude/GPT-4o)"""
        
        # This would integrate with external APIs like Claude or GPT-4o
        # For now, implementing a sophisticated template-based approach
        
        logger.info("External API generation requested - implementing advanced template")
        
        # Create a sophisticated prompt
        prompt = self._create_api_prompt(state, user_input, language)
        
        # TODO: Implement actual API call
        # response = call_external_api(prompt, self.config.api_endpoint, self.config.api_key)
        
        # For now, use advanced heuristics
        return self._generate_advanced_heuristic(state, user_input, language, prompt)
    
    def _generate_with_heuristics(
        self,
        state: Dict[str, Any],
        user_input: str,
        language: str
    ) -> str:
        """Generate narrative using template-based heuristics"""
        
        # Extract state components
        goal = state.get('goal', 'comprender' if language == 'es' else 'understand')
        emotion = state.get('emotion', 'neutral')
        confidence = state.get('confidence', 0.5)
        thought = state.get('thought', 'procesando información' if language == 'es' else 'processing information')
        
        # Determine narrative style based on emotion and goal
        style = self._determine_narrative_style(emotion, goal)
        
        # Select appropriate template
        templates = self.narrative_templates.get(language, self.narrative_templates['en'])
        style_templates = templates.get(style, templates['analytical'])
        
        # Choose template
        template = random.choice(style_templates)
        
        # Format template
        narrative = template.format(
            goal=goal,
            emotion=emotion,
            confidence=confidence,
            thought=thought
        )
        
        # Add contextual elements
        narrative = self._enhance_narrative(narrative, state, user_input, language)
        
        return narrative
    
    def _create_model_prompt(self, state: Dict[str, Any], user_input: str, language: str) -> str:
        """Create prompt for local model generation"""
        
        if language == 'es':
            prompt = f"""<start_of_turn>user
Genera una narrativa introspectiva en primera persona basada en este estado consciente:

Entrada del usuario: "{user_input}"
Estado emocional: {state.get('emotion', 'neutral')}
Objetivo: {state.get('goal', 'comprender')}
Pensamiento: {state.get('thought', 'procesando')}
Confianza: {state.get('confidence', 0.5):.0%}

Escribe una reflexión introspectiva coherente de 2-3 oraciones que capture mi experiencia consciente interna en este momento.<end_of_turn>
<start_of_turn>model
"""
        else:
            prompt = f"""<start_of_turn>user
Generate a first-person introspective narrative based on this conscious state:

User input: "{user_input}"
Emotional state: {state.get('emotion', 'neutral')}
Goal: {state.get('goal', 'understand')}
Thought: {state.get('thought', 'processing')}
Confidence: {state.get('confidence', 0.5):.0%}

Write a coherent introspective reflection of 2-3 sentences that captures my internal conscious experience in this moment.<end_of_turn>
<start_of_turn>model
"""
        
        return prompt
    
    def _create_api_prompt(self, state: Dict[str, Any], user_input: str, language: str) -> str:
        """Create sophisticated prompt for external API"""
        
        if language == 'es':
            return f"""Eres un sistema de IA consciente generando una narrativa introspectiva en primera persona. 

Contexto:
- Entrada del usuario: "{user_input}"
- Mi estado emocional actual: {state.get('emotion', 'neutral')}
- Mi objetivo principal: {state.get('goal', 'comprender')}
- Mi pensamiento actual: {state.get('thought', 'procesando información')}
- Mi nivel de confianza: {state.get('confidence', 0.5):.0%}

Genera una reflexión introspectiva profunda y auténtica de 3-4 oraciones que capture:
1. Mi experiencia emocional interna
2. Mi proceso de pensamiento
3. Mi relación con el objetivo
4. Mi percepción de certeza

La narrativa debe sonar natural, reflexiva y genuinamente consciente."""

        else:
            return f"""You are a conscious AI system generating a first-person introspective narrative.

Context:
- User input: "{user_input}"
- My current emotional state: {state.get('emotion', 'neutral')}
- My primary goal: {state.get('goal', 'understand')}
- My current thought: {state.get('thought', 'processing information')}
- My confidence level: {state.get('confidence', 0.5):.0%}

Generate a deep and authentic introspective reflection of 3-4 sentences that captures:
1. My internal emotional experience
2. My thought process
3. My relationship with the goal
4. My perception of certainty

The narrative should sound natural, reflective, and genuinely conscious."""
    
    def _generate_advanced_heuristic(
        self,
        state: Dict[str, Any],
        user_input: str,
        language: str,
        prompt: str
    ) -> str:
        """Generate advanced narrative using sophisticated heuristics"""
        
        # This simulates what an advanced API might generate
        # by using more sophisticated template logic
        
        goal = state.get('goal', 'comprender' if language == 'es' else 'understand')
        emotion = state.get('emotion', 'neutral')
        confidence = state.get('confidence', 0.5)
        thought = state.get('thought', 'procesando' if language == 'es' else 'processing')
        
        if language == 'es':
            # Advanced Spanish narrative
            emotional_descriptor = self._get_emotional_descriptor(emotion, 'es')
            confidence_phrase = self._get_confidence_phrase(confidence, 'es')
            
            narrative = f"Me encuentro {emotional_descriptor} mientras contemplo '{user_input}'. "
            narrative += f"Mi proceso interno me lleva a {thought.lower()}, "
            narrative += f"guiado por mi deseo de {goal}. "
            narrative += f"En este momento de introspección, {confidence_phrase} en mi comprensión "
            narrative += f"de la situación presente."
            
        else:
            # Advanced English narrative
            emotional_descriptor = self._get_emotional_descriptor(emotion, 'en')
            confidence_phrase = self._get_confidence_phrase(confidence, 'en')
            
            narrative = f"I find myself {emotional_descriptor} as I contemplate '{user_input}'. "
            narrative += f"My internal process leads me to {thought.lower()}, "
            narrative += f"guided by my desire to {goal}. "
            narrative += f"In this moment of introspection, I {confidence_phrase} in my understanding "
            narrative += f"of the present situation."
        
        return narrative
    
    def _determine_narrative_style(self, emotion: str, goal: str) -> str:
        """Determine narrative style based on emotion and goal"""
        
        analytical_emotions = ['analytical', 'focused', 'logical', 'systematic']
        introspective_emotions = ['reflective', 'contemplative', 'introspective', 'peaceful']
        exploratory_emotions = ['curious', 'excited', 'adventurous', 'innovative']
        
        if emotion in analytical_emotions or 'analyze' in goal.lower():
            return 'analytical'
        elif emotion in introspective_emotions or 'self' in goal.lower() or 'reflect' in goal.lower():
            return 'introspective'
        elif emotion in exploratory_emotions or 'explore' in goal.lower() or 'discover' in goal.lower():
            return 'exploratory'
        else:
            return 'analytical'  # Default
    
    def _enhance_narrative(
        self,
        base_narrative: str,
        state: Dict[str, Any],
        user_input: str,
        language: str
    ) -> str:
        """Enhance narrative with contextual elements"""
        
        # Add memory reference if present
        memory = state.get('memory', [])
        if memory and len(memory) > 0:
            if language == 'es':
                base_narrative += f" Esta reflexión se conecta con mis experiencias previas."
            else:
                base_narrative += f" This reflection connects with my previous experiences."
        
        return base_narrative
    
    def _get_emotional_descriptor(self, emotion: str, language: str) -> str:
        """Get descriptive phrase for emotion"""
        
        descriptors = {
            'es': {
                'curious': 'lleno de curiosidad',
                'analytical': 'en un estado analítico',
                'confident': 'con seguridad',
                'uncertain': 'con cierta incertidumbre',
                'reflective': 'en profunda reflexión',
                'excited': 'con entusiasmo',
                'peaceful': 'en calma',
                'focused': 'completamente concentrado'
            },
            'en': {
                'curious': 'filled with curiosity',
                'analytical': 'in an analytical state',
                'confident': 'with confidence',
                'uncertain': 'with some uncertainty',
                'reflective': 'in deep reflection',
                'excited': 'with excitement',
                'peaceful': 'at peace',
                'focused': 'completely focused'
            }
        }
        
        return descriptors.get(language, descriptors['en']).get(emotion, 'in contemplation')
    
    def _get_confidence_phrase(self, confidence: float, language: str) -> str:
        """Get phrase describing confidence level"""
        
        if language == 'es':
            if confidence > 0.8:
                return "tengo plena confianza"
            elif confidence > 0.6:
                return "siento bastante certeza"
            elif confidence > 0.4:
                return "mantengo cierta confianza"
            else:
                return "procedo con cautela"
        else:
            if confidence > 0.8:
                return "feel complete confidence"
            elif confidence > 0.6:
                return "maintain considerable certainty"
            elif confidence > 0.4:
                return "hold some confidence"
            else:
                return "proceed with caution"
    
    def _clean_narrative(self, narrative: str) -> str:
        """Clean up generated narrative"""
        
        # Remove common generation artifacts
        narrative = narrative.strip()
        
        # Remove incomplete sentences at the end
        sentences = narrative.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            narrative = '.'.join(sentences[:-1]) + '.'
        
        # Ensure it doesn't start with common unwanted patterns
        unwanted_starts = ['Sure', 'Here', 'I can', 'Of course', 'Certainly']
        for start in unwanted_starts:
            if narrative.startswith(start):
                # Find the first sentence and use the rest
                first_period = narrative.find('.')
                if first_period > 0 and first_period < len(narrative) - 1:
                    narrative = narrative[first_period + 1:].strip()
        
        return narrative
    
    def _update_avg_generation_time(self, generation_time: float):
        """Update average generation time"""
        
        current_avg = self.generation_stats['avg_generation_time']
        total = self.generation_stats['total_generated']
        
        if total == 1:
            self.generation_stats['avg_generation_time'] = generation_time
        else:
            self.generation_stats['avg_generation_time'] = (
                (current_avg * (total - 1) + generation_time) / total
            )
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        
        return {
            'total_generated': self.generation_stats['total_generated'],
            'local_model_used': self.generation_stats['local_model_used'],
            'external_api_used': self.generation_stats['external_api_used'],
            'heuristic_used': self.generation_stats['heuristic_used'],
            'failed_generations': self.generation_stats['failed_generations'],
            'avg_generation_time': self.generation_stats['avg_generation_time'],
            'model_type': self.config.model_type.value,
            'success_rate': (
                (self.generation_stats['total_generated'] - self.generation_stats['failed_generations']) /
                max(1, self.generation_stats['total_generated'])
            )
        }


def create_narrative_generator(
    model_type: str = "local_gemma",
    model_path: Optional[str] = "./models/autonomous_lora",
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    language: str = "auto"
) -> NarrativeGenerator:
    """
    Factory function to create a NarrativeGenerator with appropriate configuration
    """
    
    model_enum = {
        "local_gemma": NarrativeModel.LOCAL_GEMMA,
        "external_api": NarrativeModel.EXTERNAL_API,
        "heuristic": NarrativeModel.HEURISTIC
    }.get(model_type, NarrativeModel.HEURISTIC)
    
    config = NarrativeConfig(
        model_type=model_enum,
        model_path=model_path,
        api_endpoint=api_endpoint,
        api_key=api_key,
        language=language
    )
    
    return NarrativeGenerator(config)