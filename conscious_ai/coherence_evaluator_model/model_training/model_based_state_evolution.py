"""
Motor de evolución de estados usando modelos entrenados
Genera SCt+1 usando el modelo LoRA entrenado en fases anteriores
"""
# Contenido COMPLETO y CORREGIDO para model_based_state_evolution.py

import json
import logging
import torch
import numpy as np
import os
from typing import Dict, Any, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Shared utilities for Mistral models
from conscious_ai.shared_utilities.mistral_inference_utils import (
    load_mistral_model,
    format_mistral_prompt,
    extract_json_from_mistral_response,
    validate_mistral_response
)

from conscious_ai.Train.language_detector import LanguageDetector
from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import ModelBasedCoherenceEvaluator

logger = logging.getLogger(__name__)


class ModelBasedStateEvolution:
    """
    Genera evolución de estados usando modelos entrenados (El "Trabajador")
    """
    
    def __init__(
        self,
        model_checkpoint: str = "./models/autonomous_lora",
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.1",
        device: str = None
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.language_detector = LanguageDetector()
        self._load_model(model_checkpoint, base_model)
        self.coherence_evaluator = ModelBasedCoherenceEvaluator(use_ml_classifier=False)
        self.generation_cache = {}
        self.max_cache_size = 100
    
    def _load_model(self, checkpoint_path: str, base_model_name: str):
        logger.info(f"🔧 Loading model from {checkpoint_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        # Check if LoRA checkpoint exists
        if os.path.exists(checkpoint_path):
            try:
                self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
                self.model.to(self.device)
                self.model.eval()
                self.is_lora_loaded = True
                logger.info("✅ LoRA model loaded successfully from ./models/autonomous_lora")
            except Exception as e:
                logger.error(f"❌ Error loading LoRA model: {e}")
                logger.warning("🔄 Using base model without fine-tuning as fallback")
                self.model = base_model.to(self.device)
                self.model.eval()
                self.is_lora_loaded = False
        else:
            logger.warning(f"⚠️ LoRA checkpoint not found at {checkpoint_path}")
            logger.info("🔄 Using base model without fine-tuning as fallback")
            self.model = base_model.to(self.device)
            self.model.eval()
            self.is_lora_loaded = False
    
    def evolve_state(
        self,
        current_state: Dict[str, Any],
        external_input: Optional[str] = None,
        temperature: float = 0.4,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        components = self._extract_state_components(current_state)
        lang = self._detect_language(components)
        prompt = self._prepare_prompt(components, external_input, lang)
        
        cache_key = f"{hash(prompt)}_{temperature}"
        if cache_key in self.generation_cache:
            return self.generation_cache[cache_key]
        
        for attempt in range(max_attempts):
            try:
                new_state = self._generate_with_model(prompt, temperature)
                # Handle None return (JSON parsing failed)
                if new_state is None:
                    logger.warning(f"⚠️ JSON parsing failed on attempt {attempt + 1}, using fallback")
                    return self._fallback_generation(components, lang)
                
                if self._validate_state_format(new_state):
                    coherence_analysis = self.coherence_evaluator.evaluate_transition(current_state, new_state)
                    if coherence_analysis.verdict.value == 'coherente' or attempt == max_attempts - 1:
                        self._update_cache(cache_key, new_state)
                        return new_state
                    temperature *= 0.9
            except Exception as e:
                logger.warning(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    logger.info("🔄 All model attempts failed, using fallback generation")
                    return self._fallback_generation(components, lang)
        
        return self._fallback_generation(components, lang)

    def _extract_state_components(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if 'G_t' in state:
            return {'goal': state.get('G_t', {}).get('primary_goal', 'understand'), 'emotion': state.get('S_t', {}).get('emotional_state', 'neutral'), 'confidence': state.get('S_t', {}).get('confidence_level', 0.5), 'thought': state.get('A_t', [''])[0] if state.get('A_t') else '', 'memory': state.get('M_t', [])}
        else:
            return {'goal': state.get('goal', 'understand'), 'emotion': state.get('emotion', 'neutral'), 'confidence': state.get('confidence', 0.5), 'thought': state.get('thought', ''), 'memory': state.get('memory', [])}
    
    def _detect_language(self, components: Dict[str, Any]) -> str:
        thought = components.get('thought', '')
        if thought:
            lang, _ = self.language_detector.detect_language(thought)
            return lang
        memories = components.get('memory', [])
        if memories and isinstance(memories[0], str):
            lang, _ = self.language_detector.detect_language(memories[0])
            return lang
        return 'en'
    
    def _prepare_prompt(self, components: Dict[str, Any], external_input: Optional[str], lang: str) -> str:
        previous_state_dict = {"goal": components['goal'], "emotion": components['emotion'], "confidence": components['confidence'], "thought": components['thought'], "memory": components['memory'][-3:]}
        previous_state_json = json.dumps(previous_state_dict, ensure_ascii=False)
        example_output = {"goal": "example_goal", "emotion": "example_emotion", "confidence": 0.5, "thought": "example thought", "memory": ["example memory"]}
        example_json = json.dumps(example_output, ensure_ascii=False)
        
        if lang == 'es':
            content = f"""Genera el siguiente estado en formato JSON exacto.

Estado anterior:
{previous_state_json}

Genera JSON válido con estas claves: goal, emotion, confidence, thought, memory.
Ejemplo de formato: {example_json}

Siguiente estado JSON:"""
        else:
            content = f"""Generate the next state in exact JSON format.

Previous state:
{previous_state_json}

Generate valid JSON with keys: goal, emotion, confidence, thought, memory.
Example format: {example_json}

Next state JSON:"""
        
        return format_mistral_prompt("You are an AI consciousness state evolution system. Generate the next conscious state based on the previous state.", content)
    
    def _generate_with_model(self, prompt: str, temperature: float) -> Dict[str, Any]:
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True, padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode only the new tokens (exclude input)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        logger.debug(f"Texto generado (limpio): {generated_text}")
        
        # Use shared utility for JSON extraction
        extracted_json = extract_json_from_mistral_response(generated_text)
        
        if extracted_json:
            try:
                new_state = json.loads(extracted_json)
                if validate_mistral_response(new_state, required_keys=["goal", "emotion", "confidence", "thought", "memory"]):
                    return self._normalize_state(new_state)
                else:
                    logger.warning("Generated state missing required keys")
            except json.JSONDecodeError as e:
                logger.warning(f"❌ JSON parsing failed: {e}")
        
        logger.info("🔄 Using fallback generation due to invalid JSON")
        return None

    def _validate_state_format(self, state: Dict[str, Any]) -> bool:
        required_fields = ['goal', 'emotion', 'confidence', 'thought', 'memory']
        for field in required_fields:
            if field not in state: return False
        if not isinstance(state['confidence'], (int, float)): return False
        if not isinstance(state['memory'], list): return False
        if not 0 <= state['confidence'] <= 1: return False
        return True
    
    def _normalize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {'goal': state.get('goal', 'understand'), 'emotion': state.get('emotion', 'neutral'), 'confidence': float(state.get('confidence', 0.5)), 'thought': state.get('thought', ''), 'memory': state.get('memory', [])}
        normalized['confidence'] = max(0.0, min(1.0, normalized['confidence']))
        if not isinstance(normalized['memory'], list):
            normalized['memory'] = [str(normalized['memory'])]
        lang, _ = self.language_detector.detect_language(normalized['thought'])
        normalized['language'] = lang
        return normalized

    def _fallback_generation(self, components: Dict[str, Any], lang: str) -> Dict[str, Any]:
        new_state = components.copy()
        new_state['confidence'] = round(max(0.1, min(0.9, components['confidence'] + np.random.uniform(-0.1, 0.1))), 2)
        if lang == 'es':
            connectors = ["Reflexionando más,", "Esto me lleva a pensar que", "Además,"]
            new_thought = f"{np.random.choice(connectors)} continúo explorando estas ideas."
        else:
            connectors = ["Reflecting further,", "This leads me to think that", "Moreover,"]
            new_thought = f"{np.random.choice(connectors)} I continue exploring these ideas."
        new_state['thought'] = new_thought
        new_state['memory'] = components['memory'][-3:] if len(components['memory']) > 3 else components['memory']
        new_state['language'] = lang
        return new_state
    
    def _update_cache(self, key: str, state: Dict[str, Any]):
        self.generation_cache[key] = state
        if len(self.generation_cache) > self.max_cache_size:
            keys_to_remove = list(self.generation_cache.keys())[:10]
            for k in keys_to_remove:
                del self.generation_cache[k]
    
    def batch_evolve_states(
        self,
        states: List[Dict[str, Any]],
        temperature: float = 0.8
    ) -> List[Dict[str, Any]]:
        evolved_states = []
        batch_size = 4
        for i in range(0, len(states), batch_size):
            batch = states[i:i + batch_size]
            for state in batch:
                new_state = self.evolve_state(state, temperature=temperature)
                evolved_states.append(new_state)
        return evolved_states


# --- CLASE GERENTE (VERSIÓN CORREGIDA Y LIMPIA) ---
class HybridStateEvolution:
    """
    Combina evolución basada en modelos con heurísticas para mayor robustez (El "Gerente")
    """
    
    def __init__(
        self,
        model_checkpoint: Optional[str] = "./models/autonomous_lora",
        use_model_primary: bool = True
    ):
        self.use_model_primary = use_model_primary
        if model_checkpoint and os.path.exists(model_checkpoint):
            # El gerente crea una instancia del trabajador
            self.model_evolution = ModelBasedStateEvolution(model_checkpoint)
            self.has_model = True
        else:
            logger.warning(f"No se encontró modelo en {model_checkpoint}, se usará solo heurísticas.")
            self.has_model = False
            self.use_model_primary = False
        
        # El gerente también conoce el Plan B (el motor heurístico)
        from conscious_ai.coherence_evaluator_model.heuristic_training.state_evolution_engine import StateEvolutionEngine
        self.heuristic_evolution = StateEvolutionEngine()
    
    def evolve_state(
        self,
        current_state: Dict[str, Any],
        external_input: Optional[str] = None,
        evolution_mode: str = 'natural'
    ) -> Dict[str, Any]:
        """
        Decide qué motor de evolución usar y delega el trabajo.
        Esta es la única función del gerente: decidir y delegar.
        """
        
        if self.use_model_primary and self.has_model:
            try:
                logger.debug("Gerente: Delegando a ModelBasedStateEvolution (Trabajador)...")
                # El gerente le pasa el trabajo al trabajador
                return self.model_evolution.evolve_state(
                    current_state,
                    external_input
                )
            except Exception as e:
                logger.error(f"FALLO CRÍTICO en el trabajador. Activando Plan B. Error: {e}", exc_info=True)
        
        # Si el modelo no está activado o falla, el gerente usa el Plan B
        logger.debug("Gerente: Usando motor heurístico de respaldo (Plan B)...")
        return self.heuristic_evolution.evolve_state(
            current_state,
            external_input,
            evolution_mode
        )