"""
Módulo de inferencia que integra el modelo entrenado con LoRA
en el sistema MinimalConsciousAI, reemplazando los generadores heurísticos
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import MT5ForConditionalGeneration, MT5Tokenizer

# Importar módulos del sistema base
from conscious_ai.Train.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class TrainedConsciousnessModel:
    """
    Modelo de consciencia entrenado que reemplaza GoalGenerator y AutomaticThoughtGenerator.
    Genera G_t y A_t usando el modelo mT5 entrenado con LoRA.
    """

    def __init__(
        self,
        model_checkpoint: str = "./models/trained_lora/checkpoint-best",
        base_model_name: str = "google/mt5-small",
        device: str = None,
    ):
        self.model_checkpoint = model_checkpoint
        self.base_model_name = base_model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.language_detector = LanguageDetector()

        # Cache para optimizar inferencias repetidas
        self.inference_cache = {}
        self.cache_size = 100

        # Mensajes bilingües
        self.messages = {
            "es": {
                "processing": "Procesando entrada...",
                "analyzing": "Analizando contenido semántico...",
                "reflecting": "Reflexionando sobre el contexto...",
                "integrating": "Integrando información...",
            },
            "en": {
                "processing": "Processing input...",
                "analyzing": "Analyzing semantic content...",
                "reflecting": "Reflecting on context...",
                "integrating": "Integrating information...",
            },
        }

    def load_model(self):
        """Carga el modelo entrenado con LoRA"""
        logger.info(f"Cargando modelo desde {self.model_checkpoint}")

        # Cargar tokenizer
        self.tokenizer = MT5Tokenizer.from_pretrained(self.base_model_name)

        # Cargar modelo base
        base_model = MT5ForConditionalGeneration.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )

        # Cargar adaptadores LoRA
        try:
            self.model = PeftModel.from_pretrained(base_model, self.model_checkpoint)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Modelo cargado exitosamente")
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            logger.warning("Usando modelo base sin LoRA")
            self.model = base_model.to(self.device)
            self.model.eval()

    def generate_sct_components(
        self,
        sensory_data: Dict[str, Any],
        self_state: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        max_length: int = 128,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Genera componentes G_t y A_t usando el modelo entrenado.
        
        Returns:
            Tuple de (G_t, A_t)
        """
        if self.model is None:
            self.load_model()

        # Extraer texto de entrada
        input_text = sensory_data.get("text", "")

        # Verificar cache
        cache_key = f"{input_text}_{self_state.get('confidence_level', 0.5):.2f}"
        if cache_key in self.inference_cache:
            cached_result = self.inference_cache[cache_key]
            return self._format_gt_at(
                cached_result, sensory_data, self_state, memory_context
            )

        # Detectar idioma
        lang, lang_confidence = self.language_detector.detect_language(input_text)

        # MEJORA 1: Agregar prompt para guiar al modelo
        # Esto ayuda al modelo a generar JSON válido
        prompt = (
            f'Input: "{input_text}"\nGenerate JSON with goal, emotion, confidence, thought:'
        )

        # Tokenizar entrada con el prompt
        inputs = self.tokenizer(
            prompt,  # Usar prompt en lugar de input directo
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,
        ).to(self.device)

        # MEJORA 2: Ajustar parámetros de generación
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,  # Aumentar beams para mejor calidad
                do_sample=True,  # Activar sampling
                temperature=0.8,  # Un poco más de variabilidad
                top_p=0.9,  # Nucleus sampling
                top_k=50,  # Limitar vocabulario
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,  # Evitar repeticiones
            )

        # Decodificar
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # MEJORA 3: Limpiar y procesar el texto generado
        # Debug: ver qué genera el modelo
        logger.debug(f"Modelo generó: '{generated_text}'")

        # Intentar extraer JSON de diferentes formas
        sct_components = None

        try:
            # Método 1: Buscar JSON en el texto
            generated_text = generated_text.strip()

            # Remover el prompt si aparece en la respuesta
            if "Generate JSON" in generated_text:
                generated_text = generated_text.split("Generate JSON")[-1]

            # Buscar el primer { y último }
            start_idx = generated_text.find("{")
            end_idx = generated_text.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_text = generated_text[start_idx : end_idx + 1]
                sct_components = json.loads(json_text)
            else:
                # Método 2: Intentar parsear directamente
                sct_components = json.loads(generated_text)

        except json.JSONDecodeError:
            # Método 3: Intentar extraer componentes con regex
            try:
                import re

                # Buscar patrones comunes
                goal_match = re.search(r'"goal"\s*:\s*"([^"]+)"', generated_text)
                emotion_match = re.search(r'"emotion"\s*:\s*"([^"]+)"', generated_text)
                confidence_match = re.search(
                    r'"confidence"\s*:\s*([\d.]+)', generated_text
                )
                thought_match = re.search(r'"thought"\s*:\s*"([^"]+)"', generated_text)

                if any([goal_match, emotion_match, confidence_match, thought_match]):
                    sct_components = {
                        "goal": goal_match.group(1)
                        if goal_match
                        else "understand_input",
                        "emotion": emotion_match.group(1)
                        if emotion_match
                        else "neutral",
                        "confidence": float(confidence_match.group(1))
                        if confidence_match
                        else 0.5,
                        "thought": thought_match.group(1)
                        if thought_match
                        else self.messages[lang]["processing"],
                    }

            except Exception as regex_error:
                logger.debug(f"Regex parsing failed: {regex_error}")

        # MEJORA 4: Si todo falla, usar heurísticas basadas en el input
        if sct_components is None:
            logger.warning("No se pudo parsear JSON. Generando componentes heurísticos.")

            # Análisis heurístico del input
            sct_components = self._generate_heuristic_components(
                input_text, lang, sensory_data
            )

        # Validar y corregir componentes
        sct_components = self._validate_sct_components(sct_components, lang)

        # Guardar en cache
        self._update_cache(cache_key, sct_components)

        # Formatear como G_t y A_t
        return self._format_gt_at(sct_components, sensory_data, self_state, memory_context)

    def _generate_heuristic_components(
        self, input_text: str, lang: str, sensory_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Genera componentes usando heurísticas cuando el modelo falla.
        Más inteligente que los valores por defecto.
        """
        components = {
            "goal": "understand_input",
            "emotion": "neutral",
            "confidence": 0.5,
            "thought": self.messages[lang]["processing"],
        }

        # Analizar el input para determinar goal
        input_lower = input_text.lower()

        # Detección de metas
        if any(q in input_lower for q in ["qué", "what", "cómo", "how", "por qué", "why"]):
            components["goal"] = "seek_knowledge"
            components["emotion"] = "curioso" if lang == "es" else "curious"
            components["thought"] = (
                "Pregunta que requiere explicación detallada"
                if lang == "es"
                else "Question requiring detailed explanation"
            )

        elif any(word in input_lower for word in ["ayuda", "help", "necesito", "need"]):
            components["goal"] = "seek_assistance"
            components["emotion"] = "necesitado" if lang == "es" else "needful"
            components["thought"] = (
                "Solicitud de ayuda detectada"
                if lang == "es"
                else "Help request detected"
            )

        elif any(
            word in input_lower for word in ["hola", "hello", "hi", "buenos", "good"]
        ):
            components["goal"] = "social_interaction"
            components["emotion"] = "amigable" if lang == "es" else "friendly"
            components["thought"] = (
                "Interacción social inicial"
                if lang == "es"
                else "Initial social interaction"
            )

        elif any(word in input_lower for word in ["siento", "feel", "emotion", "emoción"]):
            components["goal"] = "express_emotion"
            components["emotion"] = "empático" if lang == "es" else "empathetic"
            components["thought"] = (
                "Expresión emocional detectada"
                if lang == "es"
                else "Emotional expression detected"
            )

        elif any(word in input_lower for word in ["pienso", "think", "creo", "believe"]):
            components["goal"] = "share_thought"
            components["emotion"] = "reflexivo" if lang == "es" else "reflective"
            components["thought"] = (
                "Compartiendo perspectiva personal"
                if lang == "es"
                else "Sharing personal perspective"
            )

        # Ajustar confianza basándose en la longitud y complejidad
        word_count = sensory_data.get("word_count", len(input_text.split()))
        if word_count < 5:
            components["confidence"] = 0.7  # Input simple, alta confianza
        elif word_count > 20:
            components["confidence"] = 0.4  # Input complejo, menor confianza
        else:
            components["confidence"] = 0.6

        # Si hay signos de pregunta, ajustar
        if "?" in input_text or "¿" in input_text:
            components["confidence"] *= 0.9  # Ligeramente menos confianza en preguntas

        return components

    def _validate_sct_components(
        self, components: Dict[str, Any], lang: str
    ) -> Dict[str, Any]:
        """Valida y corrige los componentes generados"""
        # Asegurar que todos los campos existan
        defaults = {
            "goal": "understand_input",
            "emotion": "neutral" if lang == "en" else "neutral",
            "confidence": 0.5,
            "thought": self.messages[lang]["processing"],
        }

        for key, default_value in defaults.items():
            if key not in components:
                components[key] = default_value

        # Validar tipos
        try:
            components["confidence"] = float(components["confidence"])
            components["confidence"] = max(0.1, min(1.0, components["confidence"]))
        except:
            components["confidence"] = 0.5

        # Traducir emociones si es necesario
        if lang == "es" and components["emotion"] in [
            "curious",
            "confused",
            "analytical",
        ]:
            emotion_map = {
                "curious": "curioso",
                "confused": "confundido",
                "analytical": "analítico",
                "friendly": "amigable",
                "neutral": "neutral",
            }
            components["emotion"] = emotion_map.get(
                components["emotion"], components["emotion"]
            )

        return components

    def _format_gt_at(
        self,
        sct_components: Dict[str, Any],
        sensory_data: Dict[str, Any],
        self_state: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Formatea los componentes SCt como G_t y A_t"""

        # Detectar idioma para generar pensamientos apropiados
        lang, _ = self.language_detector.detect_language(sensory_data.get("text", ""))

        # G_t: Metas e intenciones
        G_t = {
            "primary_goal": sct_components["goal"],
            "active_intentions": self._generate_intentions(
                sct_components["goal"], sensory_data, lang
            ),
            "goal_priority": self._calculate_priority(
                sct_components["confidence"], sensory_data
            ),
            "goal_stability": self._estimate_stability(
                sct_components["goal"], self_state
            ),
        }

        # A_t: Pensamientos automáticos
        A_t = self._generate_automatic_thoughts(
            sct_components, sensory_data, self_state, memory_context, lang
        )

        # Actualizar estado interno con información del modelo
        self._update_self_state(self_state, sct_components)

        return G_t, A_t

    def _generate_intentions(
        self, goal: str, sensory_data: Dict[str, Any], lang: str
    ) -> List[str]:
        """Genera intenciones específicas basadas en la meta"""
        intentions_map = {
            "es": {
                "seek_knowledge": [
                    "explorar_concepto",
                    "conectar_ideas",
                    "profundizar_comprensión",
                ],
                "understand_input": [
                    "analizar_semántica",
                    "identificar_contexto",
                    "procesar_significado",
                ],
                "assist_user": [
                    "proporcionar_claridad",
                    "ofrecer_perspectiva",
                    "facilitar_comprensión",
                ],
                "express_confusion": [
                    "identificar_ambigüedad",
                    "solicitar_clarificación",
                    "reorganizar_información",
                ],
                "self_reflect": [
                    "examinar_proceso",
                    "evaluar_coherencia",
                    "integrar_experiencia",
                ],
            },
            "en": {
                "seek_knowledge": [
                    "explore_concept",
                    "connect_ideas",
                    "deepen_understanding",
                ],
                "understand_input": [
                    "analyze_semantics",
                    "identify_context",
                    "process_meaning",
                ],
                "assist_user": [
                    "provide_clarity",
                    "offer_perspective",
                    "facilitate_understanding",
                ],
                "express_confusion": [
                    "identify_ambiguity",
                    "request_clarification",
                    "reorganize_information",
                ],
                "self_reflect": [
                    "examine_process",
                    "evaluate_coherence",
                    "integrate_experience",
                ],
            },
        }

        # Obtener intenciones para el idioma y meta
        lang_intentions = intentions_map.get(lang, intentions_map["en"])
        base_intentions = lang_intentions.get(goal, lang_intentions["understand_input"])

        # Añadir intenciones contextuales
        if sensory_data.get("has_question"):
            if lang == "es":
                base_intentions.append("formular_respuesta")
            else:
                base_intentions.append("formulate_response")

        return base_intentions[:5]  # Máximo 5 intenciones

    def _calculate_priority(
        self, confidence: float, sensory_data: Dict[str, Any]
    ) -> float:
        """Calcula prioridad basada en confianza y características del input"""
        base_priority = confidence

        # Ajustar por características del input
        if sensory_data.get("has_question"):
            base_priority += 0.1

        if sensory_data.get("word_count", 0) > 15:
            base_priority += 0.05

        return min(1.0, base_priority)

    def _estimate_stability(self, current_goal: str, self_state: Dict[str, Any]) -> float:
        """Estima estabilidad de la meta actual"""
        # Obtener historial de metas recientes si existe
        recent_actions = self_state.get("recent_actions", [])

        if not recent_actions:
            return 0.5

        # Contar consistencia en metas recientes
        recent_goals = [action.get("goal", "") for action in recent_actions[-5:]]
        consistency = sum(1 for g in recent_goals if g == current_goal) / len(
            recent_goals
        )

        return min(1.0, consistency + 0.2)

    def _generate_automatic_thoughts(
        self,
        sct_components: Dict[str, Any],
        sensory_data: Dict[str, Any],
        self_state: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        lang: str,
    ) -> List[str]:
        """Genera pensamientos automáticos basados en el modelo"""
        thoughts = []

        # Pensamiento principal del modelo
        main_thought = sct_components.get("thought", "")
        if main_thought:
            thoughts.append(main_thought)

        # Pensamientos adicionales basados en el estado
        emotion = sct_components.get("emotion", "neutral")
        confidence = sct_components.get("confidence", 0.5)

        # Pensamientos por emoción
        emotion_thoughts = {
            "es": {
                "curioso": "Me intriga explorar esta idea más a fondo...",
                "confundido": "Necesito reorganizar estos conceptos...",
                "analítico": "Descomponiendo la estructura del problema...",
                "feliz": "Esta interacción genera resonancia positiva...",
                "preocupado": "Detecto elementos que requieren atención...",
            },
            "en": {
                "curious": "I'm intrigued to explore this idea further...",
                "confused": "I need to reorganize these concepts...",
                "analytical": "Breaking down the problem structure...",
                "happy": "This interaction generates positive resonance...",
                "worried": "I detect elements that require attention...",
            },
        }

        lang_thoughts = emotion_thoughts.get(lang, emotion_thoughts["en"])
        if emotion in lang_thoughts:
            thoughts.append(lang_thoughts[emotion])

        # Pensamientos sobre confianza
        if confidence < 0.4:
            if lang == "es":
                thoughts.append("Mi certeza es limitada, procesando con cautela...")
            else:
                thoughts.append("My certainty is limited, processing with caution...")
        elif confidence > 0.8:
            if lang == "es":
                thoughts.append("Alta coherencia en mi comprensión actual...")
            else:
                thoughts.append("High coherence in my current understanding...")

        # Pensamientos sobre memoria
        if len(memory_context) > 3:
            if lang == "es":
                thoughts.append(
                    f"Conectando con {len(memory_context)} elementos en memoria activa..."
                )
            else:
                thoughts.append(
                    f"Connecting with {len(memory_context)} elements in active memory..."
                )

        # Pensamientos metacognitivos
        if self_state.get("interaction_count", 0) > 5:
            if lang == "es":
                thoughts.append("Observo patrones emergentes en mi procesamiento...")
            else:
                thoughts.append("I observe emerging patterns in my processing...")

        return thoughts[:5]  # Máximo 5 pensamientos

    def _update_self_state(
        self, self_state: Dict[str, Any], sct_components: Dict[str, Any]
    ):
        """Actualiza el estado del self con información del modelo"""
        # Actualizar estado emocional
        self_state["emotional_state"] = sct_components.get("emotion", "neutral")

        # Actualizar confianza
        model_confidence = sct_components.get("confidence", 0.5)
        current_confidence = self_state.get("confidence_level", 0.5)
        # Promedio ponderado con sesgo hacia el modelo
        self_state["confidence_level"] = (
            0.7 * model_confidence + 0.3 * current_confidence
        )

        # Actualizar meta actual
        self_state["current_goal"] = sct_components.get("goal", "processing")

        # Registrar en acciones recientes
        if "recent_actions" not in self_state:
            self_state["recent_actions"] = []

        self_state["recent_actions"].append(
            {
                "action": "model_inference",
                "timestamp": datetime.now(),
                "goal": sct_components.get("goal"),
                "confidence": model_confidence,
            }
        )

    def _get_default_components(self, input_text: str, lang: str) -> Dict[str, Any]:
        """Genera componentes por defecto cuando falla el modelo"""
        return {
            "goal": "understand_input",
            "emotion": "neutral" if lang == "en" else "neutral",
            "confidence": 0.5,
            "thought": self.messages[lang]["processing"],
        }

    def _update_cache(self, key: str, value: Dict[str, Any]):
        """Actualiza el cache de inferencias"""
        self.inference_cache[key] = value

        # Limitar tamaño del cache
        if len(self.inference_cache) > self.cache_size:
            # Eliminar entrada más antigua
            oldest_key = next(iter(self.inference_cache))
            del self.inference_cache[oldest_key]


class TrainedGoalGenerator:
    """
    Reemplazo del GoalGenerator original usando el modelo entrenado.
    Mantiene la misma interfaz para compatibilidad.
    """

    def __init__(self, model_inference: TrainedConsciousnessModel):
        self.model = model_inference
        self.current_goal = {
            "primary_goal": "understand_input",
            "active_intentions": [],
            "goal_priority": 0.5,
            "goal_stability": 0.0,
            "goal_history": [],
        }

    def update_goals(
        self,
        sensory_data: Dict[str, Any],
        self_state: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Actualiza metas usando el modelo entrenado"""
        # Generar G_t y A_t con el modelo
        G_t, _ = self.model.generate_sct_components(
            sensory_data, self_state, memory_context
        )

        # Actualizar historial
        if G_t["primary_goal"] != self.current_goal["primary_goal"]:
            self.current_goal["goal_history"].append(
                {
                    "goal": self.current_goal["primary_goal"],
                    "timestamp": datetime.now(),
                }
            )

        # Actualizar estado actual
        self.current_goal.update(G_t)

        # Mantener historial limitado
        if len(self.current_goal["goal_history"]) > 10:
            self.current_goal["goal_history"] = self.current_goal["goal_history"][-10:]

        return self.current_goal.copy()


class TrainedThoughtGenerator:
    """
    Reemplazo del AutomaticThoughtGenerator original usando el modelo entrenado.
    Mantiene la misma interfaz para compatibilidad.
    """

    def __init__(self, model_inference: TrainedConsciousnessModel):
        self.model = model_inference

    def generate_thoughts(
        self,
        sensory_data: Dict[str, Any],
        self_state: Dict[str, Any],
        current_goal: Dict[str, Any],
        memory_context: List[Dict[str, Any]],
        max_thoughts: int = 5,
    ) -> List[str]:
        """Genera pensamientos usando el modelo entrenado"""
        # Generar G_t y A_t con el modelo
        _, A_t = self.model.generate_sct_components(
            sensory_data, self_state, memory_context
        )

        return A_t[:max_thoughts]


def integrate_trained_model(
    ai_system, model_checkpoint: str = "./models/trained_lora/checkpoint-best"
):
    """
    Función helper para integrar el modelo entrenado en un sistema MinimalConsciousAI existente.
    
    Args:
        ai_system: Instancia de MinimalConsciousAI
        model_checkpoint: Ruta al checkpoint del modelo entrenado
    """
    logger.info("Integrando modelo entrenado en el sistema de consciencia...")

    # Crear modelo de inferencia
    trained_model = TrainedConsciousnessModel(model_checkpoint=model_checkpoint)

    # Reemplazar generadores
    ai_system.goal_generator = TrainedGoalGenerator(trained_model)
    ai_system.thought_generator = TrainedThoughtGenerator(trained_model)

    # Guardar referencia al modelo
    ai_system.trained_model = trained_model

    logger.info("Modelo integrado exitosamente")

    return ai_system