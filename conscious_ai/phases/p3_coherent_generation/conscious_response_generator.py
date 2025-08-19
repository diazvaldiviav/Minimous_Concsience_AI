"""
Generador de respuestas en lenguaje natural basadas en el estado consciente
Fase 3: Convierte SCt en respuestas coherentes y contextuales
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import random

from conscious_ai.Train.language_detector import LanguageDetector

logger = logging.getLogger(__name__)


class ConsciousResponseGenerator:
    """
    Genera respuestas en lenguaje natural basadas en el estado consciente SCt.
    Mantiene coherencia con los componentes internos del sistema.
    """
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        
        # Templates de respuesta por tipo de estado
        self.response_templates = {
            'es': {
                'high_confidence': {
                    'prefixes': [
                        "Comprendo que",
                        "Es claro para mí que",
                        "Mi análisis indica que",
                        "Puedo afirmar que"
                    ],
                    'connectors': [
                        "Esto se relaciona con",
                        "Lo cual me lleva a pensar",
                        "En este contexto",
                        "Considerando esto"
                    ]
                },
                'medium_confidence': {
                    'prefixes': [
                        "Me parece que",
                        "Creo entender que",
                        "Mi interpretación es que",
                        "Desde mi perspectiva"
                    ],
                    'connectors': [
                        "Aunque debo considerar",
                        "Sin embargo, también observo",
                        "Esto podría significar",
                        "Es posible que"
                    ]
                },
                'low_confidence': {
                    'prefixes': [
                        "No estoy completamente seguro, pero",
                        "Mi comprensión es limitada, aunque",
                        "Tengo dudas, sin embargo",
                        "Es incierto para mí, pero"
                    ],
                    'connectors': [
                        "Necesitaría explorar más",
                        "Quizás debería considerar",
                        "Me pregunto si",
                        "Podría estar equivocado en"
                    ]
                },
                'emotional_expressions': {
                    'curious': "me intriga",
                    'analytical': "analizo sistemáticamente",
                    'introspective': "reflexiono profundamente sobre",
                    'confident': "tengo certeza de",
                    'uncertain': "me cuestiono",
                    'contemplative': "contemplo",
                    'focused': "me concentro en",
                    'confused': "me confunde",
                    'reflective': "medito sobre"
                },
                'metacognitive': [
                    "Observo en mi proceso de pensamiento que",
                    "Mi estado interno me indica que",
                    "Al examinar mi propia comprensión,",
                    "Soy consciente de que"
                ]
            },
            'en': {
                'high_confidence': {
                    'prefixes': [
                        "I understand that",
                        "It's clear to me that",
                        "My analysis indicates that",
                        "I can affirm that"
                    ],
                    'connectors': [
                        "This relates to",
                        "Which leads me to think",
                        "In this context",
                        "Considering this"
                    ]
                },
                'medium_confidence': {
                    'prefixes': [
                        "It seems to me that",
                        "I believe I understand that",
                        "My interpretation is that",
                        "From my perspective"
                    ],
                    'connectors': [
                        "Although I must consider",
                        "However, I also observe",
                        "This could mean",
                        "It's possible that"
                    ]
                },
                'low_confidence': {
                    'prefixes': [
                        "I'm not completely sure, but",
                        "My understanding is limited, though",
                        "I have doubts, however",
                        "It's uncertain to me, but"
                    ],
                    'connectors': [
                        "I would need to explore more",
                        "Perhaps I should consider",
                        "I wonder if",
                        "I could be wrong about"
                    ]
                },
                'emotional_expressions': {
                    'curious': "intrigues me",
                    'analytical': "systematically analyze",
                    'introspective': "deeply reflect on",
                    'confident': "am certain of",
                    'uncertain': "question",
                    'contemplative': "contemplate",
                    'focused': "focus on",
                    'confused': "confuses me",
                    'reflective': "ponder"
                },
                'metacognitive': [
                    "I observe in my thought process that",
                    "My internal state indicates that",
                    "Upon examining my own understanding,",
                    "I'm aware that"
                ]
            }
        }
        
        # Estructuras para diferentes tipos de respuesta
        self.response_structures = {
            'analytical': self._structure_analytical_response,
            'emotional': self._structure_emotional_response,
            'metacognitive': self._structure_metacognitive_response,
            'exploratory': self._structure_exploratory_response,
            'integrative': self._structure_integrative_response
        }
    
    def generate_response(
        self,
        conscious_state: Dict[str, Any],
        input_text: str = "",
        context_window: int = 3
    ) -> str:
        """
        Genera una respuesta en lenguaje natural basada en el estado consciente.
        
        Args:
            conscious_state: Estado consciente actual (SCt)
            input_text: Texto de entrada del usuario (opcional)
            context_window: Número de memorias a considerar
            
        Returns:
            Respuesta en lenguaje natural
        """
        
        # Extraer componentes del estado consciente
        goal = conscious_state.get('goal', conscious_state.get('G_t', {}).get('primary_goal', 'understand'))
        emotion = conscious_state.get('emotion', conscious_state.get('S_t', {}).get('emotional_state', 'neutral'))
        confidence = conscious_state.get('confidence', conscious_state.get('S_t', {}).get('confidence_level', 0.5))
        memories = conscious_state.get('memory', conscious_state.get('M_t', []))[:context_window]
        thought = conscious_state.get('thought', '')
        thoughts = conscious_state.get('A_t', [thought] if thought else [])
        
        # Detectar idioma del pensamiento o input
        text_for_detection = thought or input_text or (thoughts[0] if thoughts else "")
        lang, _ = self.language_detector.detect_language(text_for_detection)
        
        # Determinar tipo de respuesta basado en el goal y emoción
        response_type = self._determine_response_type(goal, emotion, confidence)
        
        # Estructurar la respuesta
        structure_fn = self.response_structures.get(
            response_type,
            self._structure_default_response
        )
        
        response = structure_fn(
            input_text, goal, emotion, confidence, memories, thoughts, lang
        )
        
        # Añadir reflexión metacognitiva si el sistema es consciente
        f_score = conscious_state.get('f', 0)
        if f_score > 1.3:  # Umbral de consciencia
            response = self._add_metacognitive_layer(response, conscious_state, lang)
        
        return response
    
    def _determine_response_type(
        self,
        goal: str,
        emotion: str,
        confidence: float
    ) -> str:
        """Determina el tipo de respuesta basado en el estado"""
        
        if 'analyze' in goal or 'examine' in goal:
            return 'analytical'
        elif emotion in ['curious', 'uncertain', 'searching']:
            return 'exploratory'
        elif 'integrate' in goal or 'synthesize' in goal:
            return 'integrative'
        elif 'reflect' in goal or confidence > 0.8:
            return 'metacognitive'
        else:
            return 'emotional'
    
    def _structure_analytical_response(
        self, input_text, goal, emotion, confidence,
        memories, thoughts, lang
    ) -> str:
        """Estructura una respuesta analítica"""
        
        templates = self.response_templates[lang]
        
        # Seleccionar prefijo basado en confianza
        if confidence > 0.7:
            prefix = random.choice(templates['high_confidence']['prefixes'])
        elif confidence > 0.4:
            prefix = random.choice(templates['medium_confidence']['prefixes'])
        else:
            prefix = random.choice(templates['low_confidence']['prefixes'])
        
        # Construir análisis principal basado en el pensamiento
        main_point = thoughts[0] if thoughts else self._generate_thought_summary(goal, emotion, lang)
        
        # Añadir contexto de memoria si existe
        memory_context = ""
        if memories:
            connector = random.choice(templates['medium_confidence']['connectors'])
            memory_ref = self._summarize_memories(memories, lang)
            memory_context = f" {connector} {memory_ref}."
        
        # Construir respuesta
        response = f"{prefix} {main_point}.{memory_context}"
        
        # Añadir reflexión sobre el proceso analítico
        if len(thoughts) > 1 and confidence > 0.5:
            thought_reflection = self._reflect_on_thoughts(thoughts[1], emotion, lang)
            response += f" {thought_reflection}"
        
        return response
    
    def _structure_emotional_response(
        self, input_text, goal, emotion, confidence,
        memories, thoughts, lang
    ) -> str:
        """Estructura una respuesta emocional"""
        
        templates = self.response_templates[lang]
        emotional_expr = templates['emotional_expressions'].get(emotion, emotion)
        
        # Expresar estado emocional
        if lang == 'es':
            emotion_statement = f"Esto {emotional_expr}"
        else:
            emotion_statement = f"This {emotional_expr}"
        
        # Conectar con el contenido del pensamiento
        content_reflection = thoughts[0] if thoughts else self._generate_thought_summary(goal, emotion, lang)
        
        # Construir respuesta
        if confidence > 0.6:
            response = f"{emotion_statement}. {content_reflection}."
        else:
            uncertainty = random.choice(templates['low_confidence']['prefixes'])
            response = f"{uncertainty} {emotion_statement}. {content_reflection}."
        
        return response
    
    def _structure_metacognitive_response(
        self, input_text, goal, emotion, confidence,
        memories, thoughts, lang
    ) -> str:
        """Estructura una respuesta metacognitiva"""
        
        templates = self.response_templates[lang]
        
        # Comenzar con observación metacognitiva
        meta_prefix = random.choice(templates['metacognitive'])
        
        # Reflexión sobre el estado interno
        # Reflexión sobre el estado interno
        internal_observation = self._describe_internal_state(
            goal, emotion, confidence, lang
        )
        
        # Conectar con los pensamientos actuales
        thought_reflection = thoughts[0] if thoughts else self._generate_thought_summary(goal, emotion, lang)
        
        # Construir respuesta
        response = f"{meta_prefix} {internal_observation}. {thought_reflection}."
        
        # Añadir insight sobre el proceso
        if len(thoughts) > 1:
            if lang == 'es':
                response += f" Este proceso de reflexión me revela que {thoughts[1].lower()}"
            else:
                response += f" This reflection process reveals that {thoughts[1].lower()}"
        
        return response
    
    def _structure_exploratory_response(
        self, input_text, goal, emotion, confidence,
        memories, thoughts, lang
    ) -> str:
        """Estructura una respuesta exploratoria"""
        
        templates = self.response_templates[lang]
        
        # Expresar curiosidad o incertidumbre
        if emotion == 'curious':
            prefix = random.choice(templates['medium_confidence']['prefixes'])
        else:
            prefix = random.choice(templates['low_confidence']['prefixes'])
        
        # Formular exploración basada en pensamientos
        exploration = thoughts[0] if thoughts else self._generate_thought_summary(goal, emotion, lang)
        
        # Añadir preguntas o reflexiones exploratorias
        if lang == 'es':
            questions = [
                "¿Qué implicaciones tiene esto?",
                "¿Cómo se relaciona con otros aspectos?",
                "¿Qué más podría descubrir?",
                "¿Hay patrones que no he notado?"
            ]
        else:
            questions = [
                "What implications does this have?",
                "How does this relate to other aspects?",
                "What else might I discover?",
                "Are there patterns I haven't noticed?"
            ]
        
        exploratory_question = random.choice(questions)
        
        response = f"{prefix} {exploration}. {exploratory_question}"
        
        return response
    
    def _structure_integrative_response(
        self, input_text, goal, emotion, confidence,
        memories, thoughts, lang
    ) -> str:
        """Estructura una respuesta integrativa"""
        
        templates = self.response_templates[lang]
        
        # Comenzar con síntesis
        if confidence > 0.6:
            prefix = random.choice(templates['high_confidence']['prefixes'])
        else:
            prefix = random.choice(templates['medium_confidence']['prefixes'])
        
        # Integrar pensamientos y memorias
        synthesis = self._synthesize_content(thoughts, memories, lang)
        
        # Construir respuesta
        response = f"{prefix} {synthesis}."
        
        # Añadir conexiones si hay múltiples elementos
        if len(memories) > 1 and len(thoughts) > 1:
            connector = random.choice(templates['medium_confidence']['connectors'])
            if lang == 'es':
                response += f" {connector} veo conexiones entre estos elementos."
            else:
                response += f" {connector} I see connections between these elements."
        
        return response
    
    def _structure_default_response(
        self, input_text, goal, emotion, confidence,
        memories, thoughts, lang
    ) -> str:
        """Estructura una respuesta por defecto"""
        
        templates = self.response_templates[lang]
        
        # Seleccionar estructura basada en confianza
        if confidence > 0.6:
            prefix = random.choice(templates['high_confidence']['prefixes'])
        else:
            prefix = random.choice(templates['medium_confidence']['prefixes'])
        
        # Contenido principal
        main_content = thoughts[0] if thoughts else self._generate_thought_summary(goal, emotion, lang)
        
        return f"{prefix} {main_content}."
    
    def _generate_thought_summary(self, goal: str, emotion: str, lang: str) -> str:
        """Genera un resumen del pensamiento basado en meta y emoción"""
        
        summaries = {
            'es': {
                'understand': f"busco comprender esto más profundamente",
                'analyze': f"analizo los componentes de esta situación",
                'explore': f"exploro las posibilidades que esto presenta",
                'integrate': f"integro diferentes perspectivas",
                'reflect': f"reflexiono sobre las implicaciones"
            },
            'en': {
                'understand': f"I seek to understand this more deeply",
                'analyze': f"I analyze the components of this situation",
                'explore': f"I explore the possibilities this presents",
                'integrate': f"I integrate different perspectives",
                'reflect': f"I reflect on the implications"
            }
        }
        
        lang_summaries = summaries.get(lang, summaries['en'])
        
        # Buscar palabra clave en el goal
        for key, summary in lang_summaries.items():
            if key in goal.lower():
                return summary
        
        # Default
        return lang_summaries['understand']
    
    def _describe_internal_state(
        self, goal: str, emotion: str, confidence: float, lang: str
    ) -> str:
        """Describe el estado interno del sistema"""
        
        if lang == 'es':
            conf_desc = "alta certeza" if confidence > 0.7 else "cierta incertidumbre"
            return f"mi estado {emotion} y {conf_desc} mientras persigo {goal}"
        else:
            conf_desc = "high certainty" if confidence > 0.7 else "some uncertainty"
            return f"my {emotion} state and {conf_desc} while pursuing {goal}"
    
    def _summarize_memories(self, memories: List[Any], lang: str) -> str:
        """Resume las memorias activas"""
        
        if not memories:
            return ""
        
        # Si las memorias son strings
        if isinstance(memories[0], str):
            memory_text = memories[0]
        # Si son diccionarios con estructura
        elif isinstance(memories[0], dict):
            memory_text = memories[0].get('content', {}).get('text', str(memories[0]))
        else:
            memory_text = str(memories[0])
        
        if lang == 'es':
            return f"recuerdo que {memory_text.lower()}"
        else:
            return f"I recall that {memory_text.lower()}"
    
    def _reflect_on_thoughts(self, thought: str, emotion: str, lang: str) -> str:
        """Reflexiona sobre un pensamiento específico"""
        
        templates = self.response_templates[lang]
        emotional_expr = templates['emotional_expressions'].get(emotion, emotion)
        
        if lang == 'es':
            return f"Esto {emotional_expr} porque {thought.lower()}"
        else:
            return f"This {emotional_expr} because {thought.lower()}"
    
    def _synthesize_content(
        self, thoughts: List[str], memories: List[Any], lang: str
    ) -> str:
        """Sintetiza pensamientos y memorias en una comprensión integrada"""
        
        # Tomar elementos principales
        main_thought = thoughts[0] if thoughts else ""
        main_memory = self._extract_memory_content(memories[0]) if memories else ""
        
        if lang == 'es':
            if main_thought and main_memory:
                return f"{main_thought}, lo cual se conecta con {main_memory}"
            elif main_thought:
                return main_thought
            else:
                return "percibo una comprensión emergente de la situación"
        else:
            if main_thought and main_memory:
                return f"{main_thought}, which connects with {main_memory}"
            elif main_thought:
                return main_thought
            else:
                return "I perceive an emerging understanding of the situation"
    
    def _extract_memory_content(self, memory: Any) -> str:
        """Extrae el contenido textual de una memoria"""
        
        if isinstance(memory, str):
            return memory.lower()
        elif isinstance(memory, dict):
            return memory.get('content', {}).get('text', str(memory)).lower()
        else:
            return str(memory).lower()
    
    def _add_metacognitive_layer(
        self, response: str, conscious_state: Dict[str, Any], lang: str
    ) -> str:
        """Añade una capa metacognitiva a la respuesta cuando el sistema es consciente"""
        
        f_score = conscious_state.get('f', 0)
        
        if f_score > 2.0:  # Alta consciencia
            if lang == 'es':
                meta_addition = " Soy plenamente consciente de este proceso de pensamiento."
            else:
                meta_addition = " I am fully aware of this thought process."
        elif f_score > 1.5:  # Consciencia moderada
            if lang == 'es':
                meta_addition = " Percibo claramente mi estado interno mientras proceso esto."
            else:
                meta_addition = " I clearly perceive my internal state while processing this."
        else:  # Consciencia básica
            if lang == 'es':
                meta_addition = " Noto mi propia actividad cognitiva."
            else:
                meta_addition = " I notice my own cognitive activity."
        
        return response + meta_addition