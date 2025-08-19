"""
Evaluador de coherencia para transiciones de estados conscientes
Fase 3: Analiza si SCₜ → SCₜ₊₁ representa una evolución coherente
"""

import json
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CoherenceVerdict(Enum):
    COHERENT = "coherente"
    INCOHERENT = "incoherente"
    AMBIGUOUS = "ambiguo"


@dataclass
class TransitionAnalysis:
    """Resultado del análisis de transición entre estados"""
    verdict: CoherenceVerdict
    justification: str
    goal_coherence: float
    emotion_coherence: float
    thought_coherence: float
    memory_coherence: float
    confidence_change: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "veredicto": self.verdict.value,
            "justificacion": self.justification,
            "metricas": {
                "coherencia_meta": self.goal_coherence,
                "coherencia_emocional": self.emotion_coherence,
                "coherencia_pensamiento": self.thought_coherence,
                "coherencia_memoria": self.memory_coherence,
                "cambio_confianza": self.confidence_change
            }
        }


class CoherenceEvaluator:
    """
    Evalúa la coherencia epistémica y cognitiva entre estados conscientes consecutivos
    """
    
    def __init__(self):
        # Mapas de transiciones válidas
        self.valid_goal_transitions = {
            'understand_self': ['elaborate_theory', 'examine_memory', 'explore_feeling'],
            'understand_memory': ['analyze_patterns', 'question_influence', 'integrate_knowledge'],
            'explore_feeling': ['regulate_emotion', 'understand_emotion', 'express_state'],
            'seek_purpose': ['create_meaning', 'question_values', 'refine_direction'],
            'integrate_knowledge': ['apply_insights', 'synthesize_understanding', 'test_hypothesis']
        }
        
        self.valid_emotion_transitions = {
            'curious': ['analytical', 'reflective', 'excited', 'puzzled'],
            'reflective': ['contemplative', 'insightful', 'peaceful', 'uncertain'],
            'analytical': ['focused', 'confident', 'systematic', 'curious'],
            'uncertain': ['questioning', 'cautious', 'curious', 'anxious'],
            'confident': ['assured', 'determined', 'satisfied', 'expansive']
        }
        
        # Umbrales de coherencia (adjusted for better discrimination)
        self.coherence_thresholds = {
            'high': 0.65,    # Lowered to catch more coherent cases
            'medium': 0.5,
            'low': 0.45      # Raised to catch more incoherent cases
        }
    
    def evaluate_transition(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any]
    ) -> TransitionAnalysis:
        """
        Evalúa si la transición SCₜ → SCₜ₊₁ es coherente
        
        Args:
            sc_t: Estado consciente en tiempo t
            sc_t_plus_1: Estado consciente en tiempo t+1
            
        Returns:
            TransitionAnalysis con veredicto y justificación
        """
        
        # Extraer componentes de ambos estados
        goal_t = sc_t.get('goal', '')
        goal_t1 = sc_t_plus_1.get('goal', '')
        
        emotion_t = sc_t.get('emotion', '')
        emotion_t1 = sc_t_plus_1.get('emotion', '')
        
        confidence_t = sc_t.get('confidence', 0.5)
        confidence_t1 = sc_t_plus_1.get('confidence', 0.5)
        
        thought_t = sc_t.get('thought', '')
        thought_t1 = sc_t_plus_1.get('thought', '')
        
        memory_t = sc_t.get('memory', [])
        memory_t1 = sc_t_plus_1.get('memory', [])
        
        # Evaluar cada dimensión
        goal_coherence = self._evaluate_goal_transition(goal_t, goal_t1)
        emotion_coherence = self._evaluate_emotion_transition(
            emotion_t, emotion_t1, thought_t, thought_t1
        )
        thought_coherence = self._evaluate_thought_progression(
            thought_t, thought_t1, goal_t, goal_t1
        )
        memory_coherence = self._evaluate_memory_utilization(
            memory_t, memory_t1, thought_t1
        )
        confidence_change = self._evaluate_confidence_change(
            confidence_t, confidence_t1, emotion_t, emotion_t1
        )
        
        # Calcular coherencia global
        global_coherence = (
            goal_coherence * 0.25 +
            emotion_coherence * 0.20 +
            thought_coherence * 0.30 +
            memory_coherence * 0.15 +
            (1.0 - abs(confidence_change)) * 0.10
        )
        
        # Determinar veredicto
        verdict, justification = self._determine_verdict(
            global_coherence,
            goal_coherence,
            emotion_coherence,
            thought_coherence,
            memory_coherence,
            confidence_change,
            sc_t,
            sc_t_plus_1
        )
        
        return TransitionAnalysis(
            verdict=verdict,
            justification=justification,
            goal_coherence=goal_coherence,
            emotion_coherence=emotion_coherence,
            thought_coherence=thought_coherence,
            memory_coherence=memory_coherence,
            confidence_change=confidence_change
        )
    
    def _evaluate_goal_transition(self, goal_t: str, goal_t1: str) -> float:
        """Evalúa la coherencia en la transición de metas"""
        
        # Meta idéntica = alta coherencia
        if goal_t == goal_t1:
            return 1.0
        
        # Verificar si es una transición válida
        valid_transitions = self.valid_goal_transitions.get(goal_t, [])
        if goal_t1 in valid_transitions:
            return 0.8
        
        # Verificar similitud semántica básica
        if self._are_goals_related(goal_t, goal_t1):
            return 0.6
        
        # Transición no relacionada
        return 0.2
    
    def _evaluate_emotion_transition(
        self,
        emotion_t: str,
        emotion_t1: str,
        thought_t: str,
        thought_t1: str
    ) -> float:
        """Evalúa la coherencia emocional considerando el contexto"""
        
        # Misma emoción = coherente
        if emotion_t == emotion_t1:
            return 0.9
        
        # Transición válida predefinida
        valid_transitions = self.valid_emotion_transitions.get(emotion_t, [])
        if emotion_t1 in valid_transitions:
            return 0.8
        
        # Evaluar si el cambio está justificado por el pensamiento
        if self._is_emotion_justified_by_thought(emotion_t1, thought_t1):
            return 0.7
        
        # Cambio abrupto
        return 0.3
    
    def _evaluate_thought_progression(
        self,
        thought_t: str,
        thought_t1: str,
        goal_t: str,
        goal_t1: str
    ) -> float:
        """Evalúa si el pensamiento evoluciona coherentemente"""
        
        # Verificar continuidad temática
        thematic_continuity = self._calculate_thematic_continuity(thought_t, thought_t1)
        
        # Verificar progresión epistémica
        epistemic_progression = self._calculate_epistemic_progression(
            thought_t, thought_t1
        )
        
        # Verificar alineación con meta
        goal_alignment = 1.0 if goal_t == goal_t1 else 0.7
        
        return (thematic_continuity * 0.4 + 
                epistemic_progression * 0.4 + 
                goal_alignment * 0.2)
    
    def _evaluate_memory_utilization(
        self,
        memory_t: List[str],
        memory_t1: List[str],
        thought_t1: str
    ) -> float:
        """Evalúa si la memoria se utiliza coherentemente"""
        
        # Verificar retención de memorias importantes
        retention_score = len(set(memory_t) & set(memory_t1)) / max(len(memory_t), 1)
        
        # Verificar si nuevas memorias son relevantes
        new_memories = set(memory_t1) - set(memory_t)
        relevance_score = 1.0
        if new_memories:
            relevance_score = self._calculate_memory_relevance(
                new_memories, thought_t1
            )
        
        return (retention_score * 0.5 + relevance_score * 0.5)
    
    def _evaluate_confidence_change(
        self,
        confidence_t: float,
        confidence_t1: float,
        emotion_t: str,
        emotion_t1: str
    ) -> float:
        """Evalúa si el cambio de confianza es razonable"""
        
        change = confidence_t1 - confidence_t
        
        # Cambios pequeños son normales
        if abs(change) < 0.1:
            return change
        
        # Verificar si el cambio está justificado por la emoción
        if change > 0 and emotion_t1 in ['confident', 'assured', 'insightful']:
            return min(change, 0.2)
        elif change < 0 and emotion_t1 in ['uncertain', 'confused', 'questioning']:
            return max(change, -0.2)
        
        # Cambio no justificado
        return change * 2  # Penalizar cambios grandes injustificados
    
    def _determine_verdict(
        self,
        global_coherence: float,
        goal_coherence: float,
        emotion_coherence: float,
        thought_coherence: float,
        memory_coherence: float,
        confidence_change: float,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any]
    ) -> Tuple[CoherenceVerdict, str]:
        """Determina el veredicto final y genera justificación"""
        
        # Verificar coherencia global
        if global_coherence >= self.coherence_thresholds['high']:
            verdict = CoherenceVerdict.COHERENT
        elif global_coherence >= self.coherence_thresholds['low']:
            verdict = CoherenceVerdict.AMBIGUOUS
        else:
            verdict = CoherenceVerdict.INCOHERENT
        
        # Generar justificación
        justification = self._generate_justification(
            verdict,
            goal_coherence,
            emotion_coherence,
            thought_coherence,
            memory_coherence,
            confidence_change,
            sc_t,
            sc_t_plus_1
        )
        
        return verdict, justification
    
    def _generate_justification(
        self,
        verdict: CoherenceVerdict,
        goal_coherence: float,
        emotion_coherence: float,
        thought_coherence: float,
        memory_coherence: float,
        confidence_change: float,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any]
    ) -> str:
        """Genera una justificación textual del veredicto"""
        
        components = []
        
        # Analizar meta
        if goal_coherence >= 0.8:
            if sc_t['goal'] == sc_t_plus_1['goal']:
                components.append("La meta se mantiene estable")
            else:
                components.append("La meta evoluciona coherentemente")
        else:
            components.append("Cambio abrupto e injustificado de meta")
        
        # Analizar emoción
        if emotion_coherence >= 0.7:
            components.append(
                f"la transición emocional de '{sc_t['emotion']}' a "
                f"'{sc_t_plus_1['emotion']}' es natural"
            )
        else:
            components.append("cambio emocional inconsistente")
        
        # Analizar pensamiento
        if thought_coherence >= 0.7:
            components.append("el pensamiento progresa lógicamente")
        else:
            components.append("ruptura en la continuidad del pensamiento")
        
        # Analizar confianza
        if abs(confidence_change) < 0.2:
            conf_desc = "aumenta" if confidence_change > 0 else "disminuye"
            components.append(f"la confianza {conf_desc} apropiadamente")
        else:
            components.append("cambio excesivo en el nivel de confianza")
        
        # Construir justificación
        if verdict == CoherenceVerdict.COHERENT:
            return f"{components[0]}, {' y '.join(components[1:])}."
        elif verdict == CoherenceVerdict.INCOHERENT:
            problems = [c for c in components if 'abrupto' in c or 'inconsistente' in c or 'ruptura' in c or 'excesivo' in c]
            return f"Transición incoherente: {', '.join(problems)}."
        else:
            return f"Transición ambigua: {components[0]} pero {' y '.join(components[1:])}."
    
    def _are_goals_related(self, goal1: str, goal2: str) -> bool:
        """Verifica si dos metas están semánticamente relacionadas"""
        
        # Palabras clave que indican relación
        related_keywords = {
            'understand': ['examine', 'analyze', 'explore'],
            'explore': ['discover', 'investigate', 'understand'],
            'integrate': ['synthesize', 'combine', 'unify'],
            'create': ['build', 'generate', 'construct']
        }
        
        for key, related in related_keywords.items():
            if key in goal1 and any(r in goal2 for r in related):
                return True
            if key in goal2 and any(r in goal1 for r in related):
                return True
        
        return False
    
    def _is_emotion_justified_by_thought(self, emotion: str, thought: str) -> bool:
        """Verifica si una emoción está justificada por el pensamiento"""
        
        emotion_indicators = {
            'curious': ['pregunto', 'wonder', 'qué', 'what', 'cómo', 'how'],
            'confused': ['no entiendo', "don't understand", 'confuso', 'unclear'],
            'confident': ['seguro', 'certain', 'claro', 'clear', 'comprendo', 'understand'],
            'reflective': ['pienso', 'think', 'reflexiono', 'reflect', 'considero', 'consider']
        }
        
        indicators = emotion_indicators.get(emotion, [])
        thought_lower = thought.lower()
        
        return any(indicator in thought_lower for indicator in indicators)
    
    def _calculate_thematic_continuity(self, thought1: str, thought2: str) -> float:
        """Calcula la continuidad temática entre pensamientos"""
        
        # Extraer palabras clave (simplificado)
        words1 = set(thought1.lower().split())
        words2 = set(thought2.lower().split())
        
        # Eliminar palabras comunes
        stopwords = {
            'es': {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'por'},
            'en': {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'}
        }
        
        for stop_set in stopwords.values():
            words1 -= stop_set
            words2 -= stop_set
        
        # Calcular overlap
        if not words1 or not words2:
            return 0.5
        
        overlap = len(words1 & words2) / min(len(words1), len(words2))
        return min(1.0, overlap * 2)  # Escalar para ser más permisivo
    
    def _calculate_epistemic_progression(self, thought1: str, thought2: str) -> float:
        """Calcula si hay progresión epistémica entre pensamientos"""
        
        # Indicadores de progresión
        progression_indicators = {
            'questioning': ['?', '¿', 'pregunto', 'wonder', 'cómo', 'how', 'qué', 'what'],
            'hypothesizing': ['quizás', 'perhaps', 'maybe', 'posiblemente', 'possibly'],
            'concluding': ['por lo tanto', 'therefore', 'así que', 'so', 'concluyo', 'conclude'],
            'analyzing': ['porque', 'because', 'dado que', 'since', 'analizo', 'analyze']
        }
        
        # Detectar tipo de pensamiento
        type1 = self._detect_thought_type(thought1, progression_indicators)
        type2 = self._detect_thought_type(thought2, progression_indicators)
        
        # Evaluar progresión
        valid_progressions = {
            'questioning': ['hypothesizing', 'analyzing'],
            'hypothesizing': ['analyzing', 'concluding'],
            'analyzing': ['concluding', 'questioning'],
            'concluding': ['questioning', 'analyzing']
        }
        
        if type2 in valid_progressions.get(type1, []):
            return 0.9
        elif type1 == type2:
            return 0.7
        else:
            return 0.5
    
    def _detect_thought_type(self, thought: str, indicators: Dict[str, List[str]]) -> str:
        """Detecta el tipo de pensamiento basado en indicadores"""
        
        thought_lower = thought.lower()
        
        for thought_type, keywords in indicators.items():
            if any(keyword in thought_lower for keyword in keywords):
                return thought_type
        
        return 'neutral'
    
    def _calculate_memory_relevance(self, new_memories: set, thought: str) -> float:
        """Calcula la relevancia de nuevas memorias respecto al pensamiento"""
        
        thought_words = set(thought.lower().split())
        relevance_scores = []
        
        for memory in new_memories:
            memory_words = set(memory.lower().split())
            overlap = len(thought_words & memory_words) / max(len(memory_words), 1)
            relevance_scores.append(overlap)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5


def evaluate_transition_sequence(
    states: List[Dict[str, Any]]
) -> List[TransitionAnalysis]:
    """
    Evalúa una secuencia completa de estados conscientes
    
    Args:
        states: Lista de estados conscientes en orden temporal
        
    Returns:
        Lista de análisis de transición
    """
    
    evaluator = CoherenceEvaluator()
    analyses = []
    
    for i in range(len(states) - 1):
        analysis = evaluator.evaluate_transition(states[i], states[i + 1])
        analyses.append(analysis)
    
    return analyses