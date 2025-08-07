"""
Evaluador de coherencia basado en modelos entrenados
Usa embeddings y similitud semántica para evaluar transiciones
"""

import json
import logging
import torch
import numpy as np
import os
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceVerdict, TransitionAnalysis

logger = logging.getLogger(__name__)


class ModelBasedCoherenceEvaluator:
    """
    Evalúa coherencia usando modelos de embeddings y similitud semántica
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        coherence_threshold: float = 0.7,
        device: str = None,
        use_ml_classifier: bool = True,
        classifier_path: str = "./models/coherence_classifier"
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Cargar modelo de embeddings multilingüe
        logger.info(f"Cargando modelo de embeddings: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        
        # Umbrales de coherencia
        self.coherence_thresholds = {
            'high': coherence_threshold,
            'medium': coherence_threshold * 0.7,
            'low': coherence_threshold * 0.5
        }
        
        # Cache de embeddings para eficiencia
        self.embedding_cache = {}
        self.max_cache_size = 1000

        # Intentar cargar clasificador ML si existe
        self.ml_classifier = None
        self.use_ml_classifier = use_ml_classifier

        if use_ml_classifier and os.path.exists(classifier_path):
            try:
                from conscious_ai.coherence_evaluator_model.model_training.coherence_classifier_trainer import load_and_use_classifier
                self.ml_classifier = load_and_use_classifier(classifier_path)
                logger.info("✓ Clasificador ML de coherencia cargado exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo cargar clasificador ML: {e}")
                logger.info("Usando evaluación basada en embeddings únicamente")
    
     
    def evaluate_transition(
        self,
        sc_t: Dict[str, Any],
        sc_t_plus_1: Dict[str, Any]
    ) -> TransitionAnalysis:
        """
        Evalúa coherencia usando embeddings y opcionalmente clasificador ML
        """
        ### LOG ###
        logger.debug(f"--- NUEVA EVALUACIÓN DE TRANSICIÓN ---")
        logger.debug(f"evaluate_transition RECIBE sc_t:\n{json.dumps(sc_t, indent=2, ensure_ascii=False)}")
        logger.debug(f"evaluate_transition RECIBE sc_t_plus_1:\n{json.dumps(sc_t_plus_1, indent=2, ensure_ascii=False)}")
        
        
        # Si hay clasificador ML disponible, usarlo primero
        if self.ml_classifier is not None:
            ### LOG ###
            logger.debug("Intentando usar clasificador ML...")
            try:
                # Obtener predicción del clasificador
                ml_verdict, ml_confidence = self.ml_classifier(sc_t, sc_t_plus_1)
                
                # Si el clasificador tiene alta confianza, usar su veredicto
                if ml_confidence > 0.8:
                    logger.debug(f"Usando veredicto ML: {ml_verdict} (confianza: {ml_confidence:.2f})")
                    
                    # Mapear veredicto ML a enum
                    verdict_map = {
                        'coherent': CoherenceVerdict.COHERENT,
                        'incoherent': CoherenceVerdict.INCOHERENT,
                        'ambiguous': CoherenceVerdict.AMBIGUOUS
                    }
                    
                    # Aún calculamos métricas detalladas para el análisis
                    components_t = self._extract_components(sc_t)
                    components_t1 = self._extract_components(sc_t_plus_1)
                    embeddings_t = self._get_embeddings(components_t)
                    embeddings_t1 = self._get_embeddings(components_t1)
                    
                    # Calcular métricas individuales
                    goal_coherence = self._calculate_semantic_similarity(
                        embeddings_t['goal'], embeddings_t1['goal']
                    )
                    emotion_coherence = self._calculate_transition_coherence(
                        components_t['emotion'], components_t1['emotion'],
                        embeddings_t['thought'], embeddings_t1['thought']
                    )
                    thought_coherence = self._calculate_thought_progression(
                        embeddings_t['thought'], embeddings_t1['thought'],
                        embeddings_t['full_state'], embeddings_t1['full_state']
                    )
                    memory_coherence = self._calculate_memory_coherence(
                        components_t['memory'], components_t1['memory'],
                        embeddings_t1['thought']
                    )
                    confidence_change = components_t1['confidence'] - components_t['confidence']
                    
                    # Generar justificación que incluya el uso del ML
                    justification = self._generate_ml_based_justification(
                        verdict_map[ml_verdict],
                        ml_confidence,
                        goal_coherence,
                        emotion_coherence,
                        thought_coherence,
                        components_t,
                        components_t1
                    )
                    
                    return TransitionAnalysis(
                        verdict=verdict_map[ml_verdict],
                        justification=justification,
                        goal_coherence=goal_coherence,
                        emotion_coherence=emotion_coherence,
                        thought_coherence=thought_coherence,
                        memory_coherence=memory_coherence,
                        confidence_change=confidence_change
                    )
                    
            except Exception as e:
                logger.warning(f"Error usando clasificador ML: {e}")
                # Continuar con evaluación basada en embeddings
        
        # Evaluación estándar basada en embeddings (fallback o principal)
        return self._evaluate_with_embeddings(sc_t, sc_t_plus_1)
    
    def _evaluate_with_embeddings(
    self,
    sc_t: Dict[str, Any],
    sc_t_plus_1: Dict[str, Any]
    ) -> TransitionAnalysis:
     """
     Evaluación original basada en embeddings
     """
    
     # Extraer componentes
     components_t = self._extract_components(sc_t)
     components_t1 = self._extract_components(sc_t_plus_1)
    
     # Calcular embeddings para cada componente
     embeddings_t = self._get_embeddings(components_t)
     embeddings_t1 = self._get_embeddings(components_t1)
    
     # Calcular coherencia por componente
     goal_coherence = self._calculate_semantic_similarity(
        embeddings_t['goal'], embeddings_t1['goal']
     )
    
     emotion_coherence = self._calculate_transition_coherence(
        components_t['emotion'], components_t1['emotion'],
        embeddings_t['thought'], embeddings_t1['thought']
     )
    
     thought_coherence = self._calculate_thought_progression(
        embeddings_t['thought'], embeddings_t1['thought'],
        embeddings_t['full_state'], embeddings_t1['full_state']
     )
    
     memory_coherence = self._calculate_memory_coherence(
        components_t['memory'], components_t1['memory'],
        embeddings_t1['thought']
     )
    
     confidence_change = components_t1['confidence'] - components_t['confidence']
     confidence_coherence = self._evaluate_confidence_progression(
        confidence_change, emotion_coherence, thought_coherence
    )
    
     # Calcular coherencia global usando pesos aprendidos
     global_coherence = self._calculate_global_coherence(
        goal_coherence,
        emotion_coherence,
        thought_coherence,
        memory_coherence,
        confidence_coherence
    )
    
     # Determinar veredicto
     verdict = self._determine_verdict(global_coherence)
    
     # Generar justificación basada en análisis
     justification = self._generate_justification(
        verdict,
        goal_coherence,
        emotion_coherence,
        thought_coherence,
        memory_coherence,
        confidence_change,
        components_t,
        components_t1
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
    
    def _extract_components(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae y normaliza componentes del estado de forma robusta."""

        # Función de ayuda para convertir cualquier valor a una cadena de texto segura
        def _to_string(value: Any) -> str:
            if isinstance(value, str):
                return value
            # Si el modelo generó un objeto o lista, lo convertimos a su representación JSON
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False)
                # Como último recurso, usamos la conversión a string estándar
            return str(value)

        # Manejar diferentes formatos de estado
        if 'G_t' in state:
             # Formato completo SCt
             goal = _to_string(state.get('G_t', {}).get('primary_goal', ''))
             emotion = _to_string(state.get('S_t', {}).get('emotional_state', ''))
             confidence = state.get('S_t', {}).get('confidence_level', 0.5)
             thought_raw = state.get('A_t', [''])
             thought = _to_string(thought_raw[0] if thought_raw else '')
             memory = state.get('M_t', [])
        else:
             # Formato simplificado (el que genera nuestro modelo)
             goal = _to_string(state.get('goal', ''))
             emotion = _to_string(state.get('emotion', ''))
             confidence = state.get('confidence', 0.5)
             thought = _to_string(state.get('thought', ''))
             memory = state.get('memory', [])

         # Crear representación textual completa del estado
        full_state = f"Goal: {goal}. Emotion: {emotion}. Confidence: {confidence:.2f}. Thought: {thought}"

        return {
         'goal': goal,
         'emotion': emotion,
         'confidence': confidence,
         'thought': thought,
         'memory': memory,
         'full_state': full_state
         }
    
    def _get_embeddings(self, components: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Obtiene embeddings para cada componente"""
        
        embeddings = {}
        
        # Goal embedding
        logger.debug(f"Obteniendo embedding para GOAL: '{components['goal']}' (tipo: {type(components['goal']).__name__})")
        goal_key = f"goal_{components['goal']}"
        if goal_key in self.embedding_cache:
            logger.debug(f"Obteniendo embedding para GOAL: '{components['goal']}' (tipo: {type(components['goal']).__name__})")
            embeddings['goal'] = self.embedding_cache[goal_key]
        else:
            embeddings['goal'] = self.embedding_model.encode(components['goal'])
            self._update_cache(goal_key, embeddings['goal'])
        
        # Thought embedding
        
        thought_key = f"thought_{hash(components['thought'])}"
        if thought_key in self.embedding_cache:
            logger.debug(f"Obteniendo embedding para THOUGHT: '{components['thought']}' (tipo: {type(components['thought']).__name__})")
            embeddings['thought'] = self.embedding_cache[thought_key]
        else:
            embeddings['thought'] = self.embedding_model.encode(components['thought'])
            self._update_cache(thought_key, embeddings['thought'])
        
        # Full state embedding
        embeddings['full_state'] = self.embedding_model.encode(components['full_state'])
        
        # Memory embeddings (if any)
        if components['memory']:
            memory_texts = [str(m) for m in components['memory']]
            logger.debug(f"Obteniendo embedding para MEMORY: {memory_texts}")
            embeddings['memory'] = self.embedding_model.encode(memory_texts)
        else:
            embeddings['memory'] = np.array([])
        
        return embeddings
    
    def _calculate_semantic_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray
    ) -> float:
        """Calcula similitud coseno entre embeddings"""
        
        similarity = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0, 0]
        
        # Normalizar a rango [0, 1]
        return (similarity + 1) / 2
    
    def _calculate_transition_coherence(
        self,
        emotion1: str,
        emotion2: str,
        thought_emb1: np.ndarray,
        thought_emb2: np.ndarray
    ) -> float:
        """Calcula coherencia de transición emocional considerando contexto"""
        
        # Similitud entre pensamientos
        thought_similarity = self._calculate_semantic_similarity(thought_emb1, thought_emb2)
        
        # Reglas de transición emocional aprendidas
        valid_transitions = {
            'curious': ['analytical', 'excited', 'focused', 'uncertain'],
            'analytical': ['confident', 'focused', 'contemplative', 'curious'],
            'uncertain': ['curious', 'anxious', 'cautious', 'questioning'],
            'confident': ['satisfied', 'determined', 'assured', 'expansive'],
            'reflective': ['contemplative', 'peaceful', 'introspective', 'thoughtful']
        }
        
        # Misma emoción es coherente
        if emotion1 == emotion2:
            base_coherence = 0.9
        # Transición válida
        elif emotion2 in valid_transitions.get(emotion1, []):
            base_coherence = 0.8
        # Transición contextual (basada en similitud de pensamiento)
        elif thought_similarity > 0.7:
            base_coherence = 0.7
        else:
            base_coherence = 0.4
        
        # Ajustar por contexto del pensamiento
        return base_coherence * 0.7 + thought_similarity * 0.3
    
    def _calculate_thought_progression(
        self,
        thought_emb1: np.ndarray,
        thought_emb2: np.ndarray,
        state_emb1: np.ndarray,
        state_emb2: np.ndarray
    ) -> float:
        """Evalúa la progresión del pensamiento usando embeddings"""
        
        # Similitud directa entre pensamientos
        thought_similarity = self._calculate_semantic_similarity(thought_emb1, thought_emb2)
        
        # Similitud entre estados completos
        state_similarity = self._calculate_semantic_similarity(state_emb1, state_emb2)
        
        # La progresión ideal mantiene conexión pero no es idéntica
        # Penalizar tanto muy baja similitud como muy alta (sin progresión)
        if thought_similarity < 0.3:  # Muy diferente
            progression_score = thought_similarity * 2
        elif thought_similarity > 0.9:  # Muy similar (sin progresión)
            progression_score = 0.9 - (thought_similarity - 0.9) * 2
        else:  # Rango ideal
            progression_score = 0.6 + thought_similarity * 0.4
        
        # Combinar con coherencia del estado completo
        return progression_score * 0.6 + state_similarity * 0.4
    
    def _calculate_memory_coherence(
        self,
        memory1: List[Any],
        memory2: List[Any],
        thought_emb: np.ndarray
    ) -> float:
        """Evalúa coherencia de memoria usando embeddings"""
        
        if not memory1 and not memory2:
            return 1.0
        
        # Retención de memorias importantes
        retained = set(str(m) for m in memory1) & set(str(m) for m in memory2)
        retention_score = len(retained) / max(len(memory1), 1)
        
        # Relevancia de nuevas memorias respecto al pensamiento actual
        new_memories = set(str(m) for m in memory2) - set(str(m) for m in memory1)
        
        if new_memories and thought_emb.size > 0:
            # Calcular embeddings de nuevas memorias
            new_mem_embeddings = self.embedding_model.encode(list(new_memories))
            
            # Similitud con el pensamiento actual
            relevance_scores = []
            for mem_emb in new_mem_embeddings:
                relevance = self._calculate_semantic_similarity(mem_emb, thought_emb)
                relevance_scores.append(relevance)
            
            relevance_score = np.mean(relevance_scores) if relevance_scores else 0.5
        else:
            relevance_score = 1.0
        
        return retention_score * 0.5 + relevance_score * 0.5
    
    def _evaluate_confidence_progression(
        self,
        confidence_change: float,
        emotion_coherence: float,
        thought_coherence: float
    ) -> float:
        """Evalúa si el cambio de confianza es coherente"""
        
        # Cambios pequeños son generalmente coherentes
        if abs(confidence_change) < 0.1:
            base_score = 0.9
        elif abs(confidence_change) < 0.2:
            base_score = 0.7
        elif abs(confidence_change) < 0.3:
            base_score = 0.5
        else:
            base_score = 0.3
        
        # Ajustar por coherencia en otros componentes
        # Si otros componentes son coherentes, cambios mayores son aceptables
        context_factor = (emotion_coherence + thought_coherence) / 2
        
        return base_score * 0.6 + context_factor * 0.4
    
    def _calculate_global_coherence(
        self,
        goal_coherence: float,
        emotion_coherence: float,
        thought_coherence: float,
        memory_coherence: float,
        confidence_coherence: float
    ) -> float:
        """Calcula coherencia global con pesos aprendidos"""
        
        # Pesos optimizados (podrían venir de un modelo entrenado)
        weights = {
            'goal': 0.20,
            'emotion': 0.15,
            'thought': 0.35,  # Mayor peso al pensamiento
            'memory': 0.15,
            'confidence': 0.15
        }
        
        global_coherence = (
            weights['goal'] * goal_coherence +
            weights['emotion'] * emotion_coherence +
            weights['thought'] * thought_coherence +
            weights['memory'] * memory_coherence +
            weights['confidence'] * confidence_coherence
        )
        
        return global_coherence
    
    def _determine_verdict(self, global_coherence: float) -> CoherenceVerdict:
        """Determina veredicto basado en coherencia global"""
        
        if global_coherence >= self.coherence_thresholds['high']:
            return CoherenceVerdict.COHERENT
        elif global_coherence >= self.coherence_thresholds['low']:
            return CoherenceVerdict.AMBIGUOUS
        else:
            return CoherenceVerdict.INCOHERENT
    
    def _generate_justification(
        self,
        verdict: CoherenceVerdict,
        goal_coherence: float,
        emotion_coherence: float,
        thought_coherence: float,
        memory_coherence: float,
        confidence_change: float,
        components_t: Dict[str, Any],
        components_t1: Dict[str, Any]
    ) -> str:
        """Genera justificación basada en análisis de embeddings"""
        
        components = []
        
        # Analizar cada dimensión
        if goal_coherence >= 0.8:
            if components_t['goal'] == components_t1['goal']:
                components.append("La meta se mantiene consistente")
            else:
                components.append("La evolución de la meta es semánticamente coherente")
        else:
            components.append("Cambio de meta sin conexión semántica clara")
        
        if emotion_coherence >= 0.7:
            components.append(
                f"la transición emocional de '{components_t['emotion']}' a "
                f"'{components_t1['emotion']}' es contextualmente apropiada"
            )
        else:
            components.append("transición emocional abrupta")
        
        if thought_coherence >= 0.7:
            components.append("el pensamiento evoluciona manteniendo coherencia semántica")
        elif thought_coherence >= 0.5:
            components.append("el pensamiento muestra progresión con conexión parcial")
        else:
            components.append("ruptura en la continuidad del pensamiento")
        
        if abs(confidence_change) < 0.2:
            components.append("cambio de confianza dentro de rangos normales")
        else:
            components.append("variación significativa en el nivel de confianza")
        
        # Construir justificación final
        if verdict == CoherenceVerdict.COHERENT:
            return f"Transición coherente: {', '.join(components)}."
        elif verdict == CoherenceVerdict.INCOHERENT:
            problems = [c for c in components if any(word in c for word in 
                       ['sin conexión', 'abrupta', 'ruptura', 'significativa'])]
            return f"Transición incoherente: {', '.join(problems)}."
        else:
            return f"Transición ambigua: {components[0]}, pero {' y '.join(components[1:])}."
    
    def _update_cache(self, key: str, embedding: np.ndarray):
        """Actualiza cache de embeddings con límite de tamaño"""
        
        self.embedding_cache[key] = embedding
        
        # Limpiar cache si excede el límite
        if len(self.embedding_cache) > self.max_cache_size:
            # Eliminar 10% más antiguos (aproximación FIFO)
            keys_to_remove = list(self.embedding_cache.keys())[:int(self.max_cache_size * 0.1)]
            for key in keys_to_remove:
                del self.embedding_cache[key]


    def _generate_ml_based_justification(
        self,
        verdict: CoherenceVerdict,
        ml_confidence: float,
        goal_coherence: float,
        emotion_coherence: float,
        thought_coherence: float,
        components_t: Dict[str, Any],
        components_t1: Dict[str, Any]
    ) -> str:
        """
        Genera justificación cuando se usa el clasificador ML
        """
        
        confidence_desc = "alta certeza" if ml_confidence > 0.9 else "confianza moderada"
        
        components = []
        
        # Describir el uso del ML
        components.append(f"Evaluación ML con {confidence_desc}")
        
        # Añadir detalles basados en métricas
        if goal_coherence >= 0.8:
            components.append("coherencia semántica en metas confirmada")
        
        if thought_coherence >= 0.7:
            components.append("progresión de pensamiento validada")
        
        # Construir justificación
        if verdict == CoherenceVerdict.COHERENT:
            return f"Transición coherente ({components[0]}): {', '.join(components[1:])}."
        elif verdict == CoherenceVerdict.INCOHERENT:
            return f"Transición incoherente detectada por modelo ML: "
            f"cambio abrupto de '{components_t['emotion']}' a '{components_t1['emotion']}'."
        else:
            return f"Transición ambigua según modelo ML: requiere análisis adicional."

