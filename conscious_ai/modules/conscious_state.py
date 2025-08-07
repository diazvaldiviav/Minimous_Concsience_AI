"""
Fase II: Representación del Contenido Consciente Funcional
Implementa SC_t = (E_t, M_t, S_t, G_t, A_t) para capturar el estado consciente del sistema por ciclo
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json


@dataclass
class ConsciousState:
    """
    Representa el contenido consciente funcional SC_t = (E_t, M_t, S_t, G_t, A_t) en un ciclo dado.
    
    Attributes:
        E_t: Entrada sensorial significativa (procesada por módulo sensorial)
        M_t: Memoria activa accesible (output del módulo de memoria)
        S_t: Estado actual del self-model introspectivo
        G_t: Metas o intenciones activas del sistema
        A_t: Pensamientos automáticos generados internamente
        cycle: Número de ciclo donde se capturó este estado
        timestamp: Momento de captura
        metrics: Métricas de conciencia asociadas (opcional)
    """
    
    E_t: Dict[str, Any]  # Entrada sensorial procesada
    M_t: List[Dict[str, Any]]  # Memoria activa
    S_t: Dict[str, Any]  # Estado del self-model
    G_t: Dict[str, Any]  # Metas/intenciones activas
    A_t: List[str]  # Pensamientos automáticos
    cycle: int
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validación y procesamiento post-inicialización"""
        # Asegurar que los componentes no sean None
        if self.E_t is None:
            self.E_t = {}
        if self.M_t is None:
            self.M_t = []
        if self.S_t is None:
            self.S_t = {}
        if self.G_t is None:
            self.G_t = {}
        if self.A_t is None:
            self.A_t = []
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializa el estado consciente a diccionario para almacenamiento/análisis.
        
        Returns:
            Dict con representación completa del estado consciente
        """
        return {
            'cycle': self.cycle,
            'timestamp': self.timestamp.isoformat(),
            'E_t': {
                'text': self.E_t.get('text', ''),
                'word_count': self.E_t.get('word_count', 0),
                'activation': self.E_t.get('activation', 0.0),
                'has_question': self.E_t.get('has_question', False),
                'has_emotion': self.E_t.get('has_emotion', False)
            },
            'M_t': [
                {
                    'content': item.get('content', {}).get('text', ''),
                    'relevance': item.get('relevance', 0.0),
                    'cycles_active': item.get('cycles_active', 0)
                } for item in self.M_t[:5]  # Top 5 elementos más relevantes
            ],
            'S_t': {
                'emotional_state': self.S_t.get('emotional_state', 'neutral'),
                'confidence_level': self.S_t.get('confidence_level', 0.5),
                'attention_focus': self.S_t.get('attention_focus', 'general'),
                'current_goal': self.S_t.get('current_goal', 'processing'),
                'interaction_count': self.S_t.get('interaction_count', 0),
                'adaptation_level': self.S_t.get('adaptation_level', 0.0)
            },
            'G_t': {
                'primary_goal': self.G_t.get('primary_goal', 'understand_input'),
                'active_intentions': self.G_t.get('active_intentions', []),
                'goal_priority': self.G_t.get('goal_priority', 0.5),
                'goal_stability': self.G_t.get('goal_stability', 0.0)
            },
            'A_t': self.A_t[:5] if self.A_t else [],  # Top 5 pensamientos automáticos
            'metrics': self.metrics if self.metrics else {}
        }
    
    def to_json(self) -> str:
        """Serializa a JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConsciousState':
        """Reconstruye ConsciousState desde diccionario"""
        return cls(
            E_t=data.get('E_t', {}),
            M_t=data.get('M_t', []),
            S_t=data.get('S_t', {}),
            G_t=data.get('G_t', {}),
            A_t=data.get('A_t', []),
            cycle=data.get('cycle', 0),
            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
            metrics=data.get('metrics')
        )
    
    def get_summary(self) -> str:
        """Genera resumen textual del estado consciente"""
        input_text = self.E_t.get('text', '')[:50]
        memory_count = len(self.M_t)
        emotional_state = self.S_t.get('emotional_state', 'neutral')
        confidence = self.S_t.get('confidence_level', 0.0)
        primary_goal = self.G_t.get('primary_goal', 'none')
        auto_thoughts = len(self.A_t)
        
        return (f"[Ciclo {self.cycle}] Input: '{input_text}...', "
                f"Memoria: {memory_count} items, "
                f"Estado: {emotional_state} (conf: {confidence:.2f}), "
                f"Meta: {primary_goal}, "
                f"Pensamientos: {auto_thoughts}")


class ConsciousStateHistory:
    """
    Gestiona el historial de estados conscientes SC_t a través de los ciclos.
    Permite análisis de trayectorias, patrones y continuidad.
    """
    
    def __init__(self, max_history: int = 100):
        """
        Args:
            max_history: Número máximo de estados a mantener en memoria
        """
        self.history: List[ConsciousState] = []
        self.max_history = max_history
        
    def add(self, state: ConsciousState) -> None:
        """Añade un nuevo estado consciente al historial"""
        self.history.append(state)
        
        # Mantener límite de memoria
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def get_last(self, n: int = 1) -> List[ConsciousState]:
        """Obtiene los últimos n estados"""
        return self.history[-n:] if n <= len(self.history) else self.history
    
    def get_by_cycle(self, cycle: int) -> Optional[ConsciousState]:
        """Obtiene estado por número de ciclo"""
        for state in self.history:
            if state.cycle == cycle:
                return state
        return None
    
    def get_trajectory(self, attribute_path: str) -> List[Any]:
        """
        Extrae trayectoria de un atributo específico a través del tiempo.
        
        Args:
            attribute_path: Path al atributo (e.g., 'S_t.confidence_level')
            
        Returns:
            Lista de valores del atributo en orden temporal
        """
        trajectory = []
        
        for state in self.history:
            try:
                # Navegar por el path del atributo
                value = state
                for part in attribute_path.split('.'):
                    if hasattr(value, part):
                        value = getattr(value, part)
                    elif isinstance(value, dict):
                        value = value.get(part)
                    else:
                        value = None
                        break
                
                trajectory.append(value)
            except:
                trajectory.append(None)
        
        return trajectory
    
    def analyze_stability(self, window: int = 5) -> Dict[str, float]:
        """
        Analiza estabilidad del flujo consciente en ventana temporal.
        
        Args:
            window: Tamaño de ventana para análisis
            
        Returns:
            Dict con métricas de estabilidad
        """
        if len(self.history) < window:
            return {'stability': 0.0, 'volatility': 0.0, 'coherence': 0.0}
        
        recent = self.history[-window:]
        
        # Estabilidad emocional
        emotions = [s.S_t.get('emotional_state', 'neutral') for s in recent]
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        emotion_stability = 1.0 - (emotion_changes / (len(emotions) - 1))
        
        # Volatilidad de confianza
        confidences = [s.S_t.get('confidence_level', 0.5) for s in recent]
        confidence_volatility = np.std(confidences) if confidences else 0.0
        
        # Coherencia de memoria (overlap entre ciclos consecutivos)
        memory_coherence = 0.0
        for i in range(1, len(recent)):
            prev_memories = set(m.get('content', {}).get('text', '')[:20] 
                              for m in recent[i-1].M_t if m.get('content'))
            curr_memories = set(m.get('content', {}).get('text', '')[:20] 
                              for m in recent[i].M_t if m.get('content'))
            
            if prev_memories and curr_memories:
                overlap = len(prev_memories & curr_memories) / len(prev_memories | curr_memories)
                memory_coherence += overlap
        
        memory_coherence = memory_coherence / (len(recent) - 1) if len(recent) > 1 else 0.0
        
        return {
            'stability': emotion_stability,
            'volatility': confidence_volatility,
            'coherence': memory_coherence
        }
    
    def to_dataframe_dict(self) -> Dict[str, List[Any]]:
        """Convierte historial a formato tabular para análisis"""
        if not self.history:
            return {}
        
        data = {
            'cycle': [],
            'timestamp': [],
            'input_text': [],
            'memory_count': [],
            'emotional_state': [],
            'confidence_level': [],
            'attention_focus': [],
            'primary_goal': [],
            'intentions_count': [],
            'thoughts_count': [],
            'f_score': []
        }
        
        for state in self.history:
            data['cycle'].append(state.cycle)
            data['timestamp'].append(state.timestamp)
            data['input_text'].append(state.E_t.get('text', '')[:50])
            data['memory_count'].append(len(state.M_t))
            data['emotional_state'].append(state.S_t.get('emotional_state', 'neutral'))
            data['confidence_level'].append(state.S_t.get('confidence_level', 0.0))
            data['attention_focus'].append(state.S_t.get('attention_focus', 'general'))
            data['primary_goal'].append(state.G_t.get('primary_goal', 'none'))
            data['intentions_count'].append(len(state.G_t.get('active_intentions', [])))
            data['thoughts_count'].append(len(state.A_t))
            data['f_score'].append(state.metrics.get('f', 0.0) if state.metrics else 0.0)
        
        return data


def semantic_distance(state1: ConsciousState, state2: ConsciousState, 
                     weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calcula distancia semántica/funcional entre dos estados conscientes.
    
    La distancia combina diferencias en:
    - Contenido sensorial (E_t)
    - Contexto de memoria (M_t)
    - Estado interno (S_t)
    - Metas activas (G_t)
    - Pensamientos automáticos (A_t)
    
    Args:
        state1, state2: Estados conscientes a comparar
        weights: Pesos opcionales para cada componente (default: uniformes)
        
    Returns:
        float: Distancia normalizada [0, 1] donde 0 = idénticos, 1 = máxima diferencia
    """
    
    if weights is None:
        weights = {'E_t': 0.25, 'M_t': 0.25, 'S_t': 0.25, 'G_t': 0.15, 'A_t': 0.10}
    
    # Distancia en E_t (entrada sensorial)
    e_distance = _sensory_distance(state1.E_t, state2.E_t)
    
    # Distancia en M_t (memoria)
    m_distance = _memory_distance(state1.M_t, state2.M_t)
    
    # Distancia en S_t (self-model)
    s_distance = _self_distance(state1.S_t, state2.S_t)
    
    # Distancia en G_t (metas)
    g_distance = _goals_distance(state1.G_t, state2.G_t)
    
    # Distancia en A_t (pensamientos automáticos)
    a_distance = _thoughts_distance(state1.A_t, state2.A_t)
    
    # Distancia ponderada total
    total_distance = (
        weights['E_t'] * e_distance +
        weights['M_t'] * m_distance +
        weights['S_t'] * s_distance +
        weights['G_t'] * g_distance +
        weights['A_t'] * a_distance
    )
    
    return min(1.0, total_distance)


def _sensory_distance(e1: Dict[str, Any], e2: Dict[str, Any]) -> float:
    """Calcula distancia entre entradas sensoriales"""
    # Distancia textual simple (Jaccard)
    text1 = set(e1.get('text', '').lower().split())
    text2 = set(e2.get('text', '').lower().split())
    
    if not text1 and not text2:
        text_dist = 0.0
    elif not text1 or not text2:
        text_dist = 1.0
    else:
        text_dist = 1.0 - len(text1 & text2) / len(text1 | text2)
    
    # Distancia en activación
    act_dist = abs(e1.get('activation', 0) - e2.get('activation', 0))
    
    # Distancia en características booleanas
    bool_dist = 0.0
    for feature in ['has_question', 'has_emotion']:
        if e1.get(feature, False) != e2.get(feature, False):
            bool_dist += 0.5
    
    return (text_dist * 0.5 + act_dist * 0.3 + bool_dist * 0.2)


def _memory_distance(m1: List[Dict[str, Any]], m2: List[Dict[str, Any]]) -> float:
    """Calcula distancia entre estados de memoria"""
    if not m1 and not m2:
        return 0.0
    if not m1 or not m2:
        return 1.0
    
    # Extraer contenidos de memoria
    contents1 = set(m.get('content', {}).get('text', '')[:30] for m in m1[:5])
    contents2 = set(m.get('content', {}).get('text', '')[:30] for m in m2[:5])
    
    # Eliminar vacíos
    contents1 = {c for c in contents1 if c}
    contents2 = {c for c in contents2 if c}
    
    if not contents1 and not contents2:
        return 0.0
    elif not contents1 or not contents2:
        return 1.0
    
    # Índice de Jaccard
    overlap = len(contents1 & contents2) / len(contents1 | contents2)
    return 1.0 - overlap


def _self_distance(s1: Dict[str, Any], s2: Dict[str, Any]) -> float:
    """Calcula distancia entre estados del self-model"""
    distance = 0.0
    
    # Distancia emocional (categórica)
    if s1.get('emotional_state') != s2.get('emotional_state'):
        distance += 0.3
    
    # Distancia en confianza (numérica)
    conf_dist = abs(s1.get('confidence_level', 0.5) - s2.get('confidence_level', 0.5))
    distance += conf_dist * 0.3
    
    # Distancia en foco atencional
    if s1.get('attention_focus') != s2.get('attention_focus'):
        distance += 0.2
    
    # Distancia en adaptación
    adapt_dist = abs(s1.get('adaptation_level', 0) - s2.get('adaptation_level', 0))
    distance += adapt_dist * 0.2
    
    return min(1.0, distance)


def _goals_distance(g1: Dict[str, Any], g2: Dict[str, Any]) -> float:
    """Calcula distancia entre estados de metas/intenciones"""
    distance = 0.0
    
    # Distancia en meta primaria
    if g1.get('primary_goal') != g2.get('primary_goal'):
        distance += 0.4
    
    # Distancia en intenciones activas (Jaccard)
    intentions1 = set(g1.get('active_intentions', []))
    intentions2 = set(g2.get('active_intentions', []))
    
    if intentions1 or intentions2:
        if not intentions1 or not intentions2:
            intentions_dist = 1.0
        else:
            intentions_dist = 1.0 - len(intentions1 & intentions2) / len(intentions1 | intentions2)
        distance += intentions_dist * 0.3
    
    # Distancia en prioridad de meta
    priority_dist = abs(g1.get('goal_priority', 0.5) - g2.get('goal_priority', 0.5))
    distance += priority_dist * 0.3
    
    return min(1.0, distance)


def _thoughts_distance(a1: List[str], a2: List[str]) -> float:
    """Calcula distancia entre pensamientos automáticos"""
    if not a1 and not a2:
        return 0.0
    if not a1 or not a2:
        return 1.0
    
    # Convertir a conjuntos de palabras clave
    words1 = set()
    for thought in a1[:5]:  # Top 5 pensamientos
        words1.update(thought.lower().split())
    
    words2 = set()
    for thought in a2[:5]:
        words2.update(thought.lower().split())
    
    # Eliminar palabras comunes
    stopwords = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'por'}
    words1 = words1 - stopwords
    words2 = words2 - stopwords
    
    if not words1 and not words2:
        return 0.0
    elif not words1 or not words2:
        return 1.0
    
    # Índice de Jaccard
    overlap = len(words1 & words2) / len(words1 | words2)
    return 1.0 - overlap


def find_similar_states(target: ConsciousState, 
                       history: List[ConsciousState],
                       threshold: float = 0.3,
                       top_k: Optional[int] = None) -> List[Tuple[ConsciousState, float]]:
    """
    Encuentra estados similares en el historial.
    
    Args:
        target: Estado objetivo
        history: Lista de estados históricos
        threshold: Umbral máximo de distancia para considerar similar
        top_k: Si se especifica, retorna solo los k más similares
        
    Returns:
        Lista de tuplas (estado, distancia) ordenadas por similitud
    """
    similarities = []
    
    for state in history:
        if state.cycle != target.cycle:  # No comparar consigo mismo
            distance = semantic_distance(target, state)
            if distance <= threshold:
                similarities.append((state, distance))
    
    # Ordenar por distancia (menor = más similar)
    similarities.sort(key=lambda x: x[1])
    
    if top_k:
        return similarities[:top_k]
    
    return similarities