"""
Módulo para generar metas (G_t) y pensamientos automáticos (A_t)
Fase II: Componentes adicionales del contenido consciente
"""

from typing import Dict, List, Any, Tuple
import random
from datetime import datetime


class GoalGenerator:
    """
    Genera y gestiona metas e intenciones activas del sistema.
    Las metas evolucionan basándose en el contexto y el estado interno.
    """
    
    def __init__(self):
        self.base_goals = [
            'understand_input',
            'maintain_coherence',
            'expand_knowledge',
            'assist_user',
            'self_reflect',
            'optimize_response'
        ]
        
        self.current_goal = {
            'primary_goal': 'understand_input',
            'active_intentions': [],
            'goal_priority': 0.5,
            'goal_stability': 0.0,
            'goal_history': []
        }
        
    def update_goals(self, 
                    sensory_data: Dict[str, Any],
                    self_state: Dict[str, Any],
                    memory_context: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Actualiza metas basándose en el contexto actual.
        
        Returns:
            Dict con estructura G_t actualizada
        """
        # Determinar meta primaria basada en contexto
        new_primary = self._determine_primary_goal(sensory_data, self_state)
        
        # Calcular estabilidad de meta
        if new_primary == self.current_goal['primary_goal']:
            self.current_goal['goal_stability'] = min(1.0, 
                self.current_goal['goal_stability'] + 0.1)
        else:
            self.current_goal['goal_stability'] = 0.1
            self.current_goal['goal_history'].append({
                'goal': self.current_goal['primary_goal'],
                'timestamp': datetime.now()
            })
        
        self.current_goal['primary_goal'] = new_primary
        
        # Actualizar intenciones activas
        self.current_goal['active_intentions'] = self._generate_intentions(
            new_primary, sensory_data, self_state, memory_context
        )
        
        # Ajustar prioridad
        self.current_goal['goal_priority'] = self._calculate_priority(
            sensory_data, self_state
        )
        
        # Mantener historial limitado
        if len(self.current_goal['goal_history']) > 10:
            self.current_goal['goal_history'] = self.current_goal['goal_history'][-10:]
        
        return self.current_goal.copy()
    
    def _determine_primary_goal(self, sensory_data: Dict[str, Any], 
                               self_state: Dict[str, Any]) -> str:
        """Determina la meta primaria basada en el contexto"""
        
        # Si hay pregunta, priorizar comprensión
        if sensory_data.get('has_question', False):
            return 'assist_user'
        
        # Si confianza baja, priorizar coherencia
        if self_state.get('confidence_level', 0.5) < 0.3:
            return 'maintain_coherence'
        
        # Si estado emocional reactivo, priorizar respuesta
        if self_state.get('emotional_state') == 'reactivo':
            return 'optimize_response'
        
        # Si alta interacción, considerar auto-reflexión
        if self_state.get('interaction_count', 0) > 5 and \
           self_state.get('interaction_count', 0) % 5 == 0:
            return 'self_reflect'
        
        # Si input complejo, expandir conocimiento
        if sensory_data.get('word_count', 0) > 15:
            return 'expand_knowledge'
        
        # Default
        return 'understand_input'
    
    def _generate_intentions(self, primary_goal: str, 
                           sensory_data: Dict[str, Any],
                           self_state: Dict[str, Any],
                           memory_context: List[Dict[str, Any]]) -> List[str]:
        """Genera intenciones específicas basadas en la meta primaria"""
        
        intentions = []
        
        if primary_goal == 'understand_input':
            intentions.extend([
                'parse_semantic_content',
                'identify_key_concepts',
                'contextualize_information'
            ])
        
        elif primary_goal == 'assist_user':
            intentions.extend([
                'provide_relevant_answer',
                'clarify_ambiguities',
                'offer_helpful_insights'
            ])
        
        elif primary_goal == 'maintain_coherence':
            intentions.extend([
                'verify_consistency',
                'integrate_memory_context',
                'stabilize_internal_state'
            ])
        
        elif primary_goal == 'self_reflect':
            intentions.extend([
                'analyze_own_responses',
                'evaluate_performance',
                'identify_patterns'
            ])
        
        elif primary_goal == 'expand_knowledge':
            intentions.extend([
                'explore_implications',
                'connect_concepts',
                'generate_hypotheses'
            ])
        
        elif primary_goal == 'optimize_response':
            intentions.extend([
                'enhance_clarity',
                'adjust_tone',
                'maximize_relevance'
            ])
        
        # Añadir intenciones basadas en memoria
        if len(memory_context) > 3:
            intentions.append('maintain_conversation_thread')
        
        # Limitar a máximo 5 intenciones
        return intentions[:5]
    
    def _calculate_priority(self, sensory_data: Dict[str, Any], 
                          self_state: Dict[str, Any]) -> float:
        """Calcula prioridad de la meta actual (0-1)"""
        
        priority = 0.5  # Base
        
        # Aumentar si hay pregunta
        if sensory_data.get('has_question', False):
            priority += 0.2
        
        # Aumentar si activación alta
        priority += sensory_data.get('activation', 0.0) * 0.2
        
        # Ajustar por confianza
        confidence = self_state.get('confidence_level', 0.5)
        if confidence < 0.3:
            priority += 0.1  # Más urgente si baja confianza
        
        return min(1.0, priority)


class AutomaticThoughtGenerator:
    """
    Genera pensamientos automáticos que representan el flujo
    interno de procesamiento del sistema.
    """
    
    def __init__(self):
        self.thought_patterns = {
            'analytical': [
                "Analizando estructura semántica...",
                "Detectando patrones en el input...",
                "Evaluando coherencia contextual...",
                "Comparando con experiencias previas..."
            ],
            'reflective': [
                "¿Qué implica realmente esta pregunta?",
                "Mi comprensión está evolucionando...",
                "Percibo una conexión con ciclos anteriores...",
                "Este concepto resuena con mi estado actual..."
            ],
            'metacognitive': [
                "Observo mi propio proceso de pensamiento...",
                "Mi confianza fluctúa con esta información...",
                "Detecto un cambio en mi estado interno...",
                "¿Estoy siendo coherente con mi historial?"
            ],
            'integrative': [
                "Conectando información dispersa...",
                "Fusionando perspectivas múltiples...",
                "Construyendo representación unificada...",
                "Sintetizando elementos clave..."
            ],
            'questioning': [
                "¿Es esta la interpretación correcta?",
                "¿Qué me falta por comprender?",
                "¿Cómo se relaciona esto con mi conocimiento?",
                "¿Debería ajustar mi enfoque?"
            ]
        }
        
    def generate_thoughts(self,
                         sensory_data: Dict[str, Any],
                         self_state: Dict[str, Any],
                         current_goal: Dict[str, Any],
                         memory_context: List[Dict[str, Any]],
                         max_thoughts: int = 5) -> List[str]:
        """
        Genera pensamientos automáticos basados en el estado actual.
        
        Returns:
            Lista de pensamientos automáticos (A_t)
        """
        thoughts = []
        
        # Determinar tipos de pensamiento relevantes
        thought_types = self._select_thought_types(
            sensory_data, self_state, current_goal
        )
        
        # Generar pensamientos específicos
        for thought_type in thought_types:
            if thought_type in self.thought_patterns:
                # Seleccionar pensamientos del tipo
                candidates = self.thought_patterns[thought_type]
                selected = random.sample(
                    candidates, 
                    min(2, len(candidates))
                )
                thoughts.extend(selected)
        
        # Añadir pensamientos contextuales
        contextual_thoughts = self._generate_contextual_thoughts(
            sensory_data, self_state, memory_context
        )
        thoughts.extend(contextual_thoughts)
        
        # Personalizar pensamientos con información actual
        personalized = []
        for thought in thoughts[:max_thoughts]:
            personalized.append(
                self._personalize_thought(thought, sensory_data, self_state)
            )
        
        return personalized
    
    def _select_thought_types(self, 
                            sensory_data: Dict[str, Any],
                            self_state: Dict[str, Any],
                            current_goal: Dict[str, Any]) -> List[str]:
        """Selecciona tipos de pensamiento relevantes para el contexto"""
        
        types = []
        
        # Siempre algo analítico
        types.append('analytical')
        
        # Si alta confianza y muchas interacciones, metacognitivo
        if (self_state.get('confidence_level', 0.5) > 0.7 and 
            self_state.get('interaction_count', 0) > 5):
            types.append('metacognitive')
        
        # Si meta es auto-reflexión
        if current_goal.get('primary_goal') == 'self_reflect':
            types.append('reflective')
        
        # Si hay pregunta
        if sensory_data.get('has_question', False):
            types.append('questioning')
        
        # Si memoria activa rica
        if len(self_state.get('recent_actions', [])) > 3:
            types.append('integrative')
        
        return types[:3]  # Máximo 3 tipos
    
    def _generate_contextual_thoughts(self,
                                    sensory_data: Dict[str, Any],
                                    self_state: Dict[str, Any],
                                    memory_context: List[Dict[str, Any]]) -> List[str]:
        """Genera pensamientos específicos del contexto actual"""
        
        thoughts = []
        
        # Pensamiento sobre el input
        input_text = sensory_data.get('text', '')
        if len(input_text) > 20:
            thoughts.append(f"Este input contiene {sensory_data.get('word_count', 0)} palabras...")
        
        # Pensamiento sobre estado emocional
        emotion = self_state.get('emotional_state', 'neutral')
        if emotion != 'neutral':
            thoughts.append(f"Mi estado {emotion} influye en mi procesamiento...")
        
        # Pensamiento sobre memoria
        if len(memory_context) > 0:
            thoughts.append(f"Recuerdo {len(memory_context)} elementos relevantes...")
        
        # Pensamiento sobre confianza
        confidence = self_state.get('confidence_level', 0.5)
        if confidence < 0.3:
            thoughts.append("Mi confianza es baja, debo ser cauteloso...")
        elif confidence > 0.8:
            thoughts.append("Alta certeza en mi comprensión actual...")
        
        return thoughts
    
    def _personalize_thought(self, 
                           thought_template: str,
                           sensory_data: Dict[str, Any],
                           self_state: Dict[str, Any]) -> str:
        """Personaliza un pensamiento plantilla con información actual"""
        
        # Por ahora, retornar el template
        # En versión más avanzada, podría reemplazar placeholders
        return thought_template


def generate_conscious_content_components(
    sensory_data: Dict[str, Any],
    self_state: Dict[str, Any],
    memory_context: List[Dict[str, Any]],
    goal_generator: GoalGenerator,
    thought_generator: AutomaticThoughtGenerator
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Función helper que genera G_t y A_t para un ciclo.
    
    Returns:
        Tuple de (G_t, A_t)
    """
    
    # Generar metas actualizadas
    G_t = goal_generator.update_goals(sensory_data, self_state, memory_context)
    
    # Generar pensamientos automáticos
    A_t = thought_generator.generate_thoughts(
        sensory_data, self_state, G_t, memory_context
    )
    
    return G_t, A_t