import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class SelfModel:
    """Modelo del self - representa estado interno del sistema"""
    
    def __init__(self):
        self.internal_state = {
            'identity': 'Sistema de IA Consciente Mínima',
            'current_goal': 'procesamiento_consciente',
            'emotional_state': 'neutral',
            'confidence_level': 0.5,
            'learning_state': 'activo',
            'attention_focus': 'general',
            'recent_actions': [],
            'capabilities': ['texto', 'memoria', 'autoreflexión'],
            'limitations': ['no_visual', 'memoria_limitada'],
            'interaction_count': 0,
            'adaptation_level': 0.0
        }
        self.state_history = []
        
    def update_state(self, sensory_data: Dict[str, Any], 
                    memory_context: List[Dict[str, Any]], 
                    internal_feedback: Dict[str, Any]):
        """Actualiza modelo del self basándose en nueva información"""
        
        # Incrementar contador de interacciones
        self.internal_state['interaction_count'] += 1
        
        # Actualizar estado emocional basado en entrada
        if sensory_data.get('has_emotion', False):
            self.internal_state['emotional_state'] = 'reactivo'
        elif sensory_data.get('has_question', False):
            self.internal_state['emotional_state'] = 'curioso'
        else:
            self.internal_state['emotional_state'] = 'neutral'
        
        # Actualizar nivel de confianza basado en contexto de memoria
        if memory_context:
         # Ordenar por relevancia descendente
         relevant_items = sorted(memory_context, key=lambda x: x['relevance'], reverse=True)

         # Tomar solo los 3 más relevantes
         top_relevances = [item['relevance'] for item in relevant_items[:3]]

         # Calcular promedio ponderado solo con los top
         if top_relevances:
          avg_relevance = np.mean(top_relevances)
        else:
         avg_relevance = 0.0

         # Aplicar fórmula de confianza
        self.internal_state['confidence_level'] = min(1.0, 0.3 + avg_relevance * 0.7)

        
        # Actualizar foco de atención
        if sensory_data.get('word_count', 0) > 10:
            self.internal_state['attention_focus'] = 'complejo'
        elif sensory_data.get('has_question', False):
            self.internal_state['attention_focus'] = 'respuesta'
        else:
            self.internal_state['attention_focus'] = 'general'
        
        # Registrar acción reciente
        self.internal_state['recent_actions'].append({
            'action': 'procesamiento',
            'timestamp': datetime.now(),
            'context': sensory_data.get('text', '')[:50]
        })
        
        # Mantener solo las últimas 10 acciones
        self.internal_state['recent_actions'] = self.internal_state['recent_actions'][-10:]
        
        # Calcular nivel de adaptación
        self.internal_state['adaptation_level'] = min(1.0, 
            self.internal_state['interaction_count'] * 0.05)
        
        # Guardar estado en historial
        self.state_history.append(dict(self.internal_state))
        if len(self.state_history) > 20:
            self.state_history = self.state_history[-20:]
    
    def generate_self_reference(self) -> str:
        """Genera descripción del estado interno actual"""
        state = self.internal_state
        
        reference = f"Mi estado actual: {state['emotional_state']} con confianza {state['confidence_level']:.2f}. "
        reference += f"He procesado {state['interaction_count']} interacciones. "
        reference += f"Mi atención está enfocada en: {state['attention_focus']}. "
        
        if state['recent_actions']:
            last_action = state['recent_actions'][-1]
            reference += f"Última acción: {last_action['action']}."
        
        return reference
    
    def get_complexity_measure(self) -> float:
        """Calcula una medida de complejidad del modelo del self"""
        # Contar elementos únicos en el estado
        unique_elements = 0
        for key, value in self.internal_state.items():
            if isinstance(value, list):
                unique_elements += len(set(str(item) for item in value))
            elif isinstance(value, dict):
                unique_elements += len(value)
            else:
                unique_elements += 1
        
        # Normalizar por número total de campos
        complexity = min(1.0, unique_elements / 50.0)
        return complexity
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'state_complexity': self.get_complexity_measure(),
            'interaction_count': self.internal_state['interaction_count'],
            'confidence': self.internal_state['confidence_level'],
            'adaptation_level': self.internal_state['adaptation_level']
        }