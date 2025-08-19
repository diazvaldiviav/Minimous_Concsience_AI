import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class CentralIntegrator:
    """Integrador central - fusiona información de todos los módulos"""
    
    def __init__(self):
        self.integration_history = []
        self.decision_threshold = 0.6
        
    def integrate_information(self, sensory_data: Dict[str, Any],
                            memory_data: List[Dict[str, Any]],
                            self_model_data: Dict[str, Any],
                            feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Integra información de todos los módulos"""
        
        # Calcular pesos de integración
        sensory_weight = sensory_data.get('activation', 0.5)
        memory_weight = min(1.0, len(memory_data) * 0.2) if memory_data else 0.1
        self_weight = self_model_data.get('confidence_level', 0.5)
        feedback_weight = min(1.0, len(feedback_data) * 0.1) if feedback_data else 0.1
        
        # Normalizar pesos
        total_weight = sensory_weight + memory_weight + self_weight + feedback_weight
        if total_weight > 0:
            sensory_weight /= total_weight
            memory_weight /= total_weight
            self_weight /= total_weight
            feedback_weight /= total_weight
        
        # Crear representación integrada
        integrated_info = {
            'primary_input': sensory_data.get('text', ''),
            'contextual_memory': memory_data[:3] if memory_data else [],  # Top 3
            'self_awareness': self_model_data.get('confidence_level', 0),
            'internal_feedback': feedback_data,
            'integration_weights': {
                'sensory': sensory_weight,
                'memory': memory_weight,
                'self': self_weight,
                'feedback': feedback_weight
            },
            'integration_strength': total_weight,
            'timestamp': datetime.now()
        }
        
        # Determinar tipo de respuesta basándose en integración
        response_type = self._determine_response_type(integrated_info)
        integrated_info['response_type'] = response_type
        
        # Guardar en historial
        self.integration_history.append(integrated_info)
        if len(self.integration_history) > 15:
            self.integration_history = self.integration_history[-15:]
        
        return integrated_info
    
    def _determine_response_type(self, integrated_info: Dict[str, Any]) -> str:
        """Determina el tipo de respuesta basándose en la información integrada"""
        
        weights = integrated_info['integration_weights']
        
        if weights['self'] > 0.4:
            return 'autorreflexiva'
        elif weights['memory'] > 0.3:
            return 'contextual'
        elif weights['feedback'] > 0.2:
            return 'iterativa'
        else:
            return 'directa'
    
    def generate_response(self, integrated_info: Dict[str, Any]) -> str:
        """Genera respuesta basándose en información integrada"""
        
        response_type = integrated_info['response_type']
        primary_input = integrated_info['primary_input']
        
        # Base de la respuesta
        if response_type == 'autorreflexiva':
            response = f"Reflexionando sobre '{primary_input}': "
            response += "Analizo esta entrada desde mi perspectiva interna. "
        elif response_type == 'contextual':
            response = f"Considerando '{primary_input}' en contexto: "
            if integrated_info['contextual_memory']:
                response += "Esto se relaciona con experiencias previas. "
        elif response_type == 'iterativa':
            response = f"Procesando '{primary_input}' con retroalimentación interna: "
            response += "Mis sistemas internos están intercambiando información. "
        else:
            response = f"Procesando '{primary_input}': "
        
        # Añadir información de integración
        strength = integrated_info['integration_strength']
        response += f"Nivel de integración: {strength:.2f}. "
        
        # Añadir contexto de memoria si existe
        if integrated_info['contextual_memory']:
            response += f"Tengo {len(integrated_info['contextual_memory'])} elementos relevantes en memoria. "
        
        return response
    
    def calculate_integration_strength(self) -> float:
        """Calcula fuerza promedio de integración"""
        if not self.integration_history:
            return 0.0
        
        recent_integrations = self.integration_history[-5:]  # Últimas 5
        avg_strength = np.mean([item['integration_strength'] for item in recent_integrations])
        return avg_strength
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'integration_strength': self.calculate_integration_strength(),
            'history_length': len(self.integration_history),
            'last_response_type': self.integration_history[-1]['response_type'] if self.integration_history else 'none'
        }