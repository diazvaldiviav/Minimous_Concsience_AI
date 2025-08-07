import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class ReentranceModule:
    """Módulo de reentrancia - maneja loops de retroalimentación interna"""
    
    def __init__(self):
        self.active_loops = []
        self.feedback_history = []
        self.loop_counter = 0
        
    def create_feedback_loop(self, source_module: str, target_module: str, 
                           feedback_data: Dict[str, Any]) -> int:
        """Crea un nuevo loop de retroalimentación"""
        loop_id = self.loop_counter
        self.loop_counter += 1
        
        feedback_loop = {
            'id': loop_id,
            'source': source_module,
            'target': target_module,
            'data': feedback_data,
            'strength': 1.0,
            'cycles_active': 0,
            'created_at': datetime.now()
        }
        
        self.active_loops.append(feedback_loop)
        return loop_id
    
    def update_loops(self) -> Dict[str, List[Dict[str, Any]]]:
        """Actualiza todos los loops activos y procesa retroalimentación"""
        processed_feedback = {}
        
        for loop in self.active_loops:
            loop['cycles_active'] += 1
            loop['strength'] *= 0.95  # Decay gradual
            
            # Organizar feedback por módulo objetivo
            target = loop['target']
            if target not in processed_feedback:
                processed_feedback[target] = []
            
            processed_feedback[target].append({
                'source': loop['source'],
                'data': loop['data'],
                'strength': loop['strength'],
                'cycles': loop['cycles_active']
            })
        
        # Remover loops débiles
        self.active_loops = [loop for loop in self.active_loops if loop['strength'] > 0.1]
        
        # Guardar historial
        self.feedback_history.append({
            'timestamp': datetime.now(),
            'active_loops': len(self.active_loops),
            'total_feedback': sum(len(feedback) for feedback in processed_feedback.values())
        })
        
        return processed_feedback
    
    def calculate_loop_metrics(self) -> Dict[str, float]:
        """Calcula métricas de los loops de retroalimentación"""
        if not self.active_loops:
            return {'count': 0, 'avg_strength': 0, 'avg_cycles': 0}
        
        total_strength = sum(loop['strength'] for loop in self.active_loops)
        avg_strength = total_strength / len(self.active_loops)
        avg_cycles = np.mean([loop['cycles_active'] for loop in self.active_loops])
        
        return {
            'count': len(self.active_loops),
            'avg_strength': avg_strength,
            'avg_cycles': avg_cycles,
            'total_strength': total_strength
        }
    
    def get_state(self) -> Dict[str, Any]:
        metrics = self.calculate_loop_metrics()
        return {
            'active_loops': metrics['count'],
            'total_strength': metrics.get('total_strength', 0),
            'avg_cycles': metrics.get('avg_cycles', 0)
        }
    
    def reset(self):
     self.active_loops = []
     self.feedback_history = []
     self.loop_counter = 0