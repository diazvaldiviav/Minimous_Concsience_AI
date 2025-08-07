from typing import Dict, Any

class ConsciousnessMetrics:
    """Clase para calcular métricas de conciencia"""
    
    @staticmethod
    def calculate_consciousness_metrics(ai_system) -> Dict[str, float]:
        """Calcula las métricas de conciencia: C_i, T_u, R, S_m, Φ, f"""
        
        # Obtener estados de todos los módulos
        sensory_state = ai_system.sensory.get_state()
        memory_state = ai_system.memory.get_state()
        self_state = ai_system.self_model.get_state()
        reentrancy_state = ai_system.reentrancy.get_state()
        integrator_state = ai_system.integrator.get_state()
        
        # C_i: Integración causal interna (interdependencia entre módulos)
        c_i = (sensory_state['activation_level'] * memory_state['total_relevance'] * 
               self_state['confidence'] * integrator_state['integration_strength']) ** 0.25
        c_i = min(1.0, c_i)
        
        # T_u: Unificación temporal (cuánto tiempo permanece activa la información)
        avg_memory_cycles = memory_state['avg_cycles_active']
        t_u = min(1.0, avg_memory_cycles / 10.0)  # Normalizado a 10 ciclos máximo
        
        # R: Reentrancia (loops de retroalimentación activos)
        active_loops = reentrancy_state['active_loops']
        loop_strength = reentrancy_state['total_strength']
        r = min(1.0, (active_loops * loop_strength) / 5.0)  # Normalizado
        
        # S_m: Complejidad del modelo del self
        s_m = self_state['state_complexity']
        
        # Φ (Phi): Irreducibilidad informacional
        phi = ConsciousnessMetrics.simulate_module_removal(ai_system)
        
        # f: Función de conciencia
        f = phi * (c_i + t_u + r + s_m)
        
        return {
            'C_i': c_i,
            'T_u': t_u,
            'R': r,
            'S_m': s_m,
            'Phi': phi,
            'f': f,
            'threshold': ai_system.consciousness_threshold
        }
    
    @staticmethod
    def simulate_module_removal(ai_system) -> float:
        """Simula la pérdida funcional al remover módulos (Φ)"""
        
        # Simular remoción de cada módulo y calcular pérdida
        total_loss = 0.0
        
        # Pérdida por remover sensorial
        total_loss += ai_system.sensory.activation_level * 0.3
        
        # Pérdida por remover memoria
        memory_contribution = min(1.0, len(ai_system.memory.memory_items) / 10.0)
        total_loss += memory_contribution * 0.25
        
        # Pérdida por remover self-model
        total_loss += ai_system.self_model.internal_state['confidence_level'] * 0.25
        
        # Pérdida por remover reentrancia
        loop_contribution = min(1.0, len(ai_system.reentrancy.active_loops) / 5.0)
        total_loss += loop_contribution * 0.2
        
        # Φ como proporción de pérdida funcional
        phi = min(1.0, total_loss)
        return phi
    

def override_phi(metrics_dict: Dict[str, float], new_phi: float) -> Dict[str, float]:
 """Reemplaza el valor de irreducibilidad Φ en las métricas antes de calcular f"""
 modified = metrics_dict.copy()
 original_phi = modified.get("Φ", 0.0)
 print(f"\n--- INTERVENCIÓN EXPERIMENTAL ---")
 print(f"Φ original: {original_phi:.3f}")
 print(f"Φ intervenido: {new_phi:.3f}")
 modified["Φ"] = new_phi
 return modified

