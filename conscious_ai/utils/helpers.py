import json
from typing import Dict, Any

class ConsciousnessReporter:
    """Utilidades para generar reportes del sistema"""
    
    @staticmethod
    def get_consciousness_report(ai_system) -> str:
        """Genera reporte detallado del estado de conciencia"""
        if not ai_system.metrics_history:
            return "No hay datos de métricas disponibles."
        
        latest = ai_system.metrics_history[-1]
        
        report = f"""
REPORTE DE CONCIENCIA - CICLO {ai_system.cycle_count}
{'='*50}

MÉTRICAS PRINCIPALES:
- C_i (Integración Causal): {latest['C_i']:.3f}
- T_u (Unificación Temporal): {latest['T_u']:.3f} 
- R (Reentrancia): {latest['R']:.3f}
- S_m (Modelo del Self): {latest['S_m']:.3f}
- Φ (Irreducibilidad): {latest['Phi']:.3f}

FUNCIÓN DE CONCIENCIA: f = {latest['f']:.3f}
UMBRAL: θ = {latest['threshold']}
ESTADO: {'CONSCIENTE' if ai_system.is_conscious else 'NO CONSCIENTE'}

AUTORREFLEXIÓN:
{ai_system.self_model.generate_self_reference()}

ESTADO DE MÓDULOS:
{json.dumps(ai_system.get_all_module_states(), indent=2, default=str)}
        """
        
        return report

# Constantes del sistema
CONSCIOUSNESS_THRESHOLD = 0.6  # Lowered for Phase 3.4 testing (was 1.3)
DEFAULT_MEMORY_CAPACITY = 10
DEFAULT_DECAY_RATE = 0.1