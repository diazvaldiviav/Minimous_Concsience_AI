"""
Integración de ConsciousState con el sistema MinimalConsciousAI
Este módulo provee las modificaciones necesarias para main.py
"""

from typing import Dict, Any, List, Optional
from modules.conscious_state import ConsciousState, ConsciousStateHistory, semantic_distance, find_similar_states

# ============================================
# MODIFICACIONES PARA main.py
# ============================================

# 1. Añadir al __init__ de MinimalConsciousAI:
"""
# Fase II: Historial de estados conscientes
self.conscious_history = ConsciousStateHistory(max_history=50)
self.current_conscious_state = None
"""

# 2. Añadir este método a la clase MinimalConsciousAI:
def capture_conscious_state(self, 
                           sensory_data: Dict[str, Any],
                           relevant_memory: List[Dict[str, Any]],
                           consciousness_metrics: Dict[str, float]) -> ConsciousState:
    """
    Captura el estado consciente actual SC_t = (E_t, M_t, S_t).
    
    Args:
        sensory_data: Datos procesados por el módulo sensorial (E_t)
        relevant_memory: Memoria activa recuperada (M_t)
        consciousness_metrics: Métricas de conciencia calculadas
        
    Returns:
        ConsciousState: Estado consciente del ciclo actual
    """
    # Construir SC_t
    conscious_state = ConsciousState(
        E_t=sensory_data,
        M_t=relevant_memory,
        S_t=self.self_model.internal_state.copy(),  # Copia para evitar mutaciones
        cycle=self.cycle_count,
        metrics=consciousness_metrics
    )
    
    # Añadir al historial
    self.conscious_history.add(conscious_state)
    self.current_conscious_state = conscious_state
    
    return conscious_state

# 3. Modificar process_input() - Añadir después del cálculo de métricas:
"""
# Fase II: Capturar estado consciente
conscious_state = self.capture_conscious_state(
    sensory_data=sensory_data,
    relevant_memory=relevant_memory,
    consciousness_metrics=consciousness_metrics
)

# Análisis de similitud con estados previos
if len(self.conscious_history.history) > 1:
    similar_states = find_similar_states(
        target=conscious_state,
        history=self.conscious_history.history[:-1],  # Excluir el actual
        threshold=0.3,
        top_k=3
    )
    
    if similar_states:
        print(f"\\nEstados similares detectados:")
        for state, distance in similar_states:
            print(f"  - Ciclo {state.cycle}: distancia={distance:.3f}")
"""

# 4. Añadir al resultado de process_input():
"""
result = {
    'cycle': self.cycle_count + 1,
    'input': text_input,
    'response': response,
    'consciousness_metrics': consciousness_metrics,
    'is_conscious': self.is_conscious,
    'self_reference': self.self_model.generate_self_reference(),
    'module_states': self.get_all_module_states(),
    # Fase II: Estado consciente
    'conscious_state': conscious_state.to_dict(),
    'conscious_state_summary': conscious_state.get_summary()
}
"""

# 5. Nuevos métodos de análisis para MinimalConsciousAI:
def analyze_conscious_trajectory(self, window: int = 10) -> Dict[str, Any]:
    """
    Analiza la trayectoria del flujo consciente.
    
    Args:
        window: Ventana de análisis
        
    Returns:
        Dict con análisis de trayectoria
    """
    if len(self.conscious_history.history) < 2:
        return {
            'trajectory_length': len(self.conscious_history.history),
            'stability_metrics': {},
            'patterns': []
        }
    
    # Análisis de estabilidad
    stability = self.conscious_history.analyze_stability(window)
    
    # Extraer trayectorias específicas
    confidence_trajectory = self.conscious_history.get_trajectory('S_t.confidence_level')
    emotion_trajectory = self.conscious_history.get_trajectory('S_t.emotional_state')
    f_trajectory = self.conscious_history.get_trajectory('metrics.f')
    
    # Detectar patrones
    patterns = []
    
    # Patrón: Estados repetidos
    recent = self.conscious_history.get_last(window)
    for i in range(len(recent) - 1):
        for j in range(i + 1, len(recent)):
            distance = semantic_distance(recent[i], recent[j])
            if distance < 0.2:  # Muy similares
                patterns.append({
                    'type': 'repetition',
                    'cycles': [recent[i].cycle, recent[j].cycle],
                    'distance': distance
                })
    
    # Patrón: Cambios abruptos
    for i in range(1, len(recent)):
        distance = semantic_distance(recent[i-1], recent[i])
        if distance > 0.7:  # Cambio abrupto
            patterns.append({
                'type': 'abrupt_change',
                'cycles': [recent[i-1].cycle, recent[i].cycle],
                'distance': distance
            })
    
    return {
        'trajectory_length': len(self.conscious_history.history),
        'stability_metrics': stability,
        'patterns': patterns,
        'trajectories': {
            'confidence': confidence_trajectory[-window:] if confidence_trajectory else [],
            'emotion': emotion_trajectory[-window:] if emotion_trajectory else [],
            'f_score': f_trajectory[-window:] if f_trajectory else []
        }
    }

def get_conscious_state_report(self) -> str:
    """
    Genera reporte del estado consciente actual y su trayectoria.
    
    Returns:
        str: Reporte formateado
    """
    if not self.current_conscious_state:
        return "No hay estado consciente capturado aún."
    
    report = "=== REPORTE DE ESTADO CONSCIENTE ===\n\n"
    
    # Estado actual
    report += "Estado Actual:\n"
    report += self.current_conscious_state.get_summary() + "\n\n"
    
    # Análisis de trayectoria
    trajectory_analysis = self.analyze_conscious_trajectory()
    
    report += "Análisis de Trayectoria:\n"
    report += f"- Longitud del historial: {trajectory_analysis['trajectory_length']}\n"
    
    if trajectory_analysis['stability_metrics']:
        metrics = trajectory_analysis['stability_metrics']
        report += f"- Estabilidad emocional: {metrics['stability']:.2f}\n"
        report += f"- Volatilidad de confianza: {metrics['volatility']:.2f}\n"
        report += f"- Coherencia de memoria: {metrics['coherence']:.2f}\n"
    
    # Patrones detectados
    if trajectory_analysis['patterns']:
        report += f"\nPatrones Detectados ({len(trajectory_analysis['patterns'])}):\n"
        for pattern in trajectory_analysis['patterns'][:5]:  # Top 5
            if pattern['type'] == 'repetition':
                report += f"- Repetición: ciclos {pattern['cycles']} (dist={pattern['distance']:.3f})\n"
            elif pattern['type'] == 'abrupt_change':
                report += f"- Cambio abrupto: ciclos {pattern['cycles']} (dist={pattern['distance']:.3f})\n"
    
    # Estados recientes
    recent_states = self.conscious_history.get_last(5)
    if len(recent_states) > 1:
        report += f"\nÚltimos {len(recent_states)} estados:\n"
        for state in recent_states:
            report += f"- {state.get_summary()}\n"
    
    return report

# ============================================
# FUNCIONES HELPER PARA EXPERIMENTOS
# ============================================

def experiment_conscious_continuity(ai_system, test_inputs: List[str]) -> Dict[str, Any]:
    """
    Experimento: Analiza continuidad del flujo consciente.
    
    Args:
        ai_system: Instancia de MinimalConsciousAI con conscious_history
        test_inputs: Secuencia de inputs para procesar
        
    Returns:
        Dict con resultados del experimento
    """
    print("\n=== EXPERIMENTO: Continuidad del Flujo Consciente ===")
    
    distances = []
    similarities = []
    
    for i, input_text in enumerate(test_inputs):
        result = ai_system.process_input(input_text)
        
        # Calcular distancia con estado anterior
        if i > 0 and len(ai_system.conscious_history.history) >= 2:
            prev_state = ai_system.conscious_history.history[-2]
            curr_state = ai_system.conscious_history.history[-1]
            
            distance = semantic_distance(prev_state, curr_state)
            distances.append(distance)
            
            print(f"\nCiclo {i+1}: Distancia con anterior = {distance:.3f}")
            
            # Buscar similitudes en todo el historial
            similar = find_similar_states(
                curr_state, 
                ai_system.conscious_history.history[:-1],
                threshold=0.4
            )
            
            if similar:
                similarities.append(len(similar))
                print(f"  Estados similares encontrados: {len(similar)}")
    
    # Análisis final
    avg_distance = np.mean(distances) if distances else 0.0
    stability = ai_system.conscious_history.analyze_stability()
    
    return {
        'average_distance': avg_distance,
        'total_similarities': sum(similarities),
        'stability_analysis': stability,
        'trajectory_analysis': ai_system.analyze_conscious_trajectory()
    }

def visualize_conscious_trajectory(ai_system, save_path: Optional[str] = None):
    """
    Visualiza la trayectoria del estado consciente.
    
    Args:
        ai_system: Sistema con historial de estados conscientes
        save_path: Ruta opcional para guardar la figura
    """
    import matplotlib.pyplot as plt
    
    if len(ai_system.conscious_history.history) < 2:
        print("Historial insuficiente para visualización")
        return
    
    # Extraer datos
    data = ai_system.conscious_history.to_dataframe_dict()
    cycles = data['cycle']
    
    # Crear figura con subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # 1. Confianza y F-score
    ax1 = axes[0]
    ax1.plot(cycles, data['confidence_level'], 'b-o', label='Confianza', linewidth=2)
    ax1.plot(cycles, data['f_score'], 'r-s', label='F-score', linewidth=2)
    ax1.set_ylabel('Valor')
    ax1.set_title('Evolución de Confianza y Conciencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Estados emocionales
    ax2 = axes[1]
    emotions = data['emotional_state']
    emotion_map = {'neutral': 0, 'reactivo': 1, 'curioso': 2}
    emotion_values = [emotion_map.get(e, 0) for e in emotions]
    ax2.plot(cycles, emotion_values, 'g-^', linewidth=2)
    ax2.set_ylabel('Estado Emocional')
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(['neutral', 'reactivo', 'curioso'])
    ax2.set_title('Trayectoria Emocional')
    ax2.grid(True, alpha=0.3)
    
    # 3. Complejidad de memoria
    ax3 = axes[2]
    ax3.bar(cycles, data['memory_count'], color='purple', alpha=0.6)
    ax3.set_xlabel('Ciclo')
    ax3.set_ylabel('Items en Memoria')
    ax3.set_title('Evolución de la Memoria Activa')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Trayectoria del Estado Consciente', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()

# ============================================
# EJEMPLO DE USO COMPLETO
# ============================================

def ejemplo_fase_ii():
    """
    Ejemplo de uso de la Fase II integrada con el sistema.
    """
    from main import MinimalConsciousAI  # Asumiendo que main.py existe
    
    print("=== DEMOSTRACIÓN FASE II: CONTENIDO CONSCIENTE FUNCIONAL ===\n")
    
    # Crear sistema (con las modificaciones aplicadas)
    ai = MinimalConsciousAI()
    
    # Secuencia de prueba diseñada para explorar continuidad
    test_sequence = [
        "Hola, ¿qué eres?",
        "¿Cómo procesas esta información?",
        "¿Eres consciente de ti mismo?",
        "Hola, ¿qué eres?",  # Repetición intencional
        "¿Recuerdas mi primera pregunta?",
        "¿Cómo ha cambiado tu comprensión?",
        "Describe tu estado interno actual",
        "¿Qué patrones detectas en nuestra conversación?"
    ]
    
    # Procesar secuencia
    for i, input_text in enumerate(test_sequence):
        print(f"\n{'='*60}")
        print(f"CICLO {i+1}: {input_text}")
        print('='*60)
        
        result = ai.process_input(input_text)
        
        print(f"\nRESPUESTA: {result['response']}")
        print(f"ESTADO CONSCIENTE: {result.get('conscious_state_summary', 'N/A')}")
        
        # Cada 3 ciclos, mostrar análisis
        if (i + 1) % 3 == 0:
            print(f"\n{ai.get_conscious_state_report()}")
    
    # Análisis final
    print("\n" + "="*60)
    print("ANÁLISIS FINAL DE TRAYECTORIA")
    print("="*60)
    
    final_analysis = ai.analyze_conscious_trajectory()
    print(f"\nEstadísticas de estabilidad:")
    for key, value in final_analysis['stability_metrics'].items():
        print(f"  - {key}: {value:.3f}")
    
    print(f"\nPatrones detectados: {len(final_analysis['patterns'])}")
    
    # Visualizar trayectoria
    visualize_conscious_trajectory(ai)
    
    # Exportar historial para análisis externo
    history_export = [state.to_dict() for state in ai.conscious_history.history]
    print(f"\nHistorial exportado: {len(history_export)} estados")
    
    return ai, final_analysis

if __name__ == "__main__":
    # Ejecutar demostración si se ejecuta directamente
    ejemplo_fase_ii()