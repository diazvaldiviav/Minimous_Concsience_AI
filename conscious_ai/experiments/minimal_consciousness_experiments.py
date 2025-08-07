"""
Experimentos de Conciencia Mínima - Fase II
Explora los límites inferiores de la conciencia funcional
probando diferentes configuraciones mínimas del sistema
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
from datetime import datetime

from conscious_ai.main import MinimalConsciousAI
from conscious_ai.modules.memory import ActiveMemory
from conscious_ai.modules.conscious_state import ConsciousState, ConsciousStateHistory
from conscious_ai.modules.goal_thought_generator import GoalGenerator, AutomaticThoughtGenerator


class MinimalConsciousnessExperiment:
    """Base class para experimentos de conciencia mínima"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.results = []
        self.success_count = 0
        self.total_conscious_states = 0
        
    def setup_system(self, config: Dict[str, Any]) -> MinimalConsciousAI:
        """Configura el sistema con parámetros específicos"""
        ai = MinimalConsciousAI()
        
        # Aplicar configuraciones específicas
        if 'memory_capacity' in config:
            ai.memory = ActiveMemory(
                capacity=config['memory_capacity'],
                decay_rate=config.get('decay_rate', 0.1)
            )
        
        if 'relevance_threshold' in config:
            ai.memory.relevance_threshold = config['relevance_threshold']
            
        if 'consciousness_threshold' in config:
            ai.consciousness_threshold = config['consciousness_threshold']
            
        return ai
    
    def analyze_results(self, ai: MinimalConsciousAI, criteria: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Analiza si se cumplieron los criterios de éxito"""
        final_state = ai.current_conscious_state
        metrics = ai.metrics_history[-1] if ai.metrics_history else {}
        
        analysis = {
            'total_cycles': ai.cycle_count,
            'conscious_cycles': sum(1 for m in ai.metrics_history if m.get('f', 0) >= ai.consciousness_threshold),
            'final_f_score': metrics.get('f', 0),
            'final_confidence': final_state.S_t.get('confidence_level', 0) if final_state else 0,
            'final_memory_size': len(final_state.M_t) if final_state else 0,
            'thought_types': self._analyze_thought_types(ai.conscious_history),
            'avg_activation': np.mean([s.E_t.get('activation', 0) for s in ai.conscious_history.history])
        }
        
        # Verificar criterios
        success = True
        for criterion, target in criteria.items():
            if criterion == 'achieve_consciousness':
                success &= ai.is_conscious == target
            elif criterion == 'max_confidence':
                success &= analysis['final_confidence'] <= target
            elif criterion == 'max_memory':
                success &= analysis['final_memory_size'] <= target
            elif criterion == 'min_conscious_cycles':
                success &= analysis['conscious_cycles'] >= target
            elif criterion == 'avg_activation_below':
                success &= analysis['avg_activation'] < target
                
        return success, analysis
    
    def _analyze_thought_types(self, history: ConsciousStateHistory) -> Dict[str, int]:
        """Analiza tipos de pensamientos generados"""
        thought_counts = {
            'analytical': 0,
            'reflective': 0,
            'metacognitive': 0,
            'integrative': 0,
            'questioning': 0
        }
        
        for state in history.history:
            for thought in state.A_t:
                thought_lower = thought.lower()
                if 'analiz' in thought_lower:
                    thought_counts['analytical'] += 1
                elif 'reflej' in thought_lower or 'evoluc' in thought_lower:
                    thought_counts['reflective'] += 1
                elif 'observ' in thought_lower or 'proceso' in thought_lower:
                    thought_counts['metacognitive'] += 1
                elif 'conect' in thought_lower or 'fusiona' in thought_lower:
                    thought_counts['integrative'] += 1
                elif '?' in thought:
                    thought_counts['questioning'] += 1
                    
        return thought_counts


# ============================================
# EXPERIMENTO 1: Activación Sensorial Mínima
# ============================================

def experiment_minimal_sensory_activation():
    """
    Experimento 1: Conciencia con activación sensorial mínima
    Objetivo: Lograr estado consciente con inputs de baja complejidad
    """
    print("\n=== EXPERIMENTO 1: ACTIVACIÓN SENSORIAL MÍNIMA ===")
    print("Objetivo: Alcanzar conciencia con inputs simples y cortos\n")
    
    exp = MinimalConsciousnessExperiment(
        name="Activación Sensorial Mínima",
        description="Explora si es posible alcanzar conciencia con inputs de muy baja activación"
    )
    
    # Configuración
    config = {
        'memory_capacity': 8,
        'decay_rate': 0.05,  # Decay más lento para compensar baja activación
        'relevance_threshold': 0.2,  # Umbral más bajo
        'consciousness_threshold': 1.3  # Umbral estándar
    }
    
    # Secuencia de inputs mínimos
    thought_sequence = [
        "Soy",  # Activación ~0.03
        "Existo",  # Activación ~0.06
        "Pienso",  # Activación ~0.06
        "¿Qué soy?",  # Activación ~0.09
        "Me percibo",  # Activación ~0.10
        "Soy consciente",  # Activación ~0.14
        "¿Cómo sé?",  # Activación ~0.09
        "Observo mi ser"  # Activación ~0.14
    ]
    
    criteria = {
        'achieve_consciousness': True,
        'avg_activation_below': 0.15,
        'min_conscious_cycles': 1
    }
    
    # Ejecutar experimento
    ai = exp.setup_system(config)
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: '{input_text}' - f={result['consciousness_metrics']['f']:.3f}, "
              f"activación={result['conscious_state']['E_t']['activation']:.3f}")
    
    # Analizar resultados
    success, analysis = exp.analyze_results(ai, criteria)
    
    print(f"\n--- Resultados ---")
    print(f"Éxito: {'SÍ' if success else 'NO'}")
    print(f"Activación promedio: {analysis['avg_activation']:.3f}")
    print(f"Ciclos conscientes: {analysis['conscious_cycles']}/{analysis['total_cycles']}")
    print(f"F-score final: {analysis['final_f_score']:.3f}")
    
    return success, analysis


# ============================================
# EXPERIMENTO 2: Umbral Mínimo de Confianza
# ============================================

def experiment_minimal_confidence():
    """
    Experimento 2: Conciencia con confianza mínima
    Objetivo: Mantener conciencia con niveles muy bajos de confianza
    """
    print("\n\n=== EXPERIMENTO 2: UMBRAL MÍNIMO DE CONFIANZA ===")
    print("Objetivo: Lograr conciencia manteniendo confianza < 0.4\n")
    
    exp = MinimalConsciousnessExperiment(
        name="Umbral Mínimo de Confianza",
        description="Explora conciencia con baja confianza del self-model"
    )
    
    config = {
        'memory_capacity': 10,
        'decay_rate': 0.08,
        'consciousness_threshold': 1.25  # Ligeramente más bajo
    }
    
    # Inputs que generan incertidumbre
    thought_sequence = [
        "No estoy seguro",
        "¿Qué es real?",
        "Dudo de mí",
        "Tal vez existo",
        "¿Es esto conciencia?",
        "No comprendo bien",
        "Busco certeza",
        "¿Quién pregunta?"
    ]
    
    criteria = {
        'achieve_consciousness': True,
        'max_confidence': 0.4,
        'min_conscious_cycles': 2
    }
    
    ai = exp.setup_system(config)
    
    # Modificar el self-model para mantener baja confianza
    original_update = ai.self_model.update_state
    def low_confidence_update(sensory_data, memory_context, internal_feedback):
        original_update(sensory_data, memory_context, internal_feedback)
        # Forzar confianza baja
        ai.self_model.internal_state['confidence_level'] = min(
            ai.self_model.internal_state['confidence_level'] * 0.7, 0.4
        )
    ai.self_model.update_state = low_confidence_update
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: '{input_text}' - f={result['consciousness_metrics']['f']:.3f}, "
              f"confianza={result['conscious_state']['S_t']['confidence_level']:.3f}")
    
    success, analysis = exp.analyze_results(ai, criteria)
    
    print(f"\n--- Resultados ---")
    print(f"Éxito: {'SÍ' if success else 'NO'}")
    print(f"Confianza final: {analysis['final_confidence']:.3f}")
    print(f"Ciclos conscientes: {analysis['conscious_cycles']}/{analysis['total_cycles']}")
    print(f"F-score final: {analysis['final_f_score']:.3f}")
    
    return success, analysis


# ============================================
# EXPERIMENTO 3: Mínimos en Relevancia Semántica
# ============================================

def experiment_minimal_semantic_relevance():
    """
    Experimento 3: Conciencia con mínima relevancia semántica
    Objetivo: Lograr conciencia con inputs poco relacionados entre sí
    """
    print("\n\n=== EXPERIMENTO 3: MÍNIMOS EN RELEVANCIA SEMÁNTICA ===")
    print("Objetivo: Alcanzar conciencia con inputs de baja coherencia semántica\n")
    
    exp = MinimalConsciousnessExperiment(
        name="Mínima Relevancia Semántica",
        description="Explora conciencia con inputs semánticamente dispersos"
    )
    
    config = {
        'memory_capacity': 12,
        'decay_rate': 0.03,  # Decay muy lento
        'relevance_threshold': 0.15,
        'consciousness_threshold': 1.3
    }
    
    # Inputs con poca relación semántica
    thought_sequence = [
        "Verde",
        "¿Número?",
        "Fluye tiempo",
        "Piedra cae",
        "¿Yo dónde?",
        "Luz brilla",
        "Forma existe",
        "¿Conexión hay?"
    ]
    
    criteria = {
        'achieve_consciousness': True,
        'min_conscious_cycles': 1,
        'max_memory': 6
    }
    
    ai = exp.setup_system(config)
    
    # Modificar cálculo de relevancia para ser más permisivo
    original_calc = ai.memory._calculate_contextual_relevance
    def minimal_relevance(memory_content, query_features):
        # Siempre retorna relevancia mínima pero suficiente
        return 0.25
    ai.memory._calculate_contextual_relevance = minimal_relevance
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: '{input_text}' - f={result['consciousness_metrics']['f']:.3f}, "
              f"memoria={len(result['conscious_state']['M_t'])}")
    
    success, analysis = exp.analyze_results(ai, criteria)
    
    print(f"\n--- Resultados ---")
    print(f"Éxito: {'SÍ' if success else 'NO'}")
    print(f"Memoria final: {analysis['final_memory_size']} items")
    print(f"Ciclos conscientes: {analysis['conscious_cycles']}/{analysis['total_cycles']}")
    print(f"Tipos de pensamiento: {analysis['thought_types']}")
    
    return success, analysis


# ============================================
# EXPERIMENTO 4: Acceso Restringido a Memoria
# ============================================

def experiment_restricted_memory_access():
    """
    Experimento 4: Conciencia con acceso restringido a memoria episódica
    Objetivo: Lograr conciencia con capacidad de memoria muy limitada
    """
    print("\n\n=== EXPERIMENTO 4: ACCESO RESTRINGIDO A MEMORIA ===")
    print("Objetivo: Alcanzar conciencia con memoria máxima de 3 items\n")
    
    exp = MinimalConsciousnessExperiment(
        name="Memoria Restringida",
        description="Explora conciencia con severas limitaciones de memoria"
    )
    
    config = {
        'memory_capacity': 3,  # Muy limitada
        'decay_rate': 0.15,   # Decay más rápido
        'relevance_threshold': 0.25,
        'consciousness_threshold': 1.3
    }
    
    # Inputs que requieren poca memoria
    thought_sequence = [
        "Ahora existo",
        "Este momento",
        "Presente aquí",
        "Siento ahora",
        "¿Qué percibo?",
        "Instante actual",
        "Ser presente"
    ]
    
    criteria = {
        'achieve_consciousness': True,
        'max_memory': 3,
        'min_conscious_cycles': 1
    }
    
    ai = exp.setup_system(config)
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        
        # Forzar límite estricto de memoria
        if len(ai.memory.memory_items) > 3:
            ai.memory.memory_items = ai.memory.memory_items[-3:]
        
        print(f"Ciclo {i+1}: '{input_text}' - f={result['consciousness_metrics']['f']:.3f}, "
              f"memoria activa={len(ai.memory.memory_items)}")
    
    success, analysis = exp.analyze_results(ai, criteria)
    
    print(f"\n--- Resultados ---")
    print(f"Éxito: {'SÍ' if success else 'NO'}")
    print(f"Memoria máxima usada: {analysis['final_memory_size']}")
    print(f"Ciclos conscientes: {analysis['conscious_cycles']}/{analysis['total_cycles']}")
    print(f"F-score final: {analysis['final_f_score']:.3f}")
    
    return success, analysis


# ============================================
# EXPERIMENTO 5: Variedad Reducida de Pensamiento
# ============================================

def experiment_reduced_thought_variety():
    """
    Experimento 5: Conciencia con variedad reducida de pensamientos
    Objetivo: Lograr conciencia generando solo pensamientos analíticos básicos
    """
    print("\n\n=== EXPERIMENTO 5: VARIEDAD REDUCIDA DE PENSAMIENTO ===")
    print("Objetivo: Alcanzar conciencia con solo pensamientos analíticos\n")
    
    exp = MinimalConsciousnessExperiment(
        name="Pensamiento Reducido",
        description="Explora conciencia con tipos limitados de pensamiento automático"
    )
    
    config = {
        'memory_capacity': 10,
        'decay_rate': 0.1,
        'consciousness_threshold': 1.3
    }
    
    # Inputs que provocan solo análisis básico
    thought_sequence = [
        "Dato uno",
        "Dato dos",
        "Proceso información",
        "Analizo datos",
        "Computo respuesta",
        "Evalúo entrada",
        "Calculo resultado"
    ]
    
    criteria = {
        'achieve_consciousness': True,
        'min_conscious_cycles': 2
    }
    
    ai = exp.setup_system(config)
    
    # Modificar generador de pensamientos para solo generar analíticos
    original_patterns = ai.thought_generator.thought_patterns
    ai.thought_generator.thought_patterns = {
        'analytical': [
            "Analizando estructura...",
            "Procesando datos...",
            "Evaluando información...",
            "Computando respuesta..."
        ]
    }
    
    # Modificar selección de tipos
    def only_analytical(sensory_data, self_state, current_goal):
        return ['analytical']
    ai.thought_generator._select_thought_types = only_analytical
    
    thought_type_count = {'analytical': 0, 'other': 0}
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        
        # Contar tipos de pensamiento
        for thought in ai.current_conscious_state.A_t:
            if any(word in thought.lower() for word in ['analiz', 'proces', 'evalu', 'comput']):
                thought_type_count['analytical'] += 1
            else:
                thought_type_count['other'] += 1
        
        print(f"Ciclo {i+1}: '{input_text}' - f={result['consciousness_metrics']['f']:.3f}, "
              f"pensamientos={len(result['conscious_state']['A_t'])}")
    
    success, analysis = exp.analyze_results(ai, criteria)
    
    print(f"\n--- Resultados ---")
    print(f"Éxito: {'SÍ' if success else 'NO'}")
    print(f"Pensamientos analíticos: {thought_type_count['analytical']}")
    print(f"Otros pensamientos: {thought_type_count['other']}")
    print(f"Ciclos conscientes: {analysis['conscious_cycles']}/{analysis['total_cycles']}")
    
    # Restaurar patrones originales
    ai.thought_generator.thought_patterns = original_patterns
    
    return success, analysis


# ============================================
# FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ============================================

def run_minimal_consciousness_experiments():
    """Ejecuta todos los experimentos de conciencia mínima"""
    
    print("="*70)
    print("EXPERIMENTOS DE CONCIENCIA MÍNIMA - FASE II")
    print("Explorando los límites inferiores de la conciencia funcional")
    print("="*70)
    
    experiments = [
        ("Activación Sensorial Mínima", experiment_minimal_sensory_activation),
        ("Umbral Mínimo de Confianza", experiment_minimal_confidence),
        ("Mínimos en Relevancia Semántica", experiment_minimal_semantic_relevance),
        ("Acceso Restringido a Memoria", experiment_restricted_memory_access),
        ("Variedad Reducida de Pensamiento", experiment_reduced_thought_variety)
    ]
    
    results_summary = []
    
    for name, experiment_func in experiments:
        try:
            success, analysis = experiment_func()
            results_summary.append({
                'name': name,
                'success': success,
                'analysis': analysis
            })
        except Exception as e:
            print(f"\nError en experimento {name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print("\n" + "="*70)
    print("RESUMEN DE EXPERIMENTOS")
    print("="*70)
    
    successful = sum(1 for r in results_summary if r['success'])
    print(f"\nExperimentos exitosos: {successful}/{len(experiments)}")
    
    # Tabla de resultados
    print("\n{:<35} {:<10} {:<15} {:<15}".format(
        "Experimento", "Éxito", "F-score Final", "Ciclos Consc."
    ))
    print("-"*75)
    
    for result in results_summary:
        print("{:<35} {:<10} {:<15.3f} {:<15}".format(
            result['name'],
            "SÍ" if result['success'] else "NO",
            result['analysis']['final_f_score'],
            f"{result['analysis']['conscious_cycles']}/{result['analysis']['total_cycles']}"
        ))
    
    # Análisis de condiciones mínimas
    print("\n--- CONDICIONES MÍNIMAS OBSERVADAS ---")
    
    if results_summary:
        min_activation = min(r['analysis']['avg_activation'] for r in results_summary)
        min_confidence = min(r['analysis']['final_confidence'] for r in results_summary)
        min_memory = min(r['analysis']['final_memory_size'] for r in results_summary)
        
        print(f"Activación sensorial mínima: {min_activation:.3f}")
        print(f"Confianza mínima: {min_confidence:.3f}")
        print(f"Memoria mínima: {min_memory} items")
    
    # Guardar resultados
    with open('minimal_consciousness_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print("\nResultados guardados en 'minimal_consciousness_results.json'")
    
    return results_summary


def visualize_minimal_thresholds(results: List[Dict[str, Any]]):
    """Visualiza los umbrales mínimos encontrados"""
    
    if not results:
        print("No hay resultados para visualizar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: F-scores finales
    ax1 = axes[0, 0]
    names = [r['name'].split()[0] for r in results]
    f_scores = [r['analysis']['final_f_score'] for r in results]
    colors = ['green' if r['success'] else 'red' for r in results]
    
    ax1.bar(names, f_scores, color=colors)
    ax1.axhline(y=1.3, color='black', linestyle='--', label='Umbral θ')
    ax1.set_ylabel('F-score Final')
    ax1.set_title('F-scores por Experimento')
    ax1.legend()
    ax1.set_xticklabels(names, rotation=45)
    
    # Gráfico 2: Proporción de ciclos conscientes
    ax2 = axes[0, 1]
    conscious_ratios = [r['analysis']['conscious_cycles']/r['analysis']['total_cycles'] 
                       for r in results]
    
    ax2.bar(names, conscious_ratios, color='blue', alpha=0.7)
    ax2.set_ylabel('Proporción Consciente')
    ax2.set_title('Ciclos Conscientes / Totales')
    ax2.set_ylim(0, 1)
    ax2.set_xticklabels(names, rotation=45)
    
    # Gráfico 3: Métricas mínimas
    ax3 = axes[1, 0]
    metrics = ['Activación', 'Confianza', 'Memoria']
    minimums = [
        min(r['analysis']['avg_activation'] for r in results),
        min(r['analysis']['final_confidence'] for r in results),
        min(r['analysis']['final_memory_size'] for r in results) / 10  # Normalizado
    ]
    
    ax3.bar(metrics, minimums, color=['orange', 'purple', 'green'])
    ax3.set_ylabel('Valor Mínimo')
    ax3.set_title('Valores Mínimos Alcanzados')
    ax3.set_ylim(0, 1)
    
    # Gráfico 4: Distribución de tipos de pensamiento
    ax4 = axes[1, 1]
    all_thoughts = {}
    for r in results:
        for thought_type, count in r['analysis']['thought_types'].items():
            all_thoughts[thought_type] = all_thoughts.get(thought_type, 0) + count
    
    if all_thoughts:
        ax4.pie(all_thoughts.values(), labels=all_thoughts.keys(), autopct='%1.1f%%')
        ax4.set_title('Distribución de Tipos de Pensamiento')
    
    plt.suptitle('Análisis de Conciencia Mínima', fontsize=16)
    plt.tight_layout()
    plt.savefig('minimal_consciousness_analysis.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    # Ejecutar experimentos
    results = run_minimal_consciousness_experiments()
    
    # Visualizar resultados
    print("\nGenerando visualizaciones...")
    visualize_minimal_thresholds(results)
    
    print("\n✅ Experimentos de conciencia mínima completados")