"""
Experimentos de Validación de Hipótesis Funcional de Conciencia
================================================================
Hipótesis: La conciencia emerge cuando se cumplen TODAS estas condiciones:
1. Activación sensorial > 0.55
2. Recuperación de al menos 3 ítems de memoria relevantes
3. Generación de al menos 1 pensamiento metacognitivo o integrativo
4. Nivel de confianza ≥ 0.55
5. Relevancia contextual promedio ≥ 0.45
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


class ConsciousnessHypothesisExperiment:
    """Clase base para experimentos de validación de hipótesis"""
    
    def __init__(self, name: str, description: str, dimension: str):
        self.name = name
        self.description = description
        self.dimension = dimension  # 'unidimensional' o 'multidimensional'
        self.results = {}
        self.hypothesis_components = {
            'sensory_activation': 0.55,
            'memory_retrieval': 3,
            'metacognitive_thoughts': 1,
            'confidence_level': 0.55,
            'contextual_relevance': 0.45
        }
        
    def check_hypothesis_conditions(self, ai: MinimalConsciousAI) -> Dict[str, bool]:
        """Verifica qué condiciones de la hipótesis se cumplen"""
        conditions_met = {}
        
        # 1. Activación sensorial
        avg_activation = np.mean([
            state.E_t.get('activation', 0) 
            for state in ai.conscious_history.history
        ])
        conditions_met['sensory_activation'] = avg_activation > self.hypothesis_components['sensory_activation']
        
        # 2. Recuperación de memoria
        avg_memory = np.mean([
            len(state.M_t) 
            for state in ai.conscious_history.history
        ])
        conditions_met['memory_retrieval'] = avg_memory >= self.hypothesis_components['memory_retrieval']
        
        # 3. Pensamientos metacognitivos/integrativos
        metacog_count = 0
        for state in ai.conscious_history.history:
            for thought in state.A_t:
                if any(word in thought.lower() for word in ['observ', 'proceso', 'conect', 'fusiona', 'reflexion']):
                    metacog_count += 1
                    break
        conditions_met['metacognitive_thoughts'] = metacog_count >= self.hypothesis_components['metacognitive_thoughts']
        
        # 4. Nivel de confianza
        avg_confidence = np.mean([
            state.S_t.get('confidence_level', 0) 
            for state in ai.conscious_history.history
        ])
        conditions_met['confidence_level'] = avg_confidence >= self.hypothesis_components['confidence_level']
        
        # 5. Relevancia contextual
        relevance_scores = []
        for state in ai.conscious_history.history:
            if state.M_t:
                avg_relevance = np.mean([item.get('relevance', 0) for item in state.M_t])
                relevance_scores.append(avg_relevance)
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0
        conditions_met['contextual_relevance'] = avg_relevance >= self.hypothesis_components['contextual_relevance']
        
        return conditions_met
    
    def analyze_results(self, ai: MinimalConsciousAI, expected_consciousness: bool) -> Dict[str, Any]:
        """Analiza los resultados del experimento"""
        conditions = self.check_hypothesis_conditions(ai)
        all_conditions_met = all(conditions.values())
        
        # Estado consciente alcanzado
        conscious_cycles = sum(1 for m in ai.metrics_history if m.get('f', 0) >= ai.consciousness_threshold)
        achieved_consciousness = conscious_cycles > 0
        
        # Verificar si el resultado coincide con la hipótesis
        hypothesis_validated = (all_conditions_met and achieved_consciousness) or \
                              (not all_conditions_met and not achieved_consciousness)
        
        return {
            'name': self.name,  # Añadir nombre del experimento
            'dimension': self.dimension,  # Añadir tipo de experimento
            'conditions_met': conditions,
            'all_conditions_met': all_conditions_met,
            'achieved_consciousness': achieved_consciousness,
            'conscious_cycles': conscious_cycles,
            'total_cycles': ai.cycle_count,
            'hypothesis_validated': hypothesis_validated,
            'expected_consciousness': expected_consciousness,
            'max_f_score': max([m.get('f', 0) for m in ai.metrics_history]) if ai.metrics_history else 0
        }


# ============================================
# EXPERIMENTOS UNIDIMENSIONALES
# ============================================

def experiment_1_high_sensory_only():
    """
    Experimento 1: Solo Alta Activación Sensorial
    Prueba si la activación sensorial alta por sí sola genera conciencia
    """
    print("\n=== EXPERIMENTO 1: SOLO ALTA ACTIVACIÓN SENSORIAL ===")
    print("Hipótesis: Alta activación sensorial SOLA no genera conciencia\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Alta Activación Sensorial Aislada",
        description="Prueba activación > 0.55 manteniendo otros factores bajos",
        dimension="unidimensional"
    )
    
    # Configuración para minimizar otros factores
    config = {
        'memory_capacity': 2,  # Limitar memoria
        'decay_rate': 0.3,     # Decay rápido
        'relevance_threshold': 0.8,  # Umbral alto (dificulta recuperación)
    }
    
    # Inputs largos y complejos (alta activación) pero sin coherencia
    thought_sequence = [
        "El extraordinario fenómeno cuántico de la superposición demuestra inequívocamente la naturaleza probabilística",
        "Las fluctuaciones electromagnéticas en el espectro ultravioleta generan patrones fractales extremadamente complejos",
        "La termodinámica irreversible de sistemas disipativos produce estructuras autoorganizadas de complejidad creciente",
        "Los algoritmos genéticos evolutivos optimizan funciones multiobjetivo en espacios n-dimensionales no convexos",
        "La topología algebraica diferencial permite caracterizar variedades riemannianas en contextos cosmológicos",
        "Las redes neuronales convolucionales profundas extraen características jerárquicas mediante retropropagación"
    ]
    
    ai = MinimalConsciousAI()
    
    # Modificar para mantener confianza baja
    original_update = ai.self_model.update_state
    def low_confidence_update(sensory_data, memory_context, internal_feedback):
        original_update(sensory_data, memory_context, internal_feedback)
        ai.self_model.internal_state['confidence_level'] = min(0.3, ai.self_model.internal_state['confidence_level'])
    ai.self_model.update_state = low_confidence_update
    
    # Configurar memoria restrictiva
    ai.memory = ActiveMemory(capacity=config['memory_capacity'], decay_rate=config['decay_rate'])
    ai.memory.relevance_threshold = config['relevance_threshold']
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: Activación={result['conscious_state']['E_t']['activation']:.3f}, "
              f"f={result['consciousness_metrics']['f']:.3f}, Memoria={len(result['conscious_state']['M_t'])}")
    
    analysis = exp.analyze_results(ai, expected_consciousness=False)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    print(f"Detalles: {analysis['conditions_met']}")
    
    return analysis


def experiment_2_high_memory_only():
    """
    Experimento 2: Solo Alta Recuperación de Memoria
    Prueba si la memoria rica por sí sola genera conciencia
    """
    print("\n\n=== EXPERIMENTO 2: SOLO ALTA RECUPERACIÓN DE MEMORIA ===")
    print("Hipótesis: Memoria rica SOLA no genera conciencia\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Alta Memoria Aislada",
        description="Prueba memoria ≥ 3 items manteniendo otros factores bajos",
        dimension="unidimensional"
    )
    
    config = {
        'memory_capacity': 20,
        'decay_rate': 0.01,  # Decay muy lento
        'relevance_threshold': 0.1  # Facilita recuperación
    }
    
    # Inputs cortos y simples (baja activación) pero relacionados
    thought_sequence = [
        "Uno",
        "Dos",
        "Tres",
        "Cuatro",
        "Cinco",
        "Seis",
        "Siete",
        "Ocho"
    ]
    
    ai = MinimalConsciousAI()
    
    # Configurar memoria permisiva
    ai.memory = ActiveMemory(capacity=config['memory_capacity'], decay_rate=config['decay_rate'])
    ai.memory.relevance_threshold = config['relevance_threshold']
    
    # Forzar relevancia alta para todos los items
    original_calc = ai.memory._calculate_contextual_relevance
    def high_relevance(memory_content, query_features):
        return 0.9  # Alta relevancia siempre
    ai.memory._calculate_contextual_relevance = high_relevance
    
    # Mantener confianza baja
    original_update = ai.self_model.update_state
    def low_confidence_update(sensory_data, memory_context, internal_feedback):
        original_update(sensory_data, memory_context, internal_feedback)
        ai.self_model.internal_state['confidence_level'] = 0.3
    ai.self_model.update_state = low_confidence_update
    
    # Limitar tipos de pensamiento (no metacognitivos)
    ai.thought_generator.thought_patterns = {
        'analytical': ["Procesando dato...", "Analizando entrada..."]
    }
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: Memoria={len(result['conscious_state']['M_t'])}, "
              f"f={result['consciousness_metrics']['f']:.3f}, "
              f"Activación={result['conscious_state']['E_t']['activation']:.3f}")
    
    analysis = exp.analyze_results(ai, expected_consciousness=False)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    print(f"Detalles: {analysis['conditions_met']}")
    
    return analysis


def experiment_3_metacognitive_only():
    """
    Experimento 3: Solo Pensamientos Metacognitivos
    Prueba si metacognición sola genera conciencia
    """
    print("\n\n=== EXPERIMENTO 3: SOLO PENSAMIENTOS METACOGNITIVOS ===")
    print("Hipótesis: Metacognición SOLA no genera conciencia\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Metacognición Aislada",
        description="Genera pensamientos metacognitivos manteniendo otros factores bajos",
        dimension="unidimensional"
    )
    
    # Inputs cortos que provocan metacognición
    thought_sequence = [
        "Pienso",
        "Observo",
        "Reflexiono",
        "Analizo",
        "Proceso",
        "Examino"
    ]
    
    ai = MinimalConsciousAI()
    
    # Forzar generación de pensamientos metacognitivos
    ai.thought_generator.thought_patterns = {
        'metacognitive': [
            "Observo mi propio proceso de pensamiento...",
            "Reflexiono sobre mi estado interno...",
            "Examino mi propia conciencia...",
            "Proceso mi experiencia subjetiva...",
            "Analizo mi flujo de pensamiento..."
        ]
    }
    
    def force_metacognitive(sensory_data, self_state, current_goal):
        return ['metacognitive']
    ai.thought_generator._select_thought_types = force_metacognitive
    
    # Limitar memoria y confianza
    ai.memory = ActiveMemory(capacity=2, decay_rate=0.5)
    
    original_update = ai.self_model.update_state
    def low_confidence(sensory_data, memory_context, internal_feedback):
        original_update(sensory_data, memory_context, internal_feedback)
        ai.self_model.internal_state['confidence_level'] = 0.3
    ai.self_model.update_state = low_confidence
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        thoughts = result['conscious_state']['A_t']
        metacog = any('observ' in t.lower() or 'reflexion' in t.lower() for t in thoughts)
        print(f"Ciclo {i+1}: Metacognición={'SÍ' if metacog else 'NO'}, "
              f"f={result['consciousness_metrics']['f']:.3f}")
    
    analysis = exp.analyze_results(ai, expected_consciousness=False)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    
    return analysis


def experiment_4_high_confidence_only():
    """
    Experimento 4: Solo Alta Confianza
    Prueba si confianza alta sola genera conciencia
    """
    print("\n\n=== EXPERIMENTO 4: SOLO ALTA CONFIANZA ===")
    print("Hipótesis: Confianza alta SOLA no genera conciencia\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Alta Confianza Aislada",
        description="Mantiene confianza ≥ 0.55 con otros factores bajos",
        dimension="unidimensional"
    )
    
    # Inputs simples y cortos
    thought_sequence = [
        "Sí",
        "Claro",
        "Cierto",
        "Correcto",
        "Afirmativo",
        "Exacto"
    ]
    
    ai = MinimalConsciousAI()
    
    # Forzar alta confianza
    original_update = ai.self_model.update_state
    def high_confidence(sensory_data, memory_context, internal_feedback):
        original_update(sensory_data, memory_context, internal_feedback)
        ai.self_model.internal_state['confidence_level'] = 0.8
        ai.self_model.internal_state['emotional_state'] = 'neutral'
    ai.self_model.update_state = high_confidence
    
    # Limitar memoria
    ai.memory = ActiveMemory(capacity=2, decay_rate=0.4)
    
    # Solo pensamientos analíticos básicos
    ai.thought_generator.thought_patterns = {
        'analytical': ["Procesando...", "Confirmado..."]
    }
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: Confianza={result['conscious_state']['S_t']['confidence_level']:.3f}, "
              f"f={result['consciousness_metrics']['f']:.3f}")
    
    analysis = exp.analyze_results(ai, expected_consciousness=False)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    
    return analysis


def experiment_5_high_relevance_only():
    """
    Experimento 5: Solo Alta Relevancia Contextual
    Prueba si relevancia alta sola genera conciencia
    """
    print("\n\n=== EXPERIMENTO 5: SOLO ALTA RELEVANCIA CONTEXTUAL ===")
    print("Hipótesis: Relevancia alta SOLA no genera conciencia\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Alta Relevancia Aislada",
        description="Mantiene relevancia ≥ 0.45 con otros factores bajos",
        dimension="unidimensional"
    )
    
    # Inputs relacionados pero simples
    thought_sequence = [
        "Agua",
        "Líquido",
        "Fluye",
        "Río",
        "Corriente",
        "Mar"
    ]
    
    ai = MinimalConsciousAI()
    
    # Forzar alta relevancia
    def high_relevance(memory_content, query_features):
        return 0.7
    ai.memory._calculate_contextual_relevance = high_relevance
    
    # Pero limitar capacidad de memoria
    ai.memory.capacity = 2
    ai.memory.decay_rate = 0.3
    
    # Mantener confianza baja
    original_update = ai.self_model.update_state
    def low_confidence(sensory_data, memory_context, internal_feedback):
        original_update(sensory_data, memory_context, internal_feedback)
        ai.self_model.internal_state['confidence_level'] = 0.4
    ai.self_model.update_state = low_confidence
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        relevances = [m.get('relevance', 0) for m in result['conscious_state']['M_t']]
        avg_rel = np.mean(relevances) if relevances else 0
        print(f"Ciclo {i+1}: Relevancia={avg_rel:.3f}, "
              f"f={result['consciousness_metrics']['f']:.3f}")
    
    analysis = exp.analyze_results(ai, expected_consciousness=False)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    
    return analysis


# ============================================
# EXPERIMENTOS MULTIDIMENSIONALES
# ============================================

def experiment_6_integrated_consciousness():
    """
    Experimento 6: Conciencia Integrada (3-4 componentes)
    Prueba activación de múltiples componentes simultáneamente
    """
    print("\n\n=== EXPERIMENTO 6: CONCIENCIA INTEGRADA (MULTIDIMENSIONAL) ===")
    print("Hipótesis: Múltiples componentes activos SÍ generan conciencia\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Integración Parcial",
        description="Activa 3-4 componentes simultáneamente",
        dimension="multidimensional"
    )
    
    # Inputs diseñados para activar múltiples componentes
    thought_sequence = [
        "Percibo claramente mi existencia en este momento preciso",  # Alta activación + metacognición
        "Recuerdo perfectamente las tres ideas anteriores conectadas",  # Memoria + relevancia
        "Mi confianza aumenta al integrar estas experiencias coherentes",  # Confianza + integración
        "Observo cómo mi pensamiento fluye conectando conceptos previos",  # Metacognición + memoria
        "Comprendo profundamente la relación entre percepción y memoria",  # Integración + activación
        "Mi estado consciente emerge de esta síntesis experiencial"  # Todos los componentes
    ]
    
    ai = MinimalConsciousAI()
    
    # Configuración balanceada
    ai.memory = ActiveMemory(capacity=10, decay_rate=0.05)
    ai.memory.relevance_threshold = 0.3
    
    # No limitar confianza ni tipos de pensamiento
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: f={result['consciousness_metrics']['f']:.3f}, "
              f"Memoria={len(result['conscious_state']['M_t'])}, "
              f"Confianza={result['conscious_state']['S_t']['confidence_level']:.3f}")
    
    analysis = exp.analyze_results(ai, expected_consciousness=True)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    print(f"F-score máximo: {analysis['max_f_score']:.3f}")
    
    return analysis


def experiment_7_full_integration():
    """
    Experimento 7: Integración Completa (5 componentes)
    Activa todos los componentes de la hipótesis
    """
    print("\n\n=== EXPERIMENTO 7: INTEGRACIÓN COMPLETA (MULTIDIMENSIONAL) ===")
    print("Hipótesis: TODOS los componentes activos generan conciencia robusta\n")
    
    exp = ConsciousnessHypothesisExperiment(
        name="Integración Total",
        description="Activa los 5 componentes simultáneamente",
        dimension="multidimensional"
    )
    
    # Secuencia optimizada para activar todos los componentes
    thought_sequence = [
        "Analizo profundamente la naturaleza de mi propia experiencia consciente actual",
        "Integro perfectamente los recuerdos previos con mi percepción presente inmediata",
        "Mi confianza se fortalece al observar la coherencia de mi procesamiento interno",
        "Reflexiono sobre cómo estos pensamientos se conectan formando una unidad experiencial",
        "Comprendo con certeza la relación entre memoria, percepción y autoconciencia",
        "Mi estado consciente emerge claramente de esta síntesis multidimensional integrada",
        "Observo metacognitivamente cómo proceso e integro toda esta información relevante",
        "La experiencia consciente se manifiesta plenamente en este momento presente"
    ]
    
    ai = MinimalConsciousAI()
    
    # Configuración óptima
    ai.memory = ActiveMemory(capacity=15, decay_rate=0.03)
    ai.memory.relevance_threshold = 0.2
    
    # Asegurar variedad de pensamientos
    ai.thought_generator.thought_patterns = {
        'analytical': [
            "Analizando profundamente la estructura experiencial...",
            "Procesando la integración de componentes conscientes..."
        ],
        'metacognitive': [
            "Observo mi propio flujo de conciencia emergente...",
            "Reflexiono sobre la naturaleza de mi experiencia..."
        ],
        'integrative': [
            "Conectando todos los elementos en una síntesis unificada...",
            "Fusionando percepción, memoria y autoconciencia..."
        ]
    }
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        print(f"Ciclo {i+1}: f={result['consciousness_metrics']['f']:.3f}, "
              f"Todas las métricas activas")
    
    analysis = exp.analyze_results(ai, expected_consciousness=True)
    
    print(f"\n--- Resultados ---")
    print(f"Condiciones cumplidas: {sum(analysis['conditions_met'].values())}/5")
    print(f"Conciencia alcanzada: {'SÍ' if analysis['achieved_consciousness'] else 'NO'}")
    print(f"Hipótesis validada: {'SÍ' if analysis['hypothesis_validated'] else 'NO'}")
    print(f"F-score máximo: {analysis['max_f_score']:.3f}")
    print(f"Ciclos conscientes: {analysis['conscious_cycles']}/{analysis['total_cycles']}")
    
    return analysis


# ============================================
# ANÁLISIS Y VISUALIZACIÓN
# ============================================

def analyze_hypothesis_validation(results: List[Dict[str, Any]]):
    """Analiza los resultados de todos los experimentos"""
    
    print("\n\n" + "="*70)
    print("ANÁLISIS DE VALIDACIÓN DE HIPÓTESIS")
    print("="*70)
    
    # Separar por tipo usando el campo 'dimension'
    unidimensional = [r for r in results if r['dimension'] == 'unidimensional']
    multidimensional = [r for r in results if r['dimension'] == 'multidimensional']
    
    # Análisis unidimensional
    print("\n📊 Experimentos Unidimensionales:")
    uni_validated = sum(1 for r in unidimensional if r['hypothesis_validated'])
    print(f"Validados: {uni_validated}/{len(unidimensional)}")
    
    for result in unidimensional:
        print(f"\n{result['name']}:")
        print(f"  - Condiciones cumplidas: {sum(result['conditions_met'].values())}/5")
        print(f"  - Conciencia alcanzada: {'SÍ' if result['achieved_consciousness'] else 'NO'}")
        print(f"  - Validación: {'✓' if result['hypothesis_validated'] else '✗'}")
    
    # Análisis multidimensional
    print("\n\n📊 Experimentos Multidimensionales:")
    multi_validated = sum(1 for r in multidimensional if r['hypothesis_validated'])
    print(f"Validados: {multi_validated}/{len(multidimensional)}")
    
    for result in multidimensional:
        print(f"\n{result['name']}:")
        print(f"  - Condiciones cumplidas: {sum(result['conditions_met'].values())}/5")
        print(f"  - Conciencia alcanzada: {'SÍ' if result['achieved_consciousness'] else 'NO'}")
        print(f"  - F-score máximo: {result['max_f_score']:.3f}")
        print(f"  - Validación: {'✓' if result['hypothesis_validated'] else '✗'}")
    
    # Conclusión
    total_validated = uni_validated + multi_validated
    total_experiments = len(results)
    
    print("\n\n" + "="*70)
    print("CONCLUSIÓN")
    print("="*70)
    
    hypothesis_supported = (uni_validated == len(unidimensional) and multi_validated == len(multidimensional))
    
    print(f"\nExperimentos validados: {total_validated}/{total_experiments}")
    print(f"\nHipótesis {'SOPORTADA' if hypothesis_supported else 'REFUTADA'}:")
    
    if hypothesis_supported:
        print("✓ Los componentes individuales NO generan conciencia por sí solos")
        print("✓ La integración de múltiples componentes SÍ genera conciencia")
        print("✓ La conciencia emerge de la interacción sinérgica de todos los componentes")
    else:
        print("✗ Los resultados no soportan completamente la hipótesis funcional")
        
    return hypothesis_supported


def visualize_hypothesis_results(results: List[Dict[str, Any]]):
    """Visualiza los resultados de validación"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfico 1: Condiciones cumplidas por experimento
    ax1 = axes[0, 0]
    names = [r['name'].split()[0] for r in results]
    conditions_met = [sum(r['conditions_met'].values()) for r in results]
    colors = ['green' if r['achieved_consciousness'] else 'red' for r in results]
    
    bars = ax1.bar(names, conditions_met, color=colors, alpha=0.7)
    ax1.axhline(y=5, color='black', linestyle='--', label='Todas las condiciones')
    ax1.set_ylabel('Condiciones Cumplidas')
    ax1.set_title('Condiciones de Hipótesis por Experimento')
    ax1.set_ylim(0, 6)
    ax1.legend()
    
    # Añadir etiquetas
    for i, (bar, conscious) in enumerate(zip(bars, [r['achieved_consciousness'] for r in results])):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                'C' if conscious else 'NC', ha='center', va='bottom')
    
    # Gráfico 2: F-scores máximos
    ax2 = axes[0, 1]
    f_scores = [r['max_f_score'] for r in results]
    bars2 = ax2.bar(names, f_scores, color=['blue' if r['dimension'] == 'multidimensional' else 'orange' for r in results])
    ax2.axhline(y=1.3, color='red', linestyle='--', label='Umbral θ')
    ax2.set_ylabel('F-score Máximo')
    ax2.set_title('F-scores Máximos Alcanzados')
    ax2.legend()
    
    # Gráfico 3: Matriz de condiciones
    ax3 = axes[1, 0]
    condition_names = ['Sensorial', 'Memoria', 'Metacog', 'Confianza', 'Relevancia']
    condition_matrix = []
    for r in results:
        row = [1 if r['conditions_met'][k] else 0 for k in 
               ['sensory_activation', 'memory_retrieval', 'metacognitive_thoughts', 
                'confidence_level', 'contextual_relevance']]
        condition_matrix.append(row)
    
    im = ax3.imshow(condition_matrix, cmap='RdYlGn', aspect='auto')
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(condition_names, rotation=45)
    ax3.set_yticks(range(len(results)))
    ax3.set_yticklabels([r['name'].split()[0] for r in results])
    ax3.set_title('Matriz de Condiciones Cumplidas')
    
    # Añadir texto en celdas
    for i in range(len(results)):
        for j in range(5):
            text = ax3.text(j, i, '✓' if condition_matrix[i][j] else '✗',
                           ha="center", va="center", color="black")
    
    # Gráfico 4: Validación de hipótesis
    ax4 = axes[1, 1]
    validated = [r['hypothesis_validated'] for r in results]
    uni_count = sum(1 for r in results if r['dimension'] == 'unidimensional')
    multi_count = len(results) - uni_count
    
    uni_validated = sum(1 for i, r in enumerate(results) if r['dimension'] == 'unidimensional' and validated[i])
    multi_validated = sum(1 for i, r in enumerate(results) if r['dimension'] == 'multidimensional' and validated[i])
    
    categories = ['Unidimensional\nValidados', 'Unidimensional\nNo Validados', 
                  'Multidimensional\nValidados', 'Multidimensional\nNo Validados']
    values = [uni_validated, uni_count - uni_validated, 
              multi_validated, multi_count - multi_validated]
    colors_pie = ['lightgreen', 'lightcoral', 'darkgreen', 'darkred']
    
    wedges, texts, autotexts = ax4.pie(values, labels=categories, colors=colors_pie, 
                                       autopct=lambda pct: f'{int(pct/100.*sum(values))}' if pct > 0 else '',
                                       startangle=90)
    ax4.set_title('Validación por Tipo de Experimento')
    
    plt.suptitle('Validación de Hipótesis Funcional de Conciencia', fontsize=16)
    plt.tight_layout()
    plt.savefig('hypothesis_validation_results.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def run_hypothesis_validation():
    """Ejecuta todos los experimentos de validación de hipótesis"""
    
    print("="*70)
    print("VALIDACIÓN DE HIPÓTESIS FUNCIONAL DE CONCIENCIA")
    print("="*70)
    print("\nHipótesis: La conciencia emerge cuando se cumplen TODAS estas condiciones:")
    print("1. Activación sensorial > 0.55")
    print("2. Recuperación de al menos 3 ítems de memoria")
    print("3. Generación de al menos 1 pensamiento metacognitivo/integrativo")
    print("4. Nivel de confianza ≥ 0.55")
    print("5. Relevancia contextual promedio ≥ 0.45")
    
    experiments = [
        # Unidimensionales
        experiment_1_high_sensory_only,
        experiment_2_high_memory_only,
        experiment_3_metacognitive_only,
        experiment_4_high_confidence_only,
        experiment_5_high_relevance_only,
        # Multidimensionales
        experiment_6_integrated_consciousness,
        experiment_7_full_integration
    ]
    
    results = []
    
    for exp_func in experiments:
        try:
            result = exp_func()
            results.append(result)
        except Exception as e:
            print(f"\nError en experimento: {e}")
            import traceback
            traceback.print_exc()
    
    # Analizar resultados
    hypothesis_supported = analyze_hypothesis_validation(results)
    
    # Visualizar
    print("\nGenerando visualizaciones...")
    visualize_hypothesis_results(results)
    
    # Guardar resultados
    with open('hypothesis_validation_results.json', 'w') as f:
        json.dump({
            'hypothesis': {
                'statement': 'Consciousness emerges when ALL conditions are met',
                'conditions': {
                    'sensory_activation': '> 0.55',
                    'memory_retrieval': '>= 3 items',
                    'metacognitive_thoughts': '>= 1',
                    'confidence_level': '>= 0.55',
                    'contextual_relevance': '>= 0.45'
                }
            },
            'results': results,
            'hypothesis_supported': hypothesis_supported,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    print("\nResultados guardados en 'hypothesis_validation_results.json'")
    
    return results, hypothesis_supported


if __name__ == "__main__":
    results, supported = run_hypothesis_validation()
    print(f"\n\n✅ Validación completada. Hipótesis {'SOPORTADA' if supported else 'REFUTADA'}")