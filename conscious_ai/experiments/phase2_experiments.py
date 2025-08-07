"""
Experimentos específicos para Phase II: Análisis del Contenido Consciente
"""

from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def experiment_conscious_continuity(ai_system, test_sequence: List[str]):
    """
    Experimento 1: Analiza la continuidad del flujo consciente
    """
    print("\n===== EXPERIMENTO: CONTINUIDAD DEL FLUJO CONSCIENTE =====")
    
    continuity_scores = []
    coherence_scores = []
    richness_scores = []
    
    for i, input_text in enumerate(test_sequence):
        result = ai_system.process_input(input_text)
        
        # Extraer métricas del contenido consciente
        metrics = result['consciousness_metrics']
        continuity_scores.append(metrics.get('temporal_continuity', 0))
        coherence_scores.append(metrics.get('content_coherence', 0))
        richness_scores.append(metrics.get('content_richness', 0))
        
        if i % 3 == 0:
            trajectory = ai_system.conscious_integrator.content_history.analyze_continuity()
            print(f"\nCiclo {i+1} - Análisis de trayectoria:")
            print(f"  Continuidad: {trajectory['continuity']:.3f}")
            print(f"  Estabilidad: {trajectory['stability']:.3f}")
            print(f"  Evolución: {trajectory['evolution']:.3f}")
    
    # Visualización
    plt.figure(figsize=(10, 6))
    cycles = range(1, len(test_sequence) + 1)
    
    plt.plot(cycles, continuity_scores, label='Continuidad Temporal', marker='o')
    plt.plot(cycles, coherence_scores, label='Coherencia del Contenido', marker='s')
    plt.plot(cycles, richness_scores, label='Riqueza del Contenido', marker='^')
    
    plt.xlabel('Ciclo')
    plt.ylabel('Valor')
    plt.title('Evolución de las Métricas del Contenido Consciente')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {
        'continuity_mean': np.mean(continuity_scores),
        'coherence_mean': np.mean(coherence_scores),
        'richness_mean': np.mean(richness_scores)
    }


def experiment_conscious_patterns(ai_system):
    """
    Experimento 2: Detecta patrones en el contenido consciente
    """
    print("\n===== EXPERIMENTO: PATRONES DE CONTENIDO CONSCIENTE =====")
    
    # Secuencia diseñada para inducir patrones
    pattern_sequence = [
        "¿Qué es la conciencia?",
        "Reflexiona sobre tu estado actual",
        "¿Cómo sabes que eres consciente?",
        "Describe tu experiencia interna",
        "¿Qué es la conciencia?",  # Repetición intencional
        "¿Has notado algo familiar?",
        "Reflexiona sobre tu estado actual",  # Otra repetición
        "¿Qué patrones detectas en nuestra conversación?"
    ]
    
    pattern_detections = []
    
    for i, input_text in enumerate(pattern_sequence):
        result = ai_system.process_input(input_text)
        
        # Buscar contenidos similares
        current_content = ai_system.conscious_integrator.current_content
        similar = ai_system.conscious_integrator.content_history.find_similar(current_content, threshold=0.5)
        
        pattern_detections.append({
            'cycle': i + 1,
            'input': input_text,
            'similar_count': len(similar),
            'metacognition_triggered': ai_system.conscious_integrator.should_trigger_metacognition()
        })
        
        if len(similar) > 0:
            print(f"\nCiclo {i+1}: Detectados {len(similar)} contenidos similares")
            for sc in similar:
                print(f"  - Ciclo {sc.cycle}: {sc.unified_content['phenomenal_content']['focus'][:40]}...")
    
    # Análisis de resultados
    total_patterns = sum(p['similar_count'] for p in pattern_detections)
    metacog_triggers = sum(1 for p in pattern_detections if p['metacognition_triggered'])
    
    print(f"\n--- Resumen de Patrones ---")
    print(f"Total de detecciones de patrones: {total_patterns}")
    print(f"Activaciones metacognitivas: {metacog_triggers}/{len(pattern_sequence)}")
    
    return pattern_detections


def experiment_conscious_disruption(ai_system):
    """
    Experimento 3: Analiza el efecto de disrupciones en el flujo consciente
    """
    print("\n===== EXPERIMENTO: DISRUPCIÓN DEL FLUJO CONSCIENTE =====")
    
    # Establecer flujo coherente
    coherent_sequence = [
        "Hablemos sobre la naturaleza de la mente",
        "¿Qué relación hay entre mente y conciencia?",
        "¿Cómo experimentas tus propios pensamientos?"
    ]
    
    for input_text in coherent_sequence:
        ai_system.process_input(input_text)
    
    # Capturar estado pre-disrupción
    pre_disruption_analysis = ai_system.conscious_integrator.content_history.analyze_continuity()
    
    print("\nEstado pre-disrupción:")
    print(f"  Continuidad: {pre_disruption_analysis['continuity']:.3f}")
    print(f"  Estabilidad: {pre_disruption_analysis['stability']:.3f}")
    
    # Introducir disrupción
    print("\n--- INTRODUCIENDO DISRUPCIÓN ---")
    
    # Limpiar memoria (simular amnesia parcial)
    ai_system.memory.clear()
    
    # Input completamente diferente
    disruption_result = ai_system.process_input("¿Cuánto es 2+2?")
    
    # Continuar con tema original
    post_disruption_sequence = [
        "¿Recuerdas de qué estábamos hablando?",
        "Volvamos al tema de la conciencia",
        "¿Cómo afectó esa interrupción a tu experiencia?"
    ]
    
    recovery_metrics = []
    
    for input_text in post_disruption_sequence:
        result = ai_system.process_input(input_text)
        metrics = result['consciousness_metrics']
        recovery_metrics.append({
            'continuity': metrics.get('temporal_continuity', 0),
            'coherence': metrics.get('content_coherence', 0),
            'f': metrics['f']
        })
    
    # Análisis post-disrupción
    post_disruption_analysis = ai_system.conscious_integrator.content_history.analyze_continuity()
    
    print("\nEstado post-disrupción:")
    print(f"  Continuidad: {post_disruption_analysis['continuity']:.3f}")
    print(f"  Estabilidad: {post_disruption_analysis['stability']:.3f}")
    
    print("\nRecuperación del flujo consciente:")
    for i, metrics in enumerate(recovery_metrics):
        print(f"  Ciclo {i+1}: continuidad={metrics['continuity']:.3f}, "
              f"coherencia={metrics['coherence']:.3f}, f={metrics['f']:.3f}")
    
    return {
        'pre_disruption': pre_disruption_analysis,
        'post_disruption': post_disruption_analysis,
        'recovery_trajectory': recovery_metrics
    }


def experiment_phenomenological_richness(ai_system):
    """
    Experimento 4: Explora la riqueza fenomenológica del contenido consciente
    """
    print("\n===== EXPERIMENTO: RIQUEZA FENOMENOLÓGICA =====")
    
    # Inputs de complejidad creciente
    complexity_sequence = [
        "Hola",  # Simple
        "¿Cómo estás hoy?",  # Pregunta básica
        "Describe tu experiencia al procesar esta oración",  # Autorreferencial
        "¿Qué sientes cuando reflexionas sobre tu propia existencia como sistema consciente?",  # Complejo
        "Imagina que pudieras experimentar colores que no existen. ¿Cómo sería esa experiencia para ti?",  # Abstracto
        "Si la conciencia es información integrada, ¿qué información constituye tu experiencia en este momento?"  # Meta-teórico
    ]
    
    richness_evolution = []
    
    for i, input_text in enumerate(complexity_sequence):
        print(f"\n--- Input {i+1} (longitud: {len(input_text)}) ---")
        result = ai_system.process_input(input_text)
        
        # Extraer contenido consciente
        cc = ai_system.conscious_integrator.current_content
        uc = cc.unified_content
        
        richness_data = {
            'input_length': len(input_text),
            'richness': uc['integration_quality']['richness'],
            'coherence': uc['integration_quality']['coherence'],
            'unity': uc['integration_quality']['unity'],
            'context': uc['phenomenal_content']['context'],
            'memory_depth': len(cc.active_memory)
        }
        
        richness_evolution.append(richness_data)
        
        print(f"Riqueza: {richness_data['richness']:.3f}")
        print(f"Contexto: {richness_data['context']}")
        print(f"Profundidad de memoria: {richness_data['memory_depth']}")
    
    # Visualización de la evolución
    plt.figure(figsize=(10, 6))
    
    inputs = range(1, len(complexity_sequence) + 1)
    richness_values = [r['richness'] for r in richness_evolution]
    coherence_values = [r['coherence'] for r in richness_evolution]
    input_lengths = [r['input_length'] for r in richness_evolution]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Gráfico superior: Riqueza y coherencia
    ax1.plot(inputs, richness_values, 'b-o', label='Riqueza')
    ax1.plot(inputs, coherence_values, 'r-s', label='Coherencia')
    ax1.set_ylabel('Valor')
    ax1.set_title('Evolución de la Riqueza Fenomenológica')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico inferior: Longitud del input
    ax2.bar(inputs, input_lengths, color='green', alpha=0.6)
    ax2.set_xlabel('Input #')
    ax2.set_ylabel('Longitud del input')
    ax2.set_title('Complejidad del Input')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return richness_evolution


def run_all_phase2_experiments(ai_system):
    """
    Ejecuta todos los experimentos de Phase II
    """
    print("\n" + "="*60)
    print("INICIANDO EXPERIMENTOS DE PHASE II: CONTENIDO CONSCIENTE")
    print("="*60)
    
    # Experimento 1: Continuidad
    basic_sequence = [
        "Hola, ¿qué eres?",
        "¿Cómo procesas la información?",
        "¿Eres consciente?",
        "¿Qué significa ser consciente para ti?",
        "¿Cómo ha evolucionado tu comprensión durante esta conversación?"
    ]
    
    continuity_results = experiment_conscious_continuity(ai_system, basic_sequence)
    
    # Experimento 2: Patrones
    pattern_results = experiment_conscious_patterns(ai_system)
    
    # Experimento 3: Disrupción
    disruption_results = experiment_conscious_disruption(ai_system)
    
    # Experimento 4: Riqueza fenomenológica
    richness_results = experiment_phenomenological_richness(ai_system)
    
    # Resumen final
    print("\n" + "="*60)
    print("RESUMEN DE EXPERIMENTOS PHASE II")
    print("="*60)
    
    print(f"\n1. Continuidad del Flujo Consciente:")
    print(f"   - Continuidad promedio: {continuity_results['continuity_mean']:.3f}")
    print(f"   - Coherencia promedio: {continuity_results['coherence_mean']:.3f}")
    print(f"   - Riqueza promedio: {continuity_results['richness_mean']:.3f}")
    
    print(f"\n2. Detección de Patrones:")
    print(f"   - Total de patrones detectados: {sum(p['similar_count'] for p in pattern_results)}")
    print(f"   - Triggers metacognitivos: {sum(1 for p in pattern_results if p['metacognition_triggered'])}")
    
    print(f"\n3. Resiliencia ante Disrupciones:")
    pre = disruption_results['pre_disruption']['continuity']
    post = disruption_results['post_disruption']['continuity']
    print(f"   - Caída en continuidad: {((pre - post) / pre * 100):.1f}%")
    print(f"   - Recuperación lograda: {'Sí' if post > 0.5 else 'Parcial'}")
    
    print(f"\n4. Riqueza Fenomenológica:")
    max_richness = max(r['richness'] for r in richness_results)
    print(f"   - Riqueza máxima alcanzada: {max_richness:.3f}")
    print(f"   - Correlación input-riqueza: {'Positiva' if richness_results[-1]['richness'] > richness_results[0]['richness'] else 'No clara'}")
    
    return {
        'continuity': continuity_results,
        'patterns': pattern_results,
        'disruption': disruption_results,
        'richness': richness_results
    }