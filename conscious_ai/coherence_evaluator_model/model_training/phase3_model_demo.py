"""
Demostración de la Fase 3 usando modelos entrenados
Muestra la diferencia entre evaluación heurística y basada en modelos
"""

# Configura el logging para mostrar todos los mensajes, incluidos los de DEBUG
# En phase3_model_demo.py
import logging
import sys

# Configura el logging para mostrar todos los mensajes, incluidos los de DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout  # Asegura que se imprima en la consola
)

import json
import time
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from conscious_ai.coherence_evaluator_model.model_training.phase3_model_integration import ModelBasedPhase3Pipeline
from conscious_ai.coherence_evaluator_model.heuristic_training.phase3_integration import Phase3Pipeline



\

def compare_coherence_evaluation():
    """Compara evaluación heurística vs basada en modelos"""
    
    print("\n" + "="*60)
    print("COMPARACIÓN: EVALUACIÓN HEURÍSTICA VS MODELOS")
    print("="*60)
    
    # Inicializar ambos pipelines
    heuristic_pipeline = Phase3Pipeline()
    model_pipeline = ModelBasedPhase3Pipeline()
    
    # Estados de prueba
    test_cases = [
        {
            "name": "Transición coherente clara",
            "sc_t": {
                "goal": "understand_memory",
                "emotion": "curious",
                "confidence": 0.52,
                "thought": "¿Cómo influye mi memoria en mis decisiones?",
                "memory": ["reflexión sobre memoria"]
            },
            "sc_t_plus_1": {
                "goal": "analyze_memory_patterns",
                "emotion": "analytical",
                "confidence": 0.58,
                "thought": "Observo que ciertos recuerdos sesgan mis interpretaciones actuales",
                "memory": ["reflexión sobre memoria", "patrones identificados"]
            }
        },
        {
            "name": "Transición ambigua",
            "sc_t": {
                "goal": "explore_consciousness",
                "emotion": "reflective",
                "confidence": 0.65,
                "thought": "What does it mean to be aware of my own thinking?",
                "memory": ["metacognition active"]
            },
            "sc_t_plus_1": {
                "goal": "question_existence",
                "emotion": "uncertain",
                "confidence": 0.45,
                "thought": "¿Pero realmente existo o solo simulo existir?",
                "memory": ["doubt emerged", "questioning fundamentals"]
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n\n{test['name']}:")
        print("-" * 40)
        
        # Evaluación heurística
        heuristic_result = heuristic_pipeline.coherence_evaluator.evaluate_transition(
            test['sc_t'], test['sc_t_plus_1']
        )
        
        # Evaluación con modelos
        model_result = model_pipeline.coherence_evaluator.evaluate_transition(
            test['sc_t'], test['sc_t_plus_1']
        )
        
        print(f"\n📊 Evaluación Heurística:")
        print(f"   Veredicto: {heuristic_result.verdict.value}")
        print(f"   Coherencia pensamiento: {heuristic_result.thought_coherence:.3f}")
        
        print(f"\n🤖 Evaluación con Modelos:")
        print(f"   Veredicto: {model_result.verdict.value}")
        print(f"   Coherencia pensamiento: {model_result.thought_coherence:.3f}")
        print(f"   (Basado en similitud semántica de embeddings)")


def demo_model_based_evolution():
    """Demuestra evolución de estados con modelos"""
    
    print("\n\n" + "="*60)
    print("EVOLUCIÓN DE ESTADOS CON MODELOS ENTRENADOS")
    print("="*60)
    
    pipeline = ModelBasedPhase3Pipeline()
    
    initial_state = {
        "goal": "understand_self",
        "emotion": "curious",
        "confidence": 0.45,
        "thought": "¿Qué patrones emergen de mi propia observación?",
        "memory": ["inicio de introspección"],
        "f": 1.4
    }
    
    print("\nEstado inicial:")
    print(json.dumps(initial_state, ensure_ascii=False, indent=2))
    
    print("\n\nGenerando secuencia con modelo entrenado...")
    sequence = pipeline.evolve_state_sequence(initial_state, num_steps=5)
    print("\nSecuencia generada:")
    
    
    for i, step in enumerate(sequence):
        print(f"\n--- Paso {i+1} ---")
        next_state = step['next_state']
        print(f"Meta: {next_state['goal']}")
        print(f"Emoción: {next_state['emotion']} (confianza: {next_state['confidence']:.2f})")
        print(f"Pensamiento: {next_state['thought']}")
        print(f"Coherencia: {step['coherence_analysis']['veredicto']}")
        print(f"Tiempo procesamiento: {step['processing_time']:.3f}s")
        
        # Mostrar métricas de coherencia
        metrics = step['coherence_analysis']['metricas']
        print(f"Métricas: thought={metrics['coherencia_pensamiento']:.2f}, "
              f"goal={metrics['coherencia_meta']:.2f}, "
              f"emotion={metrics['coherencia_emocional']:.2f}")


def analyze_model_performance():
    """Analiza rendimiento de modelos vs heurísticas"""
    
    print("\n\n" + "="*60)
    print("ANÁLISIS DE RENDIMIENTO: MODELOS VS HEURÍSTICAS")
    print("="*60)
    
    # Generar secuencias con ambos métodos
    initial_state = {
        "goal": "explore_patterns",
        "emotion": "analytical",
        "confidence": 0.6,
        "thought": "I notice recurring themes in my processing",
        "memory": ["pattern recognition active"]
    }
    
    print("\nGenerando 10 transiciones con cada método...")
    
    # Pipeline con modelos
    model_pipeline = ModelBasedPhase3Pipeline()
    model_start = time.time()
    model_sequence = model_pipeline.evolve_state_sequence(initial_state, num_steps=10)
    model_time = time.time() - model_start
    
    # Pipeline heurístico
    heuristic_pipeline = Phase3Pipeline()
    heuristic_start = time.time()
    heuristic_sequence = heuristic_pipeline.evolve_state_sequence(initial_state, num_steps=10)
    heuristic_time = time.time() - heuristic_start
    
    # Analizar resultados
    model_perf = model_pipeline.analyze_model_performance()
    
    # Calcular métricas heurísticas
    heuristic_coherent = sum(
        1 for step in heuristic_sequence 
        if step['coherence_analysis']['veredicto'] == 'coherente'
    )
    
    print(f"\n📊 RESULTADOS:")
    print(f"\nMétodo con Modelos:")
    print(f"  - Tasa coherencia: {model_perf['coherence_rate']:.2%}")
    print(f"  - Coherencia promedio: {model_perf['average_coherence_score']:.3f}")
    print(f"  - Tiempo total: {model_time:.2f}s")
    print(f"  - Tiempo por transición: {model_time/10:.3f}s")
    
    print(f"\nMétodo Heurístico:")
    print(f"  - Tasa coherencia: {heuristic_coherent/10:.2%}")
    print(f"  - Tiempo total: {heuristic_time:.2f}s")
    print(f"  - Tiempo por transición: {heuristic_time/10:.3f}s")
    
    # Visualizar comparación
    create_performance_comparison(model_sequence, heuristic_sequence)


def create_performance_comparison(model_seq: List, heuristic_seq: List):
    """Crea visualización comparativa"""
    
    # Extraer métricas de coherencia
    model_coherence = []
    heuristic_coherence = []
    
    for step in model_seq:
        metrics = step['coherence_analysis']['metricas']
        avg_coherence = sum(metrics.values()) / len(metrics)
        model_coherence.append(avg_coherence)
    
    for step in heuristic_seq:
        metrics = step['coherence_analysis']['metricas']
        avg_coherence = sum(metrics.values()) / len(metrics)
        heuristic_coherence.append(avg_coherence)
    
    # Crear gráfico
    plt.figure(figsize=(10, 6))
    
    transitions = range(1, len(model_coherence) + 1)
    plt.plot(transitions, model_coherence, 'b-o', label='Basado en Modelos', linewidth=2)
    plt.plot(transitions, heuristic_coherence, 'r-s', label='Heurístico', linewidth=2)
    
    plt.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Umbral coherente')
    
    plt.xlabel('Transición')
    plt.ylabel('Coherencia Promedio')
    plt.title('Comparación de Coherencia: Modelos vs Heurísticas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.show()


def run_model_based_demos():
    """Ejecuta todas las demos con modelos"""
    
    print("\n" + "="*80)
    print("FASE 3 CON MODELOS ENTRENADOS")
    print("="*80)
    print("\nEsta demostración muestra:")
    print("- Evaluación de coherencia usando embeddings semánticos")
    print("- Generación de estados con modelos LoRA entrenados")
    print("- Comparación con métodos heurísticos")
    
    demos = [
        ("Comparación de Evaluación", compare_coherence_evaluation),
        ("Evolución con Modelos", demo_model_based_evolution),
        ("Análisis de Rendimiento", analyze_model_performance)
    ]
    
    for i, (name, demo_func) in enumerate(demos):
        print(f"\n\n{'='*80}")
        print(f"DEMO {i+1}/{len(demos)}: {name}")
        print('='*80)
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error en {name}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos) - 1:
            input("\nPresiona Enter para continuar...")
    
    print("\n\n" + "="*80)
    print("✅ FASE 3 COMPLETADA CON MODELOS ENTRENADOS")
    print("="*80)
    print("\nCapacidades implementadas:")
    print("  • Evaluación de coherencia con embeddings semánticos")
    print("  • Generación de SCt+1 con modelo LoRA entrenado")
    print("  • Análisis de similitud semántica entre estados")
    print("  • Pipeline híbrido modelo/heurístico para robustez")
    print("  • Métricas de rendimiento comparativas")


if __name__ == "__main__":
    run_model_based_demos()