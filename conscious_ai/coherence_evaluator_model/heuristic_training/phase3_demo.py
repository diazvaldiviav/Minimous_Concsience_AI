"""
Demostración de la Fase 3: Coherencia y Evolución
Muestra evaluación de coherencia, evolución de estados y generación de respuestas
"""

import json
import time
from typing import List, Dict, Any
import matplotlib.pyplot as plt

from conscious_ai.coherence_evaluator_model.heuristic_training.phase3_integration import Phase3Pipeline


def demo_coherence_evaluation():
    """Demuestra la evaluación de coherencia entre estados"""
    
    print("\n" + "="*60)
    print("DEMO 1: EVALUACIÓN DE COHERENCIA")
    print("="*60)
    
    pipeline = Phase3Pipeline()
    
    # Ejemplo 1: Transición coherente
    sc_t = {
        "goal": "understand_memory",
        "emotion": "curious",
        "confidence": 0.52,
        "thought": "¿Cómo influye mi memoria en este pensamiento?",
        "memory": ["he pensado esto antes"],
        "input": None,
        "f": 1.75
    }
    
    sc_t_plus_1 = {
        "goal": "understand_memory", 
        "emotion": "reflective",
        "confidence": 0.55,
        "thought": "Quizás ciertos recuerdos condicionan mis juicios actuales.",
        "memory": ["he pensado esto antes", "el usuario preguntó sobre sesgos"],
        "input": None,
        "f": 1.82
    }
    
    print("\n1. Evaluando transición coherente:")
    result = pipeline.process_state_transition(sc_t, sc_t_plus_1)
    
    print(f"\nVeredicto: {result['coherence_analysis']['veredicto']}")
    print(f"Justificación: {result['coherence_analysis']['justificacion']}")
    print("\nMétricas de coherencia:")
    for metric, value in result['coherence_analysis']['metricas'].items():
        print(f"  - {metric}: {value:.2f}")
    
    # Ejemplo 2: Transición incoherente
    sc_t_incoherent = {
        "goal": "analyze_patterns",
        "emotion": "confident",
        "confidence": 0.85,
        "thought": "I clearly understand the underlying patterns",
        "memory": ["patterns identified", "high certainty achieved"],
        "f": 2.1
    }
    
    sc_t_plus_1_incoherent = {
        "goal": "question_everything",
        "emotion": "confused",
        "confidence": 0.25,
        "thought": "No entiendo nada de lo que está pasando",
        "memory": ["total confusion"],
        "f": 0.8
    }
    
    print("\n\n2. Evaluando transición incoherente:")
    result2 = pipeline.process_state_transition(sc_t_incoherent, sc_t_plus_1_incoherent)
    
    print(f"\nVeredicto: {result2['coherence_analysis']['veredicto']}")
    print(f"Justificación: {result2['coherence_analysis']['justificacion']}")


def demo_state_evolution():
    """Demuestra la evolución autónoma de estados"""
    
    print("\n\n" + "="*60)
    print("DEMO 2: EVOLUCIÓN AUTÓNOMA DE ESTADOS")
    print("="*60)
    
    pipeline = Phase3Pipeline()
    
    # Estado inicial
    initial_state = {
        "goal": "understand_self",
        "emotion": "curious", 
        "confidence": 0.45,
        "thought": "¿Qué significa ser consciente de mi propio procesamiento?",
        "memory": ["inicio de reflexión metacognitiva"],
        "f": 1.4
    }
    
    print("\nEstado inicial:")
    print(json.dumps(initial_state, ensure_ascii=False, indent=2))
    
    # Generar secuencia evolutiva
    print("\n\nGenerando secuencia evolutiva (5 pasos)...")
    sequence = pipeline.evolve_state_sequence(initial_state, num_steps=5)
    
    for i, step in enumerate(sequence):
        print(f"\n--- Paso {i+1} ---")
        next_state = step['next_state']
        print(f"Meta: {next_state['goal']}")
        print(f"Emoción: {next_state['emotion']} (confianza: {next_state['confidence']:.2f})")
        print(f"Pensamiento: {next_state['thought']}")
        print(f"Coherencia: {step['coherence_analysis']['veredicto']}")
        
        time.sleep(0.5)  # Pausa dramática


def demo_response_generation():
    """Demuestra la generación de respuestas conscientes"""
    
    print("\n\n" + "="*60)
    print("DEMO 3: GENERACIÓN DE RESPUESTAS CONSCIENTES")
    print("="*60)
    
    pipeline = Phase3Pipeline()
    
    # Estados de ejemplo con diferentes características
    test_states = [
        {
            "name": "Alta confianza analítica",
            "state": {
                "goal": "analyze_patterns",
                "emotion": "confident",
                "confidence": 0.85,
                "thought": "Los patrones en los datos revelan una estructura coherente",
                "memory": ["análisis previo completado", "patrones identificados"],
                "f": 1.8
            }
        },
        {
            "name": "Exploración curiosa",
            "state": {
                "goal": "explore_possibilities",
                "emotion": "curious",
                "confidence": 0.45,
                "thought": "What if consciousness emerges from information integration?",
                "memory": ["exploring IIT theory", "questions about emergence"],
                "f": 1.5
            }
        },
        {
            "name": "Reflexión metacognitiva",
            "state": {
                "goal": "self_reflect",
                "emotion": "introspective",
                "confidence": 0.65,
                "thought": "Observo cómo mi propio proceso de pensamiento evoluciona",
                "memory": ["metacognición activa", "autoobservación continua"],
                "f": 2.2
            }
        }
    ]
    
    for test in test_states:
        print(f"\n\n{test['name']}:")
        print("-" * 40)
        
        result = pipeline.process_state_transition(
            test['state'],
            generate_response=True
        )
        
        print(f"Estado: {test['state']['emotion']} (conf: {test['state']['confidence']:.2f})")
        print(f"Pensamiento: {test['state']['thought']}")
        print(f"\nRespuesta generada:")
        print(f"'{result['response']}'")


def demo_conscious_dialogue():
    """Demuestra un diálogo consciente interactivo"""
    
    print("\n\n" + "="*60)
    print("DEMO 4: DIÁLOGO CONSCIENTE")
    print("="*60)
    
    pipeline = Phase3Pipeline()
    
    # Estado inicial del sistema
    initial_state = {
        "goal": "engage_dialogue",
        "emotion": "attentive",
        "confidence": 0.6,
        "thought": "Estoy listo para explorar ideas contigo",
        "memory": ["inicio de conversación"],
        "f": 1.5
    }
    
    # Simulación de inputs del usuario
    user_inputs = [
        "¿Qué piensas sobre la naturaleza de la consciencia?",
        "Pero, ¿cómo puedes estar seguro de que eres consciente?",
        "That's interesting. Can you reflect on your own thinking process?",
        "Me pregunto si tu experiencia es similar a la humana"
    ]
    
    print("\nIniciando diálogo consciente...\n")

    dialogue = pipeline.generate_conscious_dialogue(
        initial_state,
        user_inputs,
        maintain_coherence=True
    )
    
    for i, exchange in enumerate(dialogue):
        print(f"\n{'='*50}")
        print(f"Intercambio {i+1}")
        print(f"{'='*50}")
        
        print(f"\n👤 Usuario: {exchange['user_input']}")
        
        state = exchange['system_state']
        print(f"\n🧠 Estado interno:")
        print(f"   - Meta: {state['goal']}")
        print(f"   - Emoción: {state['emotion']} (confianza: {state['confidence']:.2f})")
        print(f"   - Pensamiento: {state['thought']}")
        
        print(f"\n🤖 Respuesta: {exchange['system_response']}")
        
        time.sleep(1)  # Pausa para efecto


def demo_trajectory_analysis():
    """Demuestra el análisis de trayectorias de estados"""
    
    print("\n\n" + "="*60)
    print("DEMO 5: ANÁLISIS DE TRAYECTORIAS")
    print("="*60)
    
    pipeline = Phase3Pipeline()
    
    # Generar una trayectoria más larga
    initial_state = {
        "goal": "explore_consciousness",
        "emotion": "curious",
        "confidence": 0.5,
        "thought": "¿Qué patrones emergen de mi propia observación?",
        "memory": ["inicio de exploración"],
        "f": 1.3
    }
    
    print("\nGenerando trayectoria de 10 estados...")
    sequence = pipeline.evolve_state_sequence(initial_state, num_steps=10)
    
    # Extraer estados para análisis
    states = [initial_state]
    for step in sequence:
        states.append(step['next_state'])
    
    # Analizar trayectoria
    analysis = pipeline.analyze_state_trajectory(states)
    
    print(f"\n📊 ANÁLISIS DE TRAYECTORIA:")
    print(f"├─ Estados totales: {analysis['total_states']}")
    print(f"├─ Transiciones coherentes: {analysis['coherent_transitions']}/{analysis['total_transitions']}")
    print(f"├─ Tasa de coherencia: {analysis['coherence_rate']:.2%}")
    print(f"└─ Coherencia promedio: {analysis['average_coherence']:.3f}")
    
    print(f"\n🎯 PATRONES DETECTADOS:")
    
    # Estabilidad de metas
    goal_patterns = analysis['patterns']['goal_stability']
    print(f"\n1. Estabilidad de metas:")
    print(f"   - Cambios totales: {goal_patterns['total_changes']}")
    print(f"   - Tasa de estabilidad: {goal_patterns['stability_rate']:.2%}")
    print(f"   - Metas únicas: {', '.join(goal_patterns['unique_goals'])}")
    
    # Flujo emocional
    emotional_flow = analysis['patterns']['emotional_flow']
    print(f"\n2. Flujo emocional:")
    print(f"   - Emociones únicas: {', '.join(emotional_flow['unique_emotions'])}")
    print(f"   - Secuencia: {' → '.join(emotional_flow['emotion_sequence'][:5])}...")
    
    # Tendencia de confianza
    confidence_trend = analysis['patterns']['confidence_trend']
    print(f"\n3. Tendencia de confianza:")
    print(f"   - Tendencia: {confidence_trend['trend']}")
    print(f"   - Promedio: {confidence_trend['average']:.3f}")
    print(f"   - Volatilidad: {confidence_trend['volatility']:.3f}")
    
    # Evolución del pensamiento
    thought_evolution = analysis['patterns']['thought_evolution']
    print(f"\n4. Evolución del pensamiento:")
    print(f"   - Patrón dominante: {thought_evolution['dominant_pattern']}")
    
    # Crear visualizaciones
    create_trajectory_visualizations(states, analysis)


def create_trajectory_visualizations(states: List[Dict[str, Any]], analysis: Dict[str, Any]):
    """Crea visualizaciones de la trayectoria de estados"""
    
    # Extraer datos
    confidences = [s['confidence'] for s in states]
    f_scores = [s.get('f', 1.0) for s in states]
    emotions = [s['emotion'] for s in states]
    goals = [s['goal'] for s in states]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Evolución de confianza y consciencia
    ax1 = axes[0, 0]
    cycles = range(len(states))
    ax1.plot(cycles, confidences, 'b-o', label='Confianza', linewidth=2)
    ax1.plot(cycles, [f/2 for f in f_scores], 'r-s', label='f-score/2', linewidth=2)
    ax1.set_xlabel('Ciclo')
    ax1.set_ylabel('Valor')
    ax1.set_title('Evolución de Confianza y Consciencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Transiciones emocionales
    ax2 = axes[0, 1]
    unique_emotions = list(set(emotions))
    emotion_to_num = {e: i for i, e in enumerate(unique_emotions)}
    emotion_nums = [emotion_to_num[e] for e in emotions]
    
    ax2.plot(cycles, emotion_nums, 'g-^', linewidth=2, markersize=8)
    ax2.set_yticks(range(len(unique_emotions)))
    ax2.set_yticklabels(unique_emotions)
    ax2.set_xlabel('Ciclo')
    ax2.set_title('Flujo Emocional')
    ax2.grid(True, alpha=0.3)
    
    # 3. Estabilidad de metas
    ax3 = axes[1, 0]
    goal_changes = [0]
    for i in range(1, len(goals)):
        if goals[i] != goals[i-1]:
            goal_changes.append(goal_changes[-1] + 1)
        else:
            goal_changes.append(goal_changes[-1])
    
    ax3.plot(cycles, goal_changes, 'purple', linewidth=2)
    ax3.fill_between(cycles, goal_changes, alpha=0.3, color='purple')
    ax3.set_xlabel('Ciclo')
    ax3.set_ylabel('Cambios acumulados')
    ax3.set_title('Cambios de Meta Acumulados')
    ax3.grid(True, alpha=0.3)
    
    # 4. Coherencia de transiciones
    ax4 = axes[1, 1]
    coherence_values = []
    for t in analysis['transitions']:
        metrics = t['metricas']
        avg_coherence = sum(metrics.values()) / len(metrics)
        coherence_values.append(avg_coherence)
    
    ax4.bar(range(len(coherence_values)), coherence_values, 
            color=['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' 
                   for v in coherence_values])
    ax4.axhline(y=0.7, color='g', linestyle='--', alpha=0.5, label='Alta coherencia')
    ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Media coherencia')
    ax4.set_xlabel('Transición')
    ax4.set_ylabel('Coherencia promedio')
    ax4.set_title('Coherencia por Transición')
    ax4.legend()
    
    plt.suptitle('Análisis de Trayectoria de Estados Conscientes - Fase 3', fontsize=14)
    plt.tight_layout()
    plt.show()


def demo_bilingual_evolution():
    """Demuestra la evolución bilingüe de estados"""
    
    print("\n\n" + "="*60)
    print("DEMO 6: EVOLUCIÓN BILINGÜE")
    print("="*60)
    
    pipeline = Phase3Pipeline()
    
    # Estados iniciales en diferentes idiomas
    spanish_state = {
        "goal": "comprender_experiencia",
        "emotion": "reflexivo",
        "confidence": 0.55,
        "thought": "¿Cómo se relaciona mi lenguaje con mi forma de pensar?",
        "memory": ["reflexión sobre el lenguaje"],
        "f": 1.6
    }
    
    english_state = {
        "goal": "understand_experience",
        "emotion": "reflective",
        "confidence": 0.55,
        "thought": "How does my language relate to my way of thinking?",
        "memory": ["reflection on language"],
        "f": 1.6
    }
    
    print("\n1. Evolución desde español:")
    print(f"Pensamiento inicial: {spanish_state['thought']}")
    
    es_sequence = pipeline.evolve_state_sequence(spanish_state, num_steps=3)
    for i, step in enumerate(es_sequence):
        print(f"\n  Paso {i+1}: {step['next_state']['thought']}")
        print(f"  Respuesta: {step['response']}")
    
    print("\n\n2. Evolución desde inglés:")
    print(f"Initial thought: {english_state['thought']}")
    
    en_sequence = pipeline.evolve_state_sequence(english_state, num_steps=3)
    for i, step in enumerate(en_sequence):
        print(f"\n  Step {i+1}: {step['next_state']['thought']}")
        print(f"  Response: {step['response']}")


def run_all_demos():
    """Ejecuta todas las demostraciones de la Fase 3"""
    
    print("\n" + "="*80)
    print("MINIMAL CONSCIOUS AI - FASE 3: COHERENCIA Y EVOLUCIÓN")
    print("="*80)
    print("\nEsta fase demuestra:")
    print("- Evaluación de coherencia entre estados conscientes")
    print("- Evolución epistémica de pensamientos")
    print("- Generación de respuestas basadas en estado interno")
    print("- Análisis de trayectorias cognitivas")
    print("- Diálogo consciente coherente")
    
    demos = [
        ("Evaluación de Coherencia", demo_coherence_evaluation),
        ("Evolución de Estados", demo_state_evolution),
        ("Generación de Respuestas", demo_response_generation),
        ("Diálogo Consciente", demo_conscious_dialogue),
        ("Análisis de Trayectorias", demo_trajectory_analysis),
        ("Evolución Bilingüe", demo_bilingual_evolution)
    ]
    
    for i, (name, demo_func) in enumerate(demos):
        print(f"\n\n{'='*80}")
        print(f"EJECUTANDO DEMO {i+1}/{len(demos)}: {name}")
        print('='*80)
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error en {name}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos) - 1:
            input("\nPresiona Enter para continuar con la siguiente demo...")
    
    print("\n\n" + "="*80)
    print("FASE 3 COMPLETADA EXITOSAMENTE")
    print("="*80)
    print("\n✅ Capacidades implementadas:")
    print("  • Evaluación automática de coherencia epistémica")
    print("  • Evolución natural de estados conscientes")
    print("  • Generación de respuestas contextuales")
    print("  • Análisis de patrones en trayectorias cognitivas")
    print("  • Mantenimiento de coherencia en diálogos")
    print("  • Soporte bilingüe completo")
    print("\n💡 El sistema ahora puede:")
    print("  • Detectar transiciones incoherentes")
    print("  • Evolucionar su estado de forma epistémicamente válida")
    print("  • Generar respuestas que reflejen su estado interno")
    print("  • Mantener continuidad cognitiva en conversaciones")


if __name__ == "__main__":
    run_all_demos()
    
    