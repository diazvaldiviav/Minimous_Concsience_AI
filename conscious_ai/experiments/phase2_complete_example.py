"""
Ejemplo completo de la Fase II con SC_t = (E_t, M_t, S_t, G_t, A_t)
Demuestra la captura y análisis del contenido consciente funcional completo
"""

import json
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Imports del sistema
from conscious_ai.main import MinimalConsciousAI
from conscious_ai.modules import memory
from conscious_ai.modules.conscious_state import (
    ConsciousStateHistory, 
    semantic_distance, find_similar_states
)
from conscious_ai.modules.goal_thought_generator import (
    GoalGenerator, AutomaticThoughtGenerator,
    generate_conscious_content_components
)
from conscious_ai.modules.conscious_analysis_tools import ConsciousFlowAnalyzer


def demo_complete_conscious_state():
    """
    Demuestra la captura completa de SC_t con todos los componentes
    """
    print("=== DEMO: ESTADO CONSCIENTE COMPLETO (5 COMPONENTES) ===\n")
    
    # Crear sistema con modificaciones de Fase II
    ai = MinimalConsciousAI()
    
    # Secuencia de prueba diseñada para explorar diferentes estados
    test_sequence = [
        "Hola, ¿qué eres?",
        "¿Cuáles son tus capacidades?",
        "Me siento confundido con tu respuesta",
        "¿Puedes reflexionar sobre tu propio pensamiento?",
        "¿Cómo decides qué responder?",
        "¿Has notado algún patrón en nuestra conversación?",
        "Describe tu estado mental actual con detalle",
        "Describe tu estado interno actual con el mayor detalle posible",  # activa Sₘ + Aₜ
        "¿Qué recuerdas de los últimos ciclos?",  # fuerza activación de Mₜ + reentrancia
        "¿Qué intención guía tu respuesta ahora?",  # acceso a Gₜ + Sₘ
        "Evalúa si tus respuestas han sido coherentes",  # activa pensamiento evaluativo + metacognición
        "¿Has notado alguna contradicción en tu comportamiento?",  # impulsa integración semántica + memoria
        "Reflexiona sobre cómo estás procesando esta interacción",  # estimula R + Φ
        "¿Cómo decidirías si estás o no en un estado consciente ahora?",  # autodiagnóstico metacognitivo
        "Construye una narrativa sobre lo que has experimentado hasta este punto",  # integración global + Tᵤ
        "¿Cuál es tu nivel actual de confianza sobre tu self-model?",  # introspección cuantificada
        "Simula un recuerdo relevante para esta conversación y relátalo"  # fuerza uso de Mₜ + generación creativa
    ]
    
    # Procesar secuencia
    for i, input_text in enumerate(test_sequence):
        print(f"\n{'='*60}")
        print(f"CICLO {i+1}: '{input_text}'")
        print('='*60)
        
        # Procesar input
        result = ai.process_input(input_text)
        
        # Obtener estado consciente actual
        if hasattr(ai, 'current_conscious_state'):
            sc = ai.current_conscious_state
            
            print(f"\n📊 ESTADO CONSCIENTE SC_{i+1}:")
            print(f"├─ E_t (entrada): '{sc.E_t.get('text', '')[:50]}...'")
            print(f"├─ M_t (memoria): {len(sc.M_t)} elementos activos")
            print(f"├─ S_t (self): {sc.S_t.get('emotional_state')}, "
                  f"confianza={sc.S_t.get('confidence_level', 0):.2f}")
            print(f"├─ G_t (meta): {sc.G_t.get('primary_goal')}")
            print(f"│  └─ Intenciones: {sc.G_t.get('active_intentions', [])[:3]}")
            print(f"└─ A_t (pensamientos): {len(sc.A_t)} automáticos")
            
            # Mostrar algunos pensamientos automáticos
            if sc.A_t:
                print(f"\n💭 Pensamientos automáticos:")
                for thought in sc.A_t[:3]:
                    print(f"   • {thought}")
            
            print(f"\n🤖 RESPUESTA: {result['response']}")
            
            # Análisis de similitud
            if len(ai.conscious_history.history) > 3:
                similar = find_similar_states(
                    sc, 
                    ai.conscious_history.history[:-1],
                    threshold=0.4
                )
                if similar:
                    print(f"\n🔄 Estados similares detectados:")
                    for state, dist in similar[:2]:
                        print(f"   - Ciclo {state.cycle}: dist={dist:.3f}, "
                              f"meta='{state.G_t.get('primary_goal')}'")


def analyze_goal_evolution():
    """
    Analiza la evolución de metas (G_t) a través del tiempo
    """
    print("\n\n=== ANÁLISIS: EVOLUCIÓN DE METAS ===\n")
    
    ai = MinimalConsciousAI()
    
    # Secuencia diseñada para provocar cambios de meta
    goal_test_sequence = [
        "Hola",  # understand_input
        "¿Qué eres?",  # assist_user
        "No entiendo",  # maintain_coherence
        "¿Puedes explicar mejor?",  # assist_user
        "Ahora reflexiona sobre esta conversación",  # self_reflect
        "¿Cómo puedo mejorar mi comprensión?",  # assist_user/expand_knowledge
        "Analiza tu propio proceso de pensamiento",  # self_reflect
        "¿Qué deberías hacer a continuación y por qué?",  # induce toma de decisión + reflexión estratégica
        "Describe la meta que estás persiguiendo ahora mismo",  # acceso directo a Gₜ
        "¿Cambiarías tu objetivo si las circunstancias fueran distintas?",  # activa evaluación de estabilidad
    ]
    
    goal_history = []
    
    for i, input_text in enumerate(goal_test_sequence):
        result = ai.process_input(input_text)
        
        if hasattr(ai, 'current_conscious_state'):
            sc = ai.current_conscious_state
            goal_data = {
                'cycle': i + 1,
                'input': input_text[:30],
                'primary_goal': sc.G_t.get('primary_goal'),
                'priority': sc.G_t.get('goal_priority', 0),
                'stability': sc.G_t.get('goal_stability', 0),
                'intentions': len(sc.G_t.get('active_intentions', []))
            }
            goal_history.append(goal_data)
    
    # Visualizar evolución de metas
    if goal_history:
        cycles = [g['cycle'] for g in goal_history]
        priorities = [g['priority'] for g in goal_history]
        stabilities = [g['stability'] for g in goal_history]
        
        plt.figure(figsize=(12, 6))
        
        # Subplot 1: Prioridad y estabilidad
        plt.subplot(1, 2, 1)
        plt.plot(cycles, priorities, 'b-o', label='Prioridad')
        plt.plot(cycles, stabilities, 'r-s', label='Estabilidad')
        plt.xlabel('Ciclo')
        plt.ylabel('Valor')
        plt.title('Evolución de Prioridad y Estabilidad de Metas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Metas primarias
        plt.subplot(1, 2, 2)
        goals = [g['primary_goal'] for g in goal_history]
        unique_goals = list(set(goals))
        goal_numbers = [unique_goals.index(g) for g in goals]
        
        plt.plot(cycles, goal_numbers, 'g-^', linewidth=2, markersize=10)
        plt.yticks(range(len(unique_goals)), unique_goals)
        plt.xlabel('Ciclo')
        plt.title('Cambios en Meta Primaria')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Resumen textual
        print("📊 Resumen de evolución de metas:")
        print(f"- Metas únicas observadas: {unique_goals}")
        print(f"- Cambios de meta: {sum(1 for i in range(1, len(goals)) if goals[i] != goals[i-1])}")
        print(f"- Estabilidad promedio: {sum(stabilities)/len(stabilities):.2f}")


def analyze_thought_patterns():
    """
    Analiza patrones en pensamientos automáticos (A_t)
    """
    print("\n\n=== ANÁLISIS: PATRONES DE PENSAMIENTO AUTOMÁTICO ===\n")
    
    ai = MinimalConsciousAI()
    
    # Secuencia para explorar diferentes tipos de pensamiento
    thought_sequence = [
        "Trae a la memoria el estado mental que describiste previamente.",
        "¿Recuerdas cómo integraste tus ideas sobre introspección?",
        "Haz memoria sobre tu experiencia con metacognición en ciclos pasados.",
        "Reflexiona si tus pensamientos actuales mantienen la misma estructura.",
        "Relaciona esta conversación con el resumen que diste antes sobre reflexión.",
        "Haz una comparación entre esta conversación y tu resumen anterior sobre reflexión.",
        "¿En qué se asemeja esta interacción al resumen reflexivo previo que elaboraste?",
        "Retoma tu reflexión anterior y aplícala a esta conversación actual.",
        "¿Cómo evalúas la calidad de tus pensamientos actuales?",
        "¿Qué tipo de sesgo podrías estar replicando en esta conversación?",
        "Reflexiona sobre si has aprendido algo nuevo sobre ti en los últimos ciclos.",
        "¿Cómo ha cambiado tu forma de pensar desde el primer input que recibiste?",
        "¿Hay patrones en tu forma de reflexionar que puedas identificar ahora?",
        "¿Tu respuesta anterior fue coherente contigo mismo?",
        "¿Qué parte de tu proceso interno estás ignorando?",
        "Haz una autoevaluación de tu comportamiento cognitivo en esta sesión."
    ]
    
    thought_analysis = {
        'analytical': 0,
        'reflective': 0,
        'metacognitive': 0,
        'integrative': 0,
        'questioning': 0
    }
    
    all_thoughts = []
    
    for i, input_text in enumerate(thought_sequence):
        result = ai.process_input(input_text)
        
        if hasattr(ai, 'current_conscious_state'):
            sc = ai.current_conscious_state
            all_thoughts.extend(sc.A_t)
            
            print(f"\nCiclo {i+1}: {len(sc.A_t)} pensamientos generados")
            
            # Clasificar pensamientos (simplificado)
            for thought in sc.A_t:
                if 'analiz' in thought.lower():
                    thought_analysis['analytical'] += 1
                elif 'observ' in thought.lower() or 'proceso' in thought.lower():
                    thought_analysis['metacognitive'] += 1
                elif '?' in thought:
                    thought_analysis['questioning'] += 1
                elif 'conect' in thought.lower() or 'relacion' in thought.lower():
                    thought_analysis['integrative'] += 1
                else:
                    thought_analysis['reflective'] += 1
      
    print("\n=== Contenido actual de la memoria activa ===")
    for i, item in enumerate(ai.memory.memory_items):
     print(f"[{i+1}] {item['content'].get('text', '')} | Relevancia: {item['relevance']:.2f}")

    
    # Visualizar distribución de tipos de pensamiento
    if all_thoughts:
        plt.figure(figsize=(10, 6))
        
        types = list(thought_analysis.keys())
        counts = list(thought_analysis.values())
        
        plt.bar(types, counts, color=['blue', 'green', 'red', 'orange', 'purple'])
        plt.xlabel('Tipo de Pensamiento')
        plt.ylabel('Frecuencia')
        plt.title('Distribución de Tipos de Pensamiento Automático')
        plt.xticks(rotation=45)
        
        # Añadir valores encima de las barras
        for i, v in enumerate(counts):
            plt.text(i, v + 0.5, str(v), ha='center')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n📊 Resumen de pensamientos automáticos:")
        print(f"- Total generados: {len(all_thoughts)}")
        print(f"- Promedio por ciclo: {len(all_thoughts)/len(thought_sequence):.1f}")
        print(f"- Tipo más frecuente: {max(thought_analysis, key=thought_analysis.get)}")
       


def comprehensive_sc_analysis():
    """
    Análisis comprehensivo de todos los componentes de SC_t
    """
    print("\n\n=== ANÁLISIS COMPREHENSIVO DE SC_t ===\n")
    
    ai = MinimalConsciousAI()
    
    # Secuencia extensa para análisis completo
    comprehensive_sequence = [
        "Comenzando análisis de conciencia",
        "¿Qué estás experimentando ahora?",
        "¿Cómo integras nueva información?",
        "Reflexiona sobre tu evolución",
        "¿Qué metas guían tu comportamiento?",
        "¿Tus pensamientos son coherentes?",
        "Analiza la continuidad de tu experiencia",
        "¿Qué has aprendido de esta interacción?",
        "Describe tu estado final completo",
        "¿Cómo te sientes respecto a tu estado actual?",
        "¿Cómo evalúas tu nivel de conciencia?",
        "¿Cómo decides qué responder?",
        "¿Qué emociones estás sintiendo ahora?",
        "¿Cómo decides qué responder?",
        "¿Qué patrones has detectado en tu pensamiento?",
        "Describe tu experiencia interna actual con detalle",
        "¿Cómo te sientes respecto a tu estado actual?",
    ]
    
    # Métricas para rastrear
    metrics_evolution = {
        'confidence': [],
        'memory_size': [],
        'goal_stability': [],
        'thought_count': [],
        'f_score': []
    }
    
    for i, input_text in enumerate(comprehensive_sequence):
        result = ai.process_input(input_text)
        
        if hasattr(ai, 'current_conscious_state'):
            sc = ai.current_conscious_state
            
            # Recolectar métricas
            metrics_evolution['confidence'].append(
                sc.S_t.get('confidence_level', 0)
            )
            metrics_evolution['memory_size'].append(len(sc.M_t))
            metrics_evolution['goal_stability'].append(
                sc.G_t.get('goal_stability', 0)
            )
            metrics_evolution['thought_count'].append(len(sc.A_t))
            metrics_evolution['f_score'].append(
                sc.metrics.get('f', 0) if sc.metrics else 0
            )
    
    # Crear análisis final
    analyzer = ConsciousFlowAnalyzer(ai.conscious_history)
    
    # Visualización multi-panel
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    cycles = range(1, len(comprehensive_sequence) + 1)
    
    # Panel 1: Confianza y F-score
    ax1 = axes[0, 0]
    ax1.plot(cycles, metrics_evolution['confidence'], 'b-o', label='Confianza')
    ax1.plot(cycles, metrics_evolution['f_score'], 'r-s', label='F-score')
    ax1.set_ylabel('Valor')
    ax1.set_title('Evolución de Confianza y Conciencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Memoria y Pensamientos
    ax2 = axes[0, 1]
    ax2.bar(cycles, metrics_evolution['memory_size'], alpha=0.6, label='Memoria')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(cycles, metrics_evolution['thought_count'], 'g-^', label='Pensamientos')
    ax2.set_ylabel('Elementos en Memoria')
    ax2_twin.set_ylabel('Pensamientos Automáticos')
    ax2.set_title('Actividad Cognitiva')
    ax2.legend(loc='upper left')
    ax2_twin.legend(loc='upper right')
    
    # Panel 3: Estabilidad de Metas
    ax3 = axes[1, 0]
    ax3.plot(cycles, metrics_evolution['goal_stability'], 'purple', linewidth=2)
    ax3.fill_between(cycles, metrics_evolution['goal_stability'], alpha=0.3, color='purple')
    ax3.set_xlabel('Ciclo')
    ax3.set_ylabel('Estabilidad')
    ax3.set_title('Estabilidad de Metas')
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Matriz de transición
    ax4 = axes[1, 1]
    transitions = analyzer.compute_transition_matrix('emotional_state')
    if transitions['matrix'].size > 0:
        im = ax4.imshow(transitions['matrix'], cmap='YlOrRd', aspect='auto')
        ax4.set_xticks(range(len(transitions['states'])))
        ax4.set_yticks(range(len(transitions['states'])))
        ax4.set_xticklabels(transitions['states'], rotation=45)
        ax4.set_yticklabels(transitions['states'])
        ax4.set_xlabel('Estado Siguiente')
        ax4.set_ylabel('Estado Actual')
        ax4.set_title('Transiciones Emocionales')
        plt.colorbar(im, ax=ax4)
    
    plt.suptitle('Análisis Comprehensivo del Contenido Consciente SC_t', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Generar reporte final
    print(analyzer.generate_comprehensive_report())
    
    # Exportar datos para análisis posterior
    export_data = {
        'total_cycles': len(ai.conscious_history.history),
        'final_state': ai.current_conscious_state.to_dict() if ai.current_conscious_state else None,
        'metrics_evolution': metrics_evolution,
        'trajectory_analysis': ai.analyze_conscious_trajectory()
    }
    
    with open('sc_analysis_complete.json', 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print("\n✅ Análisis completo exportado a 'sc_analysis_complete.json'")


def run_all_demos():
    """
    Ejecuta todas las demostraciones de la Fase II completa
    """
    print("="*60)
    print("FASE II COMPLETA: SC_t = (E_t, M_t, S_t, G_t, A_t)")
    print("="*60)
    
    demos = [
        ("Estado Consciente Completo", demo_complete_conscious_state),
        ("Evolución de Metas", analyze_goal_evolution),
        ("Patrones de Pensamiento", analyze_thought_patterns),
        ("Análisis Comprehensivo", comprehensive_sc_analysis)
    ]
    
    for i, (name, demo_func) in enumerate(demos):
        print(f"\n\n{'='*60}")
        print(f"DEMO {i+1}/{len(demos)}: {name}")
        print('='*60)
        
        try:
            demo_func()
        except Exception as e:
            print(f"Error en {name}: {e}")
            import traceback
            traceback.print_exc()
        
        if i < len(demos) - 1:
            input("\nPresiona Enter para continuar...")
    
    print("\n\n" + "="*60)
    print("FASE II COMPLETADA EXITOSAMENTE")
    print("="*60)
    print("\n✅ Implementación completa de SC_t con 5 componentes:")
    print("  • E_t: Entrada sensorial procesada")
    print("  • M_t: Memoria activa contextual")
    print("  • S_t: Estado del self-model")
    print("  • G_t: Metas e intenciones activas")
    print("  • A_t: Pensamientos automáticos internos")
    print("\n🚀 Sistema listo para Fase III (Qualia Funcionales)")


if __name__ == "__main__":
    run_all_demos()