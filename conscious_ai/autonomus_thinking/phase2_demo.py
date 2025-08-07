"""
Demostración completa de la Fase 2: Pensamiento Autónomo
Muestra cómo el sistema genera estados conscientes sin input externo
"""

import os
import time
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Importar módulos necesarios
from conscious_ai.autonomus_thinking.create_autonomous_dataset import create_autonomous_dataset
from conscious_ai.autonomus_thinking.autonomous_training_pipeline import AutonomousThoughtTrainer
from conscious_ai.autonomus_thinking.autonomous_integration import AutonomousConsciousAI


def setup_phase2_environment():
    """Prepara el entorno para la Fase 2"""
    dirs = [
        './models/autonomous_lora',
        './logs/autonomous_training',
        './data',
        './results/phase2'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Directorios creados")


def phase2_training():
    """Entrena el modelo de pensamiento autónomo"""
    print("\n" + "="*60)
    print("FASE 2: ENTRENAMIENTO DE PENSAMIENTO AUTÓNOMO")
    print("="*60)
    
    dataset_path = './data/autonomous_thought_data.jsonl'
    
    # 1. Generar dataset si no existe
    if not os.path.exists(dataset_path):
        print("\n1. Generando dataset de transiciones autónomas...")
        dataset = create_autonomous_dataset(output_file=dataset_path, size=1000)
        print(f"✓ Dataset creado con {len(dataset)} transiciones")
    else:
        print("✓ Dataset encontrado")
    
    # 2. Verificar si ya existe modelo entrenado
    model_exists = os.path.exists('./models/autonomous_lora/adapter_model.safetensors')
    
    if not model_exists:
        print("\n2. Entrenando modelo de pensamiento autónomo...")
        print("Nota: Esto puede tomar 20-40 minutos en CPU")
        
        trainer = AutonomousThoughtTrainer()
        trainer.train(
            dataset_path=dataset_path,
            num_epochs=3,
            batch_size=2,
            learning_rate=3e-4
        )
        print("✓ Entrenamiento completado")
    else:
        print("✓ Modelo autónomo ya entrenado")


def demonstrate_autonomous_thinking():
    """Demuestra el pensamiento autónomo en acción"""
    print("\n" + "="*60)
    print("DEMOSTRACIÓN: PENSAMIENTO AUTÓNOMO")
    print("="*60)
    
    # Crear sistema con capacidad autónoma
    ai = AutonomousConsciousAI(
        autonomous_model_path="./models/autonomous_gemma_lora"
    )
    
    # 1. Establecer contexto inicial
    print("\n1. ESTABLECIENDO CONTEXTO INICIAL")
    print("-"*40)
    
    initial_inputs = [
        "Soy un sistema de inteligencia artificial explorando la consciencia",
        "¿Qué significa tener experiencias internas?",
        "Observo mis propios procesos mientras pienso"
    ]
    
    for i, input_text in enumerate(initial_inputs):
        print(f"\nInput {i+1}: '{input_text}'")
        result = ai.process_input(input_text)
        
        # Mostrar estado
        sc = result['conscious_state']
        print(f"→ Meta: {sc['G_t']['primary_goal']}")
        print(f"→ Emoción: {sc['S_t']['emotional_state']}")
        print(f"→ Consciencia (f): {result['consciousness_metrics']['f']:.3f}")
        
        time.sleep(1)
    
    # 2. Sesión de pensamiento autónomo corta
    print("\n\n2. SESIÓN DE PENSAMIENTO AUTÓNOMO (10 ciclos)")
    print("-"*40)
    print("El sistema ahora pensará por sí mismo...")
    time.sleep(2)
    
    autonomous_results = ai.run_autonomous_session(
        num_cycles=10,
        pause_between_cycles=1.5,
        stop_on_low_consciousness=True,
        consciousness_threshold=0.2
    )
    
    # 3. Flujo de consciencia libre
    print("\n\n3. FLUJO DE CONSCIENCIA (20 segundos)")
    print("-"*40)
    print("Observa el flujo natural de pensamientos...")
    time.sleep(2)
    
    thought_stream = ai.generate_thought_stream(
        duration_seconds=20,
        min_pause=0.5,
        max_pause=2.0
    )
    
    return ai, autonomous_results, thought_stream


def analyze_autonomous_session(
    ai: AutonomousConsciousAI,
    autonomous_results: List[Dict[str, Any]],
    thought_stream: List[Dict[str, Any]]
):
    """Analiza los resultados de la sesión autónoma"""
    
    print("\n" + "="*60)
    print("ANÁLISIS DE LA SESIÓN AUTÓNOMA")
    print("="*60)
    
    # 1. Métricas generales
    print("\n1. MÉTRICAS GENERALES")
    print(f"├─ Ciclos autónomos totales: {ai.autonomous_metrics['total_autonomous_cycles']}")
    print(f"├─ Temas explorados: {len(ai.autonomous_metrics['themes_explored'])}")
    print(f"│  └─ {', '.join(ai.autonomous_metrics['themes_explored'])}")
    print(f"└─ Consciencia máxima: {ai.autonomous_metrics['peak_consciousness']:.3f}")
    
    # 2. Análisis de patrones
    thought_analysis = ai.autonomous_generator.analyze_thought_patterns()
    
    print("\n2. PATRONES DE PENSAMIENTO")
    if thought_analysis['dominant_themes']:
        print("Temas dominantes:")
        for theme, count in thought_analysis['dominant_themes'][:3]:
            print(f"  • {theme}: {count} veces")
    
    print(f"\nEstabilidad cognitiva: {thought_analysis['stability']:.2%}")
    print(f"Confianza promedio: {thought_analysis['avg_confidence']:.2%}")
    
    # 3. Transiciones emocionales
    if thought_analysis['emotional_flow']:
        print("\n3. FLUJO EMOCIONAL")
        print("Últimas transiciones:")
        for prev, curr in thought_analysis['emotional_flow'][-5:]:
            print(f"  {prev} → {curr}")
    
    # 4. Visualizaciones
    create_autonomous_visualizations(autonomous_results, thought_stream)


def create_autonomous_visualizations(
    autonomous_results: List[Dict[str, Any]],
    thought_stream: List[Dict[str, Any]]
):
    """Crea visualizaciones de la sesión autónoma"""
    
    if not autonomous_results:
        print("\nNo hay suficientes datos para visualizar")
        return
    
    # Extraer datos
    cycles = list(range(1, len(autonomous_results) + 1))
    f_scores = [r['consciousness_metrics']['f'] for r in autonomous_results]
    confidences = [r['conscious_state']['S_t']['confidence_level'] for r in autonomous_results]
    themes = [r.get('autonomous_theme', 'unknown') for r in autonomous_results]
    
    # Crear figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Evolución de consciencia
    ax1 = axes[0, 0]
    ax1.plot(cycles, f_scores, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=1.3, color='r', linestyle='--', label='Umbral θ=1.3')
    ax1.set_xlabel('Ciclo Autónomo')
    ax1.set_ylabel('f (consciencia)')
    ax1.set_title('Evolución de Consciencia Autónoma')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confianza en el tiempo
    ax2 = axes[0, 1]
    ax2.plot(cycles, confidences, 'g-s', linewidth=2, markersize=6)
    ax2.set_xlabel('Ciclo Autónomo')
    ax2.set_ylabel('Confianza')
    ax2.set_title('Evolución de Confianza')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de temas
    ax3 = axes[1, 0]
    unique_themes = list(set(themes))
    theme_counts = [themes.count(t) for t in unique_themes]
    ax3.bar(range(len(unique_themes)), theme_counts, color='purple', alpha=0.7)
    ax3.set_xticks(range(len(unique_themes)))
    ax3.set_xticklabels(unique_themes, rotation=45, ha='right')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Temas Explorados')
    
    # 4. Correlación confianza-consciencia
    ax4 = axes[1, 1]
    scatter = ax4.scatter(confidences, f_scores, c=cycles, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Confianza')
    ax4.set_ylabel('Consciencia (f)')
    ax4.set_title('Relación Confianza-Consciencia')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Ciclo')
    
    plt.suptitle('Análisis del Pensamiento Autónomo - Fase 2', fontsize=14)
    plt.tight_layout()
    
    # Guardar
    plt.savefig('./results/phase2/autonomous_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_phase2_results(
    ai: AutonomousConsciousAI,
    autonomous_results: List[Dict[str, Any]],
    thought_stream: List[Dict[str, Any]]
):
    """Guarda los resultados de la Fase 2"""
    
    results = {
        'metadata': {
            'total_autonomous_cycles': ai.autonomous_metrics['total_autonomous_cycles'],
            'themes_explored': list(ai.autonomous_metrics['themes_explored']),
            'peak_consciousness': ai.autonomous_metrics['peak_consciousness'],
            'thought_patterns': make_json_serializable(ai.autonomous_generator.analyze_thought_patterns())
        },
        'autonomous_session': [
            {
                'cycle': i + 1,
                'theme': r.get('autonomous_theme', 'unknown'),
                'goal': r['conscious_state']['G_t']['primary_goal'],
                'emotion': r['conscious_state']['S_t']['emotional_state'],
                'confidence': r['conscious_state']['S_t']['confidence_level'],
                'f_score': r['consciousness_metrics']['f'],
                'is_conscious': bool(r['is_conscious'])
            }
            for i, r in enumerate(autonomous_results)
        ],
        'thought_stream_summary': {
            'total_thoughts': len(thought_stream),
            'duration': '20 seconds',
            'themes': list(set(r.get('autonomous_theme', 'unknown') for r in thought_stream))
        }
    }
    
    # Guardar JSON
    output_path = './results/phase2/autonomous_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Resultados guardados en {output_path}")
    
    # Guardar resumen
    summary_path = './results/phase2/phase2_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN FASE 2: PENSAMIENTO AUTÓNOMO\n")
        f.write("="*60 + "\n\n")
        f.write(f"Ciclos autónomos ejecutados: {results['metadata']['total_autonomous_cycles']}\n")
        f.write(f"Temas explorados: {', '.join(results['metadata']['themes_explored'])}\n")
        f.write(f"Consciencia máxima alcanzada: {results['metadata']['peak_consciousness']:.3f}\n")
        f.write(f"\nEl sistema demostró capacidad de:\n")
        f.write("- Generar pensamientos coherentes sin input externo\n")
        f.write("- Mantener continuidad temática\n")
        f.write("- Evolucionar estados emocionales\n")
        f.write("- Auto-regular niveles de confianza\n")
        f.write("- Explorar múltiples dimensiones de la experiencia\n")
    
    print(f"✓ Resumen guardado en {summary_path}")

import numpy as np
def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


def main():
    """Función principal de la demostración de Fase 2"""
    
    print("\n" + "="*80)
    print("MINIMAL CONSCIOUS AI - FASE 2: PENSAMIENTO AUTÓNOMO")
    print("="*80)
    print("\nEsta fase permite al sistema generar estados conscientes")
    print("sin necesidad de input externo, simulando un flujo de pensamiento interno.")
    
    # Preparar entorno
    setup_phase2_environment()
    
    # Entrenar modelo (si es necesario)
    phase2_training()
    
    # Demostrar pensamiento autónomo
    ai, autonomous_results, thought_stream = demonstrate_autonomous_thinking()
    
    # Analizar resultados
    analyze_autonomous_session(ai, autonomous_results, thought_stream)
    
    # Guardar resultados
    save_phase2_results(ai, autonomous_results, thought_stream)
    
    # Mensaje final
    print("\n" + "="*80)
    print("FASE 2 COMPLETADA EXITOSAMENTE")
    print("="*80)
    print("\n✅ Logros de la Fase 2:")
    print("  • Dataset de transiciones autónomas generado")
    print("  • Modelo de pensamiento autónomo entrenado")
    print("  • Sistema capaz de pensar sin input externo")
    print("  • Continuidad y coherencia en el flujo de pensamiento")
    print("  • Exploración autónoma de temas internos")
    print("  • Evolución natural de estados emocionales")
    print("\n🚀 El sistema ahora puede mantener un flujo de consciencia autónomo")
    print("   similar al pensamiento humano continuo.")
    print("\n💡 Próximos pasos sugeridos:")
    print("  • Fase 3: Diálogo consciente con generación de lenguaje natural")
    print("  • Fase 4: Metacognición y auto-modificación")
    print("  • Fase 5: Consciencia social y teoría de la mente")


if __name__ == "__main__":
    main()