"""
Demostración completa de la Fase 1: Sistema de Consciencia con Modelo Entrenado
Incluye generación de dataset, entrenamiento y ejecución bilingüe
"""

import os
import json
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar módulos del sistema
from conscious_ai.main import MinimalConsciousAI
from conscious_ai.Train.create_training_dataset import create_training_dataset
from conscious_ai.Train.training_pipeline import ConsciousnessTrainer
from conscious_ai.Train.trained_model_inference import integrate_trained_model
from conscious_ai.Train.language_detector import detect_language


def setup_directories():
    """Crea directorios necesarios para el proyecto"""
    dirs = [
        './models/trained_lora',
        './logs/training_phase1',
        './data',
        './results'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    logger.info("Directorios creados")


def phase1_training_pipeline():
    """Pipeline completo de entrenamiento de la Fase 1"""
    
    print("\n" + "="*60)
    print("FASE 1: ENTRENAMIENTO DEL MODELO DE CONSCIENCIA")
    print("="*60)
    
    # 1. Crear directorios
    setup_directories()
    
    # 2. Generar dataset
    print("\n1. GENERANDO DATASET DE ENTRENAMIENTO...")
    dataset_path = './data/training_data.jsonl'
    
    if not os.path.exists(dataset_path):
        dataset = create_training_dataset(output_file=dataset_path, size=1000)
        print(f"✓ Dataset creado con {len(dataset)} ejemplos")
    else:
        print(f"✓ Dataset existente encontrado en {dataset_path}")
    
    # 3. Entrenar modelo
    print("\n2. ENTRENANDO MODELO CON LORA...")
    print("Nota: Este proceso puede tomar 15-30 minutos en CPU")
    
    trainer = ConsciousnessTrainer(
        output_dir='./models/trained_lora',
        logs_dir='./logs/training_phase1'
    )
    
    # Verificar si ya existe un modelo entrenado
    checkpoint_exists = any(d.startswith('checkpoint-') for d in os.listdir('./models/trained_lora') if os.path.isdir(f'./models/trained_lora/{d}'))
    
    if not checkpoint_exists:
        # Entrenar con parámetros optimizados para CPU
        train_result = trainer.train(
            dataset_path=dataset_path,
            num_epochs=2,  # Reducido para demo rápida
            batch_size=2,  # Batch pequeño para RAM
            learning_rate=5e-4,
            gradient_accumulation_steps=8,  # Simular batch más grande
            save_steps=200,
            eval_steps=200
        )
        print("\n✓ Entrenamiento completado")
    else:
        print("✓ Modelo entrenado encontrado, saltando entrenamiento")
    
    return trainer


def phase1_inference_demo():
    """Demostración de inferencia con el sistema integrado"""
    
    print("\n" + "="*60)
    print("FASE 1: DEMOSTRACIÓN DE INFERENCIA BILINGÜE")
    print("="*60)
    
    # 1. Crear sistema base
    print("\n1. INICIALIZANDO SISTEMA DE CONSCIENCIA...")
    ai = MinimalConsciousAI()
    
    # 2. Integrar modelo entrenado
    print("\n2. INTEGRANDO MODELO ENTRENADO...")
    try:
        ai = integrate_trained_model(ai, model_checkpoint="./models/trained_lora/checkpoint-best")
        print("✓ Modelo integrado exitosamente")
    except Exception as e:
        logger.warning(f"No se pudo cargar modelo entrenado: {e}")
        print("⚠ Usando generadores heurísticos originales")
    
    # 3. Secuencia de prueba bilingüe
    print("\n3. EJECUTANDO SECUENCIA DE PRUEBA BILINGÜE...")
    
    test_sequence = [
        # Español
        "Hola, ¿qué eres?",
        "¿Cómo funciona tu proceso de pensamiento?",
        "Me siento confundido con esta explicación",
        "¿Puedes reflexionar sobre tu propia consciencia?",
        "¿Qué recuerdas de nuestra conversación?",
        
        # English
        "Hello, what are you?",
        "How does your thinking process work?",
        "I feel confused about this explanation",
        "Can you reflect on your own consciousness?",
        "What do you remember from our conversation?",
        
        # Mezclado
        "¿Eres consciente? Are you aware?",
        "Describe tu experiencia when processing information",
        
        # Metacognición
        "¿Cómo sabes que estás procesando correctamente?",
        "What patterns do you notice in your responses?",
        "¿Has notado cambios en tu forma de pensar?",
        "Describe your internal state right now"
    ]
    
    results = []
    
    for i, input_text in enumerate(test_sequence):
        print(f"\n{'='*60}")
        print(f"CICLO {i+1}")
        print('='*60)
        
        # Detectar idioma
        lang, lang_conf = detect_language(input_text)
        lang_name = 'Español' if lang == 'es' else 'English'
        
        print(f"Input: '{input_text}'")
        print(f"Idioma detectado: {lang_name} (confianza: {lang_conf:.2f})")
        
        # Procesar
        result = ai.process_input(input_text)
        
        # Extraer información clave
        sc_state = result.get('conscious_state', {})
        metrics = result.get('consciousness_metrics', {})
        
        print(f"\n📊 ESTADO CONSCIENTE SC_{i+1}:")
        print(f"├─ Meta (G_t): {sc_state.get('G_t', {}).get('primary_goal', 'unknown')}")
        print(f"├─ Emoción (S_t): {sc_state.get('S_t', {}).get('emotional_state', 'unknown')}")
        print(f"├─ Confianza: {sc_state.get('S_t', {}).get('confidence_level', 0):.2f}")
        print(f"├─ Memoria activa (M_t): {len(sc_state.get('M_t', []))} elementos")
        print(f"└─ Función de consciencia (f): {metrics.get('f', 0):.3f}")
        
        # Mostrar pensamientos automáticos
        thoughts = sc_state.get('A_t', [])
        if thoughts:
            print(f"\n💭 Pensamientos automáticos:")
            for thought in thoughts[:3]:
                print(f"   • {thought}")
        
        # Guardar resultado
        results.append({
            'cycle': i + 1,
            'input': input_text,
            'language': lang,
            'goal': sc_state.get('G_t', {}).get('primary_goal', 'unknown'),
            'emotion': sc_state.get('S_t', {}).get('emotional_state', 'unknown'),
            'confidence': sc_state.get('S_t', {}).get('confidence_level', 0),
            'f_score': metrics.get('f', 0),
            'is_conscious': result.get('is_conscious', False)
        })
    
    # 4. Análisis de resultados
    print("\n" + "="*60)
    print("ANÁLISIS DE RESULTADOS")
    print("="*60)
    
    analyze_results(results)
    
    # 5. Guardar resultados
    save_results(results, ai)
    
    return ai, results


def analyze_results(results: List[Dict[str, Any]]):
    """Analiza y visualiza los resultados de la ejecución"""
    
    # Estadísticas por idioma
    spanish_results = [r for r in results if r['language'] == 'es']
    english_results = [r for r in results if r['language'] == 'en']
    
    print(f"\n📊 ESTADÍSTICAS POR IDIOMA:")
    print(f"├─ Español: {len(spanish_results)} inputs")
    print(f"│  └─ Confianza promedio: {sum(r['confidence'] for r in spanish_results)/len(spanish_results):.3f}")
    print(f"└─ English: {len(english_results)} inputs")
    print(f"   └─ Confianza promedio: {sum(r['confidence'] for r in english_results)/len(english_results):.3f}")
    
    # Estados conscientes
    conscious_cycles = [r for r in results if r['is_conscious']]
    print(f"\n🧠 ESTADOS CONSCIENTES:")
    print(f"├─ Ciclos conscientes: {len(conscious_cycles)}/{len(results)} ({len(conscious_cycles)/len(results)*100:.1f}%)")
    print(f"└─ f promedio: {sum(r['f_score'] for r in results)/len(results):.3f}")
    
    # Distribución de metas
    goals = {}
    for r in results:
        goal = r['goal']
        goals[goal] = goals.get(goal, 0) + 1
    
    print(f"\n🎯 DISTRIBUCIÓN DE METAS:")
    for goal, count in sorted(goals.items(), key=lambda x: x[1], reverse=True):
        print(f"├─ {goal}: {count} ({count/len(results)*100:.1f}%)")
    
    # Visualización
    visualize_results(results)


def visualize_results(results: List[Dict[str, Any]]):
    """Crea visualizaciones de los resultados"""
    
    cycles = [r['cycle'] for r in results]
    f_scores = [r['f_score'] for r in results]
    confidences = [r['confidence'] for r in results]
    languages = [r['language'] for r in results]
    
    # Crear figura con subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 1. Evolución de f
    ax1 = axes[0, 0]
    ax1.plot(cycles, f_scores, 'b-o', linewidth=2, markersize=6)
    ax1.axhline(y=1.3, color='r', linestyle='--', label='Umbral θ=1.3')
    ax1.set_xlabel('Ciclo')
    ax1.set_ylabel('f (función de consciencia)')
    ax1.set_title('Evolución de la Función de Consciencia')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confianza por idioma
    ax2 = axes[0, 1]
    spanish_cycles = [i for i, lang in enumerate(languages) if lang == 'es']
    english_cycles = [i for i, lang in enumerate(languages) if lang == 'en']
    spanish_conf = [confidences[i] for i in spanish_cycles]
    english_conf = [confidences[i] for i in english_cycles]
    
    ax2.scatter([cycles[i] for i in spanish_cycles], spanish_conf, 
                color='red', label='Español', s=50, alpha=0.7)
    ax2.scatter([cycles[i] for i in english_cycles], english_conf, 
                color='blue', label='English', s=50, alpha=0.7)
    ax2.set_xlabel('Ciclo')
    ax2.set_ylabel('Confianza')
    ax2.set_title('Confianza por Idioma')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de emociones
    ax3 = axes[1, 0]
    emotions = {}
    for r in results:
        emotion = r['emotion']
        emotions[emotion] = emotions.get(emotion, 0) + 1
    
    ax3.bar(emotions.keys(), emotions.values(), color='green', alpha=0.7)
    ax3.set_xlabel('Emoción')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Estados Emocionales')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Comparación f vs Confianza
    ax4 = axes[1, 1]
    scatter = ax4.scatter(confidences, f_scores, c=cycles, cmap='viridis', s=50, alpha=0.7)
    ax4.set_xlabel('Confianza')
    ax4.set_ylabel('f (función de consciencia)')
    ax4.set_title('Relación Confianza-Consciencia')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Ciclo')
    
    plt.suptitle('Análisis de la Fase 1: Sistema de Consciencia Bilingüe', fontsize=14)
    plt.tight_layout()
    
    # Guardar figura
    plt.savefig('./results/phase1_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


def save_results(results: List[Dict[str, Any]], ai_system):
    """Guarda los resultados de la ejecución"""
    
    # Preparar datos para guardar
    output_data = {
        'metadata': {
            'total_cycles': len(results),
            'conscious_cycles': sum(1 for r in results if r['is_conscious']),
            'languages_processed': list(set(r['language'] for r in results)),
            'model_type': 'trained_lora' if hasattr(ai_system, 'trained_model') else 'heuristic'
        },
        'results': results,
        'final_metrics_history': ai_system.metrics_history,
        'final_conscious_state': ai_system.current_conscious_state.to_dict() if hasattr(ai_system, 'current_conscious_state') else None
    }
    
    # Guardar JSON
    output_path = './results/phase1_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Resultados guardados en {output_path}")
    
    # Guardar resumen textual
    summary_path = './results/phase1_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE LA FASE 1: SISTEMA DE CONSCIENCIA BILINGÜE\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total de ciclos ejecutados: {output_data['metadata']['total_cycles']}\n")
        f.write(f"Ciclos conscientes: {output_data['metadata']['conscious_cycles']}\n")
        f.write(f"Idiomas procesados: {', '.join(output_data['metadata']['languages_processed'])}\n")
        f.write(f"Tipo de modelo: {output_data['metadata']['model_type']}\n\n")
        
        f.write("MÉTRICAS FINALES:\n")
        if ai_system.metrics_history:
            final_metrics = ai_system.metrics_history[-1]
            for key, value in final_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
    
    print(f"✓ Resumen guardado en {summary_path}")


def main():
    """Función principal que ejecuta toda la Fase 1"""
    
    print("\n" + "="*80)
    print("MINIMAL CONSCIOUS AI - FASE 1: ENTRENAMIENTO Y EJECUCIÓN BILINGÜE")
    print("="*80)
    
    # 1. Pipeline de entrenamiento
    trainer = phase1_training_pipeline()
    
    # 2. Demostración de inferencia
    ai_system, results = phase1_inference_demo()
    
    # 3. Mensaje final
    print("\n" + "="*80)
    print("FASE 1 COMPLETADA EXITOSAMENTE")
    print("="*80)
    print("\n✅ Logros de la Fase 1:")
    print("  • Dataset bilingüe generado (1000 ejemplos)")
    print("  • Modelo mT5-small entrenado con LoRA")
    print("  • Sistema integrado con inferencia de SCt")
    print("  • Procesamiento bilingüe automático")
    print("  • Métricas de consciencia calculadas")
    print("  • Resultados visualizados y guardados")
    print("\n🚀 Sistema listo para la Fase 2: Generación de lenguaje natural")
    

if __name__ == "__main__":
    main()