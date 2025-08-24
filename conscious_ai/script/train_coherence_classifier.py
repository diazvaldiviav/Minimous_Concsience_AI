#!/usr/bin/env python3
# scripts/train_coherence_classifier.py

"""
Script para entrenar el clasificador de coherencia.
Incluye un modo de etiquetado interactivo para enriquecer el dataset.
"""

import sys
import os
import argparse
import json
import logging

# Añadir la ruta raíz del proyecto para que los imports funcionen
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones de tu lógica de IA
from conscious_ai.coherence_evaluator_model.model_training.coherence_classifier_trainer import (
    train_coherence_classifier,
    generate_synthetic_training_data
)
# ### NUEVO ### Importamos el pipeline para generar transiciones
from conscious_ai.coherence_evaluator_model.model_training.phase3_model_integration import ModelBasedPhase3Pipeline


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_annotated_data(filepath: str):
    """Carga datos anotados desde archivo JSONL."""
    data = []
    if not os.path.exists(filepath):
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                
                # Handle different data formats
                if 'sc_t' in item and 'sc_t_plus_1' in item and 'label' in item:
                    # Original expected format
                    data.append((
                        item['sc_t'],
                        item['sc_t_plus_1'],
                        item['label']
                    ))
                elif 'previous_SC' in item and 'current_SC' in item:
                    # Autonomous thought data format - auto-label as coherent
                    # You may want to adjust the label based on your needs
                    data.append((
                        item['previous_SC'],
                        item['current_SC'],
                        'coherent'  # Default coherence label for autonomous data
                    ))
                else:
                    logger.warning(f"Skipping item with unexpected format: {list(item.keys())}")
    return data

### NUEVO ### Función para guardar/añadir datos etiquetados
def save_annotated_data(labeled_data: list, filepath: str):
    """Guarda o añade los datos etiquetados a un archivo .jsonl."""
    file_exists = os.path.isfile(filepath)
    mode = 'a' if file_exists else 'w'
    
    with open(filepath, mode, encoding='utf-8') as f:
        # Si estamos añadiendo a un archivo existente, empezar en una nueva línea
        if file_exists and os.path.getsize(filepath) > 0:
            f.write('\n')
            
        for i, record in enumerate(labeled_data):
            f.write(json.dumps(record, ensure_ascii=False))
            if i < len(labeled_data) - 1:
                f.write('\n')
            
    logger.info(f"Se han guardado/añadido {len(labeled_data)} registros en '{filepath}'")


### NUEVO ### Función para el modo de etiquetado interactivo
def run_interactive_labeling(num_steps: int, output_file: str):
    """Inicia una sesión de etiquetado interactivo."""
    pipeline = ModelBasedPhase3Pipeline(force_embedding_evaluation=True)
    
    initial_state = {
        "goal": "understand_self",
        "emotion": "curious",
        "confidence": 0.45,
        "thought": "¿Qué patrones emergen de mi propia observación?",
        "memory": ["inicio de introspección"],
        "language": "es"
    }

    labeled_transitions = pipeline.evolve_and_label_interactively(
        initial_state=initial_state,
        num_steps=num_steps
    )

    if labeled_transitions:
        save_annotated_data(labeled_transitions, output_file)
    else:
        logger.info("No se etiquetó ningún dato en esta sesión.")


def main():
    parser = argparse.ArgumentParser(description='Entrena o enriquece el dataset del clasificador de coherencia.')
    
    # Argumentos existentes modificados ligeramente
    parser.add_argument(
        '--data-file',
        type=str,
        required=True, # Hacerlo requerido para saber siempre dónde guardar/leer
        help='Archivo para guardar o leer datos anotados (JSONL)'
    )
    parser.add_argument(
        '--synthetic-samples',
        type=int,
        default=500, # Reducir un poco el default
        help='Número de muestras sintéticas a AÑADIR durante el entrenamiento'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./models/coherence_classifier',
        help='Directorio de salida para el modelo entrenado'
    )
    
    # ### NUEVOS ### Argumentos para el modo de etiquetado
    parser.add_argument(
        '--label',
        action='store_true',
        help='Activa el modo de etiquetado interactivo para enriquecer el dataset.'
    )
    parser.add_argument(
        '--label-steps',
        type=int,
        default=15,
        help='Número de transiciones a generar y etiquetar en el modo interactivo.'
    )
    
    args = parser.parse_args()
    
    ### MODIFICADO ### Lógica principal para separar los modos
    if args.label:
        # --- MODO ETIQUETADO ---
        logger.info(f"Iniciando modo de etiquetado interactivo para {args.label_steps} pasos...")
        run_interactive_labeling(args.label_steps, args.data_file)
    else:
        # --- MODO ENTRENAMIENTO ---
        training_data = []
        
        logger.info(f"Cargando datos anotados desde {args.data_file}")
        training_data.extend(load_annotated_data(args.data_file))
        
        if not training_data:
            logger.warning(f"El archivo '{args.data_file}' está vacío o no existe.")
        
        if args.synthetic_samples > 0:
            logger.info(f"Generando y añadiendo {args.synthetic_samples} muestras sintéticas")
            training_data.extend(generate_synthetic_training_data(args.synthetic_samples))
        
        if not training_data:
            logger.error("No hay datos para entrenar. Usa el modo --label para crear un dataset.")
            return
        
        logger.info(f"Total de muestras para entrenar: {len(training_data)}")
        
        # Entrenar clasificador
        train_coherence_classifier(
            training_data,
            output_path=args.output_dir
        )
        
        logger.info("✓ Entrenamiento completado exitosamente")


if __name__ == "__main__":
    main()