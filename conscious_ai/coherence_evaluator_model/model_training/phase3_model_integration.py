"""
Pipeline de integración de la Fase 3 con modelos entrenados
Reemplaza las heurísticas con inferencia basada en modelos
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from conscious_ai.coherence_evaluator_model.model_training.model_based_coherence_evaluator import ModelBasedCoherenceEvaluator
from conscious_ai.coherence_evaluator_model.model_training.model_based_state_evolution import HybridStateEvolution
from conscious_ai.coherence_evaluator_model.heuristic_training.conscious_response_generator import ConsciousResponseGenerator

logger = logging.getLogger(__name__)


class ModelBasedPhase3Pipeline:
    """
    Pipeline de Fase 3 usando modelos entrenados para coherencia y evolución
    """
    
    def __init__(
        self,
        evolution_model_path: Optional[str] = "./models/autonomous_lora",
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        force_embedding_evaluation: bool = False
    ):
        # Componentes basados en modelos
        self.coherence_evaluator = ModelBasedCoherenceEvaluator(
            embedding_model=embedding_model,
            use_ml_classifier=(not force_embedding_evaluation)
        )
        
        self.state_evolution = HybridStateEvolution(
            model_checkpoint=evolution_model_path,
            use_model_primary=True
        )
        
        # Generador de respuestas (mantiene lógica original)
        self.response_generator = ConsciousResponseGenerator()
        
        # Métricas y análisis
        self.performance_metrics = {
            'coherent_transitions': 0,
            'total_transitions': 0,
            'model_success_rate': 0,
            'average_coherence_score': []
        }
    
    def process_state_transition(
        self,
        current_state: Dict[str, Any],
        next_state: Optional[Dict[str, Any]] = None,
        external_input: Optional[str] = None,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa transición usando modelos entrenados
        """
        
        start_time = datetime.now()
        
        # Si no hay siguiente estado, generarlo con modelo
        if next_state is None:
            next_state = self.state_evolution.evolve_state(
                current_state,
                external_input
            )
        
        # Evaluar coherencia con embeddings
        coherence_analysis = self.coherence_evaluator.evaluate_transition(
            current_state, next_state
        )
        
        # Actualizar métricas
        self._update_metrics(coherence_analysis)
        
        # Generar respuesta si se solicita
        response = None
        if generate_response:
            response = self.response_generator.generate_response(
                next_state,
                input_text=external_input or next_state.get('thought', '')
            )

        
        
        # Calcular tiempo de procesamiento
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            'current_state': current_state,
            'next_state': next_state,
            'coherence_analysis': coherence_analysis.to_dict(),
            'response': response,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'model_based': True
        }
    
    def evolve_state_sequence(
        self,
        initial_state: Dict[str, Any],
        num_steps: int = 5,
        temperature: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Genera secuencia evolutiva usando modelos
        """
        
        sequence = []
        current_state = initial_state
        
        logger.info(f"Generando secuencia de {num_steps} estados con modelos")
        
        for step in range(num_steps):
            # Evolucionar con modelo
            result = self.process_state_transition(
                current_state,
                generate_response=True
            )
            
            sequence.append(result)
            current_state = result['next_state']
            
            # Log progreso
            coherence = result['coherence_analysis']['metricas']
            logger.info(
                f"Paso {step + 1}: Coherencia global = "
                f"{sum(coherence.values()) / len(coherence):.3f}"
            )
        
        return sequence
    
    def analyze_model_performance(self) -> Dict[str, Any]:
        """
        Analiza el rendimiento del modelo en las transiciones
        """
        
        if self.performance_metrics['total_transitions'] == 0:
            return {
                'error': 'No hay transiciones para analizar'
            }
        
        coherence_rate = (
            self.performance_metrics['coherent_transitions'] / 
            self.performance_metrics['total_transitions']
        )
        
        avg_coherence = (
            sum(self.performance_metrics['average_coherence_score']) / 
            len(self.performance_metrics['average_coherence_score'])
            if self.performance_metrics['average_coherence_score'] else 0
        )
        
        return {
            'total_transitions': self.performance_metrics['total_transitions'],
            'coherent_transitions': self.performance_metrics['coherent_transitions'],
            'coherence_rate': coherence_rate,
            'average_coherence_score': avg_coherence,
            'model_type': 'hybrid' if hasattr(self.state_evolution, 'has_model') else 'heuristic'
        }
    
    def _update_metrics(self, analysis):
        """Actualiza métricas de rendimiento"""
        
        self.performance_metrics['total_transitions'] += 1
        
        if analysis.verdict.value == 'coherente':
            self.performance_metrics['coherent_transitions'] += 1
        
        # Calcular coherencia promedio
        coherence_score = (
            analysis.goal_coherence * 0.2 +
            analysis.emotion_coherence * 0.15 +
            analysis.thought_coherence * 0.35 +
            analysis.memory_coherence * 0.15 +
            (1.0 - abs(analysis.confidence_change)) * 0.15
        )
        
        self.performance_metrics['average_coherence_score'].append(coherence_score)
    ### NUEVO MÉTODO PARA GUARDAR ###
    def save_sequence_to_jsonl(self, sequence: List[Dict[str, Any]], filepath: str):
        """
        Guarda una secuencia de transiciones en un archivo .jsonl para etiquetado.
        Cada línea contiene un par (sc_t, sc_t_plus_1).
        """
        logger.info(f"Guardando {len(sequence)} transiciones en {filepath}...")
        with open(filepath, 'w', encoding='utf-8') as f:
            for transition in sequence:
                # Estructura para el etiquetado
                record = {
                    'sc_t': transition['current_state'],
                    'sc_t_plus_1': transition['next_state'],
                    'label': ''  # Dejar vacío para el etiquetado manual
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        logger.info("Guardado completado.")

    ### NUEVO MÉTODO INTERACTIVO ###
    def evolve_and_label_interactively(
        self,
        initial_state: Dict[str, Any],
        num_steps: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Genera una secuencia y pide al usuario que etiquete cada transición.
        """
        labeled_data = []
        current_state = initial_state
        
        print("\n--- INICIO DE ETIQUETADO INTERACTIVO ---")
        print("Califica cada transición: (c=coherente, i=incoherente, a=ambiguo, s=saltar, q=salir)")

        for i in range(num_steps):
            print(f"\n--- Paso {i + 1}/{num_steps} ---")
            
            # Generar la siguiente transición
            transition = self.process_state_transition(current_state, generate_response=False)
            next_state = transition['next_state']
            
            # Mostrar la transición al usuario
            print(f"ESTADO ANTERIOR:\n  Pensamiento: {current_state.get('thought')}\n  Meta: {current_state.get('goal')}")
            print(f"ESTADO GENERADO:\n  Pensamiento: {next_state.get('thought')}\n  Meta: {next_state.get('goal')}")
            
            # Pedir etiqueta
            while True:
                choice = input("Califica la coherencia (c/i/a/s/q): ").lower()
                if choice in ['c', 'i', 'a', 's', 'q']:
                    break
                print("Opción no válida. Inténtalo de nuevo.")

            if choice == 'q':
                print("Saliendo del modo de etiquetado.")
                break
            
            if choice != 's':
                label_map = {'c': 'coherent', 'i': 'incoherent', 'a': 'ambiguous'}
                label = label_map[choice]
                
                record = {
                    'sc_t': current_state,
                    'sc_t_plus_1': next_state,
                    'label': label
                }
                labeled_data.append(record)
                print(f"  > Guardado como: {label}")

            current_state = next_state
        
        print(f"\n--- ETIQUETADO FINALIZADO ---")
        print(f"Se han etiquetado {len(labeled_data)} nuevas transiciones.")
        return labeled_data