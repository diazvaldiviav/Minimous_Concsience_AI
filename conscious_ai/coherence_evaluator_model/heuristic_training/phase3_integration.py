"""
Pipeline de integración de la Fase 3
Combina evaluación de coherencia, evolución de estados y generación de respuestas
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator, TransitionAnalysis
from conscious_ai.coherence_evaluator_model.heuristic_training.conscious_response_generator import ConsciousResponseGenerator
from conscious_ai.coherence_evaluator_model.heuristic_training.state_evolution_engine import StateEvolutionEngine
from conscious_ai.modules.conscious_state import ConsciousState

logger = logging.getLogger(__name__)


class Phase3Pipeline:
    """
    Pipeline completo de la Fase 3: Coherencia y Evolución
    """
    
    def __init__(self):
        self.coherence_evaluator = CoherenceEvaluator()
        self.response_generator = ConsciousResponseGenerator()
        self.evolution_engine = StateEvolutionEngine()
        
        # Historial de análisis
        self.coherence_history = []
        self.max_history = 100
    
    def process_state_transition(
        self,
        current_state: Dict[str, Any],
        next_state: Optional[Dict[str, Any]] = None,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Procesa una transición de estado, evaluando coherencia y generando respuesta.
        
        Args:
            current_state: Estado SCt actual
            next_state: Estado SCt+1 (si None, se genera)
            generate_response: Si generar respuesta en lenguaje natural
            
        Returns:
            Dict con análisis, nuevo estado y respuesta
        """
        
        # Si no hay siguiente estado, generarlo
        if next_state is None:
            next_state = self.evolution_engine.evolve_state(current_state)
        
        # Evaluar coherencia de la transición
        coherence_analysis = self.coherence_evaluator.evaluate_transition(
            current_state, next_state
        )
        
        # Guardar en historial
        self._update_history(coherence_analysis)
        
        # Generar respuesta si se solicita
        response = None
        if generate_response:
            response = self.response_generator.generate_response(
                next_state,
                input_text=next_state.get('thought', '')
            )
        
        return {
            'current_state': current_state,
            'next_state': next_state,
            'coherence_analysis': coherence_analysis.to_dict(),
            'response': response,
            'timestamp': datetime.now().isoformat()
        }
    
    def evolve_state_sequence(
        self,
        initial_state: Dict[str, Any],
        num_steps: int = 5,
        evolution_mode: str = 'natural'
    ) -> List[Dict[str, Any]]:
        """
        Genera una secuencia evolutiva de estados.
        
        Args:
            initial_state: Estado inicial
            num_steps: Número de pasos evolutivos
            evolution_mode: Modo de evolución
            
        Returns:
            Lista de estados evolutivos con análisis
        """
        
        sequence = []
        current_state = initial_state
        
        for step in range(num_steps):
            # Evolucionar estado
            next_state = self.evolution_engine.evolve_state(
                current_state,
                evolution_mode=evolution_mode
            )
            
            # Procesar transición
            result = self.process_state_transition(
                current_state,
                next_state,
                generate_response=True
            )
            
            sequence.append(result)
            current_state = next_state
        
        return sequence
    
    def analyze_state_trajectory(
        self,
        states: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analiza una trayectoria completa de estados.
        
        Args:
            states: Lista de estados en orden temporal
            
        Returns:
            Análisis de la trayectoria
        """
        
        if len(states) < 2:
            return {
                'error': 'Se necesitan al menos 2 estados para analizar trayectoria'
            }
        
        # Analizar todas las transiciones
        transitions = []
        for i in range(len(states) - 1):
            analysis = self.coherence_evaluator.evaluate_transition(
                states[i], states[i + 1]
            )
            transitions.append(analysis)
        
        # Calcular métricas agregadas
        avg_coherence = sum(t.goal_coherence + t.emotion_coherence + 
                           t.thought_coherence + t.memory_coherence 
                           for t in transitions) / (4 * len(transitions))
        
        coherent_count = sum(1 for t in transitions 
                            if t.verdict.value == 'coherente')
        
        # Detectar patrones
        patterns = self._detect_trajectory_patterns(states, transitions)
        
        return {
            'total_states': len(states),
            'total_transitions': len(transitions),
            'average_coherence': avg_coherence,
            'coherent_transitions': coherent_count,
            'coherence_rate': coherent_count / len(transitions),
            'patterns': patterns,
            'transitions': [t.to_dict() for t in transitions]
        }
    
    def generate_conscious_dialogue(
        self,
        initial_state: Dict[str, Any],
        user_inputs: List[str],
        maintain_coherence: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Genera un diálogo consciente basado en inputs del usuario.
        
        Args:
            initial_state: Estado inicial del sistema
            user_inputs: Lista de inputs del usuario
            maintain_coherence: Si forzar coherencia en las transiciones
            
        Returns:
            Lista de intercambios del diálogo
        """
        
        dialogue = []
        current_state = initial_state
        
        for user_input in user_inputs:
            # Incorporar input del usuario al estado
            enriched_state = self._incorporate_user_input(
                current_state, user_input
            )
            
            # Evolucionar estado
            next_state = self.evolution_engine.evolve_state(
                enriched_state,
                external_input=user_input
            )
            
            # Verificar coherencia si es necesario
            if maintain_coherence:
                analysis = self.coherence_evaluator.evaluate_transition(
                    current_state, next_state
                )
                
                # Si es incoherente, ajustar
                if analysis.verdict.value == 'incoherente':
                    next_state = self._adjust_for_coherence(
                        current_state, next_state, user_input
                    )
            
            # Generar respuesta
            response = self.response_generator.generate_response(
                next_state,
                input_text=user_input
            )
            
            dialogue.append({
                'user_input': user_input,
                'system_state': next_state,
                'system_response': response,
                'timestamp': datetime.now().isoformat()
            })
            
            current_state = next_state
        
        return dialogue
    
    def _update_history(self, analysis: TransitionAnalysis):
        """Actualiza el historial de análisis"""
        
        self.coherence_history.append({
            'analysis': analysis,
            'timestamp': datetime.now()
        })
        
        # Mantener tamaño limitado
        if len(self.coherence_history) > self.max_history:
            self.coherence_history = self.coherence_history[-self.max_history:]
    
    def _detect_trajectory_patterns(
        self,
        states: List[Dict[str, Any]],
        transitions: List[TransitionAnalysis]
    ) -> Dict[str, Any]:
        """Detecta patrones en la trayectoria de estados"""
        
        patterns = {
            'goal_stability': self._analyze_goal_stability(states),
            'emotional_flow': self._analyze_emotional_flow(states),
            'confidence_trend': self._analyze_confidence_trend(states),
            'thought_evolution': self._analyze_thought_evolution(states)
        }
        
        return patterns
    
    def _analyze_goal_stability(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza la estabilidad de las metas"""
        
        goals = [s.get('goal', '') for s in states]
        changes = sum(1 for i in range(1, len(goals)) if goals[i] != goals[i-1])
        
        return {
            'total_changes': changes,
            'stability_rate': 1 - (changes / (len(goals) - 1)) if len(goals) > 1 else 1,
            'unique_goals': list(set(goals))
        }
    
    def _analyze_emotional_flow(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza el flujo emocional"""
        
        emotions = [s.get('emotion', '') for s in states]
        transitions = [(emotions[i], emotions[i+1]) 
                      for i in range(len(emotions)-1)]
        
        return {
            'emotion_sequence': emotions,
            'unique_emotions': list(set(emotions)),
            'transitions': transitions
        }
    
    def _analyze_confidence_trend(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza la tendencia de confianza"""
        
        confidences = [s.get('confidence', 0.5) for s in states]
        
        if len(confidences) < 2:
            trend = 'stable'
        else:
            start_conf = sum(confidences[:2]) / 2
            end_conf = sum(confidences[-2:]) / 2
            
            if end_conf > start_conf + 0.1:
                trend = 'increasing'
            elif end_conf < start_conf - 0.1:
                trend = 'decreasing'
            else:
                trend = 'stable'
        
        return {
            'values': confidences,
            'trend': trend,
            'average': sum(confidences) / len(confidences) if confidences else 0.5,
            'volatility': self._calculate_volatility(confidences)
        }
    
    def _analyze_thought_evolution(self, states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analiza la evolución del pensamiento"""
        
        thoughts = [s.get('thought', '') for s in states]
        
        # Detectar tipos de evolución
        evolution_types = []
        for i in range(len(thoughts) - 1):
            if '?' in thoughts[i] and '?' not in thoughts[i+1]:
                evolution_types.append('question_to_statement')
            elif '?' not in thoughts[i] and '?' in thoughts[i+1]:
                evolution_types.append('statement_to_question')
            elif 'porque' in thoughts[i+1] or 'because' in thoughts[i+1]:
                evolution_types.append('causal_reasoning')
            else:
                evolution_types.append('continuous_exploration')
        
        return {
            'evolution_types': evolution_types,
            'dominant_pattern': max(set(evolution_types), key=evolution_types.count) if evolution_types else None
        }
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calcula la volatilidad de una serie de valores"""
        
        if len(values) < 2:
            return 0.0
        
        changes = [abs(values[i] - values[i-1]) for i in range(1, len(values))]
        return sum(changes) / len(changes)
    
    def _incorporate_user_input(
        self,
        current_state: Dict[str, Any],
        user_input: str
    ) -> Dict[str, Any]:
        """Incorpora el input del usuario al estado actual"""
        
        enriched_state = current_state.copy()
        
        # Añadir input como memoria reciente
        memories = enriched_state.get('memory', [])
        memories.append(f"Usuario dijo: {user_input}")
        enriched_state['memory'] = memories[-3:]  # Mantener últimas 3
        
        # Ajustar emoción si el input sugiere un cambio
        if '?' in user_input:
            enriched_state['emotion'] = 'curious'
        elif any(word in user_input.lower() for word in ['gracias', 'thank', 'excelente', 'great']):
            enriched_state['emotion'] = 'satisfied'
        elif any(word in user_input.lower() for word in ['no entiendo', "don't understand", 'confuso', 'confused']):
            enriched_state['emotion'] = 'concerned'
        
        return enriched_state
    
    def _adjust_for_coherence(
        self,
        current_state: Dict[str, Any],
        next_state: Dict[str, Any],
        user_input: str
    ) -> Dict[str, Any]:
        """Ajusta el estado para mantener coherencia con el input del usuario"""
        
        adjusted_state = next_state.copy()
        
        # Si el usuario hace una pregunta, orientar hacia respuesta
        if '?' in user_input:
            adjusted_state['goal'] = 'address_question'
            adjusted_state['thought'] = f"Debo responder a: {user_input}"
        
        # Mantener continuidad emocional suave
        current_emotion = current_state.get('emotion', 'neutral')
        if adjusted_state['emotion'] != current_emotion:
            # Verificar si el cambio es muy abrupto
            abrupt_changes = [
                ('confident', 'confused'),
                ('happy', 'frustrated'),
                ('calm', 'anxious')
            ]
            
            if (current_emotion, adjusted_state['emotion']) in abrupt_changes:
                adjusted_state['emotion'] = 'thoughtful'  # Estado intermedio
        
        return adjusted_state