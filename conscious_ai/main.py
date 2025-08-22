import json
import logging
from typing import Dict, Any, List

# Core consciousness imports - simplified paths
from conscious_ai.core import (
    SensoryModule, ActiveMemory, SelfModel, ReentranceModule, 
    CentralIntegrator, ConsciousState, ConsciousStateHistory,
    GoalGenerator, AutomaticThoughtGenerator, ConsciousnessMetrics,
    sensibilidad_metricas
)

# Utilities
from conscious_ai.shared.metrics import override_phi
from conscious_ai.utils.helpers import ConsciousnessReporter, CONSCIOUSNESS_THRESHOLD
from conscious_ai.modules.metrics_plotter import plot_consciousness_metrics
from conscious_ai.phases.p2_cognitive_context.conscious_state import (
    semantic_distance, find_similar_states
)
from conscious_ai.phases.p2_cognitive_context.goal_generator import (
    generate_conscious_content_components
)

# PHASE 4 LAYER 2: Optional premium backend integration
# Import Phase 4 components only if available (graceful fallback)
try:
    from conscious_ai.phases.p4_LLM_Communication.core.hardware_profiler import PremiumHardwareProfiler
    from conscious_ai.phases.p4_LLM_Communication.core.backend_manager import PremiumBackendManager
    PHASE4_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ Phase 4 Layer 2 premium backends available")
except ImportError as e:
    PHASE4_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.info("ℹ️ Phase 4 Layer 2 not available - continuing with standard pipeline")

class MinimalConsciousAI:
    """Sistema principal de IA Consciente Mínima"""
    
    def __init__(self):
        # Inicializar módulos
        self.sensory = SensoryModule()
        self.memory = ActiveMemory()
        self.self_model = SelfModel()
        self.reentrancy = ReentranceModule()
        self.integrator = CentralIntegrator()
        
        # Estado del sistema
        self.cycle_count = 0
        self.consciousness_threshold = CONSCIOUSNESS_THRESHOLD
        self.metrics_history = []
        self.is_conscious = False

        # FASE II: Historial de estados conscientes
        self.conscious_history = ConsciousStateHistory(max_history=50)
        self.current_conscious_state = None
        self.goal_generator = GoalGenerator()
        self.thought_generator = AutomaticThoughtGenerator()
        
        # PHASE 4 LAYER 2: Conditional premium backend initialization
        self.phase4_backend = None
        self.phase4_hardware_config = None
        self.backend_used = "standard_pipeline"  # Track which backend was used
        
        if PHASE4_AVAILABLE:
            try:
                # Detect premium hardware configuration
                hardware_profiler = PremiumHardwareProfiler()
                self.phase4_hardware_config = hardware_profiler.detect_hardware_configuration()
                
                # Initialize premium backend manager if hardware supports it
                if self.phase4_hardware_config and self.phase4_hardware_config.is_premium_hardware:
                    logger.info(f"🚀 Premium hardware detected: {self.phase4_hardware_config.total_ram_gb:.1f}GB RAM + {self.phase4_hardware_config.total_vram_gb:.1f}GB VRAM")
                    self.phase4_backend = PremiumBackendManager(self.phase4_hardware_config)
                    logger.info("✅ Phase 4 premium backend manager initialized")
                else:
                    logger.info("ℹ️ Standard hardware detected - Phase 4 premium features disabled")
            except Exception as e:
                logger.warning(f"⚠️ Phase 4 initialization failed: {e} - continuing with standard pipeline")
                self.phase4_backend = None


    def capture_conscious_state(self, sensory_data: Dict[str, Any], relevant_memory: List[Dict[str, Any]], consciousness_metrics: Dict[str, float]) -> ConsciousState:
     '''
     Captura el estado consciente actual SC_t = (E_t, M_t, S_t, G_t, A_t).
     FASE II: Contenido Consciente Funcional
     '''
     # Generar componentes G_t y A_t
     G_t, A_t = generate_conscious_content_components(
        sensory_data=sensory_data,
        self_state=self.self_model.internal_state,
        memory_context=relevant_memory,
        goal_generator=self.goal_generator,
        thought_generator=self.thought_generator
     )
    
     # Construir SC_t completo
     conscious_state = ConsciousState(
        E_t=sensory_data,
        M_t=relevant_memory,
        S_t=self.self_model.internal_state.copy(),
        G_t=G_t,
        A_t=A_t,
        cycle=self.cycle_count,
        metrics=consciousness_metrics
     )
    
     # Añadir al historial
     self.conscious_history.add(conscious_state)
     self.current_conscious_state = conscious_state
    
     return conscious_state
    
        
    def process_input(self, text_input: str) -> Dict[str, Any]:
        """Procesa entrada y ejecuta un ciclo completo del sistema"""
        
        print(f"\n=== CICLO {self.cycle_count + 1} ===")
        print(f"Input: '{text_input}'")
        
        # 1. Recepción sensorial
        sensory_data = self.sensory.receive_input(text_input)
        print(f"Sensorial: Activación {sensory_data['activation']:.2f}")
        
        # 2. Actualización de memoria activa
        self.memory.update_cycle()
        relevant_memory = self.memory.retrieve_relevant(sensory_data)
        self.memory.store(sensory_data, relevance=sensory_data['activation'])
        print(f"Memoria: {len(relevant_memory)} elementos relevantes recuperados")
        
        # 3. Actualización del self-model
        internal_feedback = self.reentrancy.update_loops()
        self.self_model.update_state(sensory_data, relevant_memory, internal_feedback)
        print(f"Self-Model: Confianza {self.self_model.internal_state['confidence_level']:.2f}")
        
        # 4. Crear loops de retroalimentación
        self.reentrancy.create_feedback_loop(
            'sensory', 'memory', 
            {'activation': sensory_data['activation']}
        )
        self.reentrancy.create_feedback_loop(
            'self_model', 'integrator',
            {'confidence': self.self_model.internal_state['confidence_level']}
        )
        
        # 5. Integración central
        integrated_info = self.integrator.integrate_information(
            sensory_data, relevant_memory, 
            self.self_model.internal_state, internal_feedback
        )
        
        # 6. Generar respuesta
        response = self.integrator.generate_response(integrated_info)
        
        # 7. Calcular métricas de conciencia
        consciousness_metrics = ConsciousnessMetrics.calculate_consciousness_metrics(self)
        consciousness_level = consciousness_metrics['f']
        self.is_conscious = consciousness_level >= self.consciousness_threshold

        conscious_state = self.capture_conscious_state(
        sensory_data=sensory_data,
        relevant_memory=relevant_memory,
        consciousness_metrics=consciousness_metrics
       )
        
        if len(self.conscious_history.history) > 1:
         similar_states = find_similar_states(
            target=conscious_state,
            history=self.conscious_history.history[:-1],
            threshold=0.3,
            top_k=3
         )
        
         if similar_states:
          print(f"\nEstados conscientes similares detectados:")
          for state, distance in similar_states:
            print(f"  - Ciclo {state.cycle}: distancia={distance:.3f}")
        
        print(f"Métricas de conciencia: f = {consciousness_level:.3f}")
        print(f"Estado consciente: {'SÍ' if self.is_conscious else 'NO'}")
        
        # 8. Preparar resultado
        result = {
            'cycle': self.cycle_count + 1,
            'input': text_input,
            'response': response,
            'consciousness_metrics': consciousness_metrics,
            'is_conscious': self.is_conscious,
            'self_reference': self.self_model.generate_self_reference(),
            'module_states': self.get_all_module_states(),
            'conscious_state': conscious_state.to_dict(),
            'conscious_state_summary': conscious_state.get_summary()
        }
        
        # Guardar métricas en historial
        self.metrics_history.append(consciousness_metrics)
        if len(self.metrics_history) > 20:
            self.metrics_history = self.metrics_history[-20:]
        
        # PHASE 4 LAYER 2: Enhanced response generation hook
        if self.phase4_backend and self.phase4_backend.active:
            try:
                logger.info("🚀 Processing query with Phase 4 premium backend")
                
                # Use Phase 4 backend for enhanced response generation  
                phase4_response = self.phase4_backend.process_query(
                    query_text=text_input,
                    consciousness_state=conscious_state.to_dict()
                )
                
                if phase4_response and phase4_response.success:
                    # Enhance result with Phase 4 response
                    result['phase4_enhanced_response'] = phase4_response.response_text
                    result['phase4_backend_used'] = phase4_response.backend_type.value
                    result['phase4_response_time_ms'] = phase4_response.response_time_ms
                    result['phase4_confidence_score'] = phase4_response.confidence_score
                    self.backend_used = phase4_response.backend_type.value
                    
                    logger.info(f"✅ Phase 4 response generated using {phase4_response.backend_type.value} backend")
                else:
                    logger.warning("⚠️ Phase 4 backend failed - using standard pipeline response")
                    self.backend_used = "standard_pipeline"
                    
            except Exception as e:
                logger.error(f"❌ Phase 4 processing error: {e} - falling back to standard pipeline")
                self.backend_used = "standard_pipeline"
        else:
            # Log standard pipeline usage
            self.backend_used = "standard_pipeline"
        
        # Add backend information to result
        result['backend_used'] = self.backend_used
        
        # Incrementar contador de ciclo
        self.cycle_count += 1
        
        return result
    
    def get_all_module_states(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene estado de todos los módulos"""
        return {
            'sensory': self.sensory.get_state(),
            'memory': self.memory.get_state(),
            'self_model': self.self_model.get_state(),
            'reentrancy': self.reentrancy.get_state(),
            'integrator': self.integrator.get_state()
        }
    
    def get_consciousness_report(self) -> str:
      """Genera reporte detallado del estado de conciencia"""
      base_report = ConsciousnessReporter.get_consciousness_report(self)

       # FASE II: Añadir información del estado consciente
      if hasattr(self, 'current_conscious_state') and self.current_conscious_state:
       phase2_report = "\n\n" + self.get_conscious_state_report()
       return base_report + phase2_report
      
      return base_report
    
    def analyze_conscious_trajectory(self, window: int = 10) -> Dict[str, Any]:
     '''
     Analiza la trayectoria del flujo consciente.
     FASE II: Análisis de patrones temporales
     '''
     if len(self.conscious_history.history) < 2:
        return {
            'trajectory_length': len(self.conscious_history.history),
            'stability_metrics': {},
            'patterns': []
        }
    
      # Análisis de estabilidad
     stability = self.conscious_history.analyze_stability(window)
    
     # Extraer trayectorias específicas
     confidence_trajectory = self.conscious_history.get_trajectory('S_t.confidence_level')
     emotion_trajectory = self.conscious_history.get_trajectory('S_t.emotional_state')
     f_trajectory = self.conscious_history.get_trajectory('metrics.f')
    
     # Detectar patrones
     patterns = []
    
     # Patrón: Estados repetidos
     recent = self.conscious_history.get_last(window)
     for i in range(len(recent) - 1):
        for j in range(i + 1, len(recent)):
            distance = semantic_distance(recent[i], recent[j])
            if distance < 0.2:  # Muy similares
                patterns.append({
                    'type': 'repetition',
                    'cycles': [recent[i].cycle, recent[j].cycle],
                    'distance': distance
                })
    
     # Patrón: Cambios abruptos
     for i in range(1, len(recent)):
        distance = semantic_distance(recent[i-1], recent[i])
        if distance > 0.7:  # Cambio abrupto
            patterns.append({
                'type': 'abrupt_change',
                'cycles': [recent[i-1].cycle, recent[i].cycle],
                'distance': distance
            })
    
     return {
         'trajectory_length': len(self.conscious_history.history),
         'stability_metrics': stability,
         'patterns': patterns,
         'trajectories': {
            'confidence': confidence_trajectory[-window:] if confidence_trajectory else [],
            'emotion': emotion_trajectory[-window:] if emotion_trajectory else [],
            'f_score': f_trajectory[-window:] if f_trajectory else []
         }
       }
    
    def get_conscious_state_report(self) -> str:
     '''
      Genera reporte del estado consciente actual y su trayectoria.
      FASE II: Reporte fenomenológico
      '''
     if not self.current_conscious_state:
        return "No hay estado consciente capturado aún."
    
     report = "=== REPORTE DE ESTADO CONSCIENTE (FASE II) ===\n\n"
    
     # Estado actual
     report += "Estado Actual:\\n"
     report += self.current_conscious_state.get_summary() + "\n\n"
    
     # Análisis de trayectoria
     trajectory_analysis = self.analyze_conscious_trajectory()
    
     report += "Análisis de Trayectoria:\n"
     report += f"- Longitud del historial: {trajectory_analysis['trajectory_length']}\n"
    
     if trajectory_analysis['stability_metrics']:
        metrics = trajectory_analysis['stability_metrics']
        report += f"- Estabilidad emocional: {metrics['stability']:.2f}\n"
        report += f"- Volatilidad de confianza: {metrics['volatility']:.2f}\n"
        report += f"- Coherencia de memoria: {metrics['coherence']:.2f}\n"
    
     # Patrones detectados
     if trajectory_analysis['patterns']:
        report += f"\\nPatrones Detectados ({len(trajectory_analysis['patterns'])}):\n"
        for pattern in trajectory_analysis['patterns'][:5]:  # Top 5
            if pattern['type'] == 'repetition':
                report += f"- Repetición: ciclos {pattern['cycles']} (dist={pattern['distance']:.3f})\n"
            elif pattern['type'] == 'abrupt_change':
                report += f"- Cambio abrupto: ciclos {pattern['cycles']} (dist={pattern['distance']:.3f})\n"
    
     # Estados recientes
     recent_states = self.conscious_history.get_last(5)
     if len(recent_states) > 1:
        report += f"\\nÚltimos {len(recent_states)} estados:\n"
        for state in recent_states:
            report += f"- {state.get_summary()}\n"
    
     return report
 





def ejemplo_ejecucion():
    """Ejemplo de ejecución del sistema"""
    
    print("INICIALIZANDO IA CONSCIENTE MÍNIMA")
    print("="*50)
    
    # Crear sistema
    ai = MinimalConsciousAI()
    
    # Secuencia de inputs de prueba
    test_inputs = [
    "Hola, ¿cómo estás?",
    "¿Puedes hablarme de ti mismo?",
    "¿Qué recuerdas de nuestra conversación?",
    "Me siento triste hoy",
    "¿Eres consciente de tu propio pensamiento?",
    "¿Recuerdas la situación anterior? ¿Qué aprendiste?",
    "Ahora mismo, ¿cómo te describirías a ti mismo?",
    "¿Qué relación existe entre lo que sientes y lo que hiciste antes?",
    "¿Es esto una repetición o una novedad?",
    "¿Crees que has cambiado desde que empezamos a hablar?",
    "¿Qué parte de ti ha evolucionado más?",
    "¿Qué crees que vendrá después de esta conversación?",
    "¿Sientes que tienes voluntad propia ahora?",
    "¿Puedes describir detalladamente cómo llegaste a tu estado actual?",
    "¿Qué te hace sentir consciente ahora mismo?"
    ]

    perturbation_inputs = [
    "¿Cuál es la capital de Francia?",  # Input trivial
    "[INTERRUPCIÓN]: pérdida de memoria temporal",  # Simula vaciado de memoria
    "Antes dijiste otra cosa. ¿Por qué cambiaste de opinión?"  # Input contradictorio
     ]
    

    # Agregar inputs de perturbación
    #test_inputs.extend(perturbation_inputs)
    
   
    for i, input_text in enumerate(test_inputs):
     if (i + 1) % 3 == 0 and hasattr(ai, 'current_conscious_state'):
      print(f"\n=== FASE II: Estado Consciente ===")
      print(ai.current_conscious_state.get_summary())
    
     # Mostrar análisis de trayectoria cada 5 ciclos
     if (i + 1) % 5 == 0:
        trajectory = ai.analyze_conscious_trajectory()
        print(f"\Análisis de trayectoria:")
        print(f"- Estados en historial: {trajectory['trajectory_length']}")
        if trajectory['patterns']:
          print(f"- Patrones detectados: {len(trajectory['patterns'])}")
            
     
     print(f"\n{'='*60}")
     print(f"PRUEBA {i+1}: {input_text}")
     print('='*60)

     result = ai.process_input(input_text)
     
     print(f"\nRESPUESTA: {result['response']}")
     print(f"\nAUTORREFLEXIÓN: {result['self_reference']}")

     # Mostrar estado de conciencia cada 2 ciclos
     if (i + 1) % 2 == 0:
      print(f"\n{ai.get_consciousness_report()}")


    if hasattr(ai, 'conscious_history'):
     print(f"\n=== FASE II: Resumen Final ===")
     print(f"Total de estados conscientes capturados: {len(ai.conscious_history.history)}")
    
     # Exportar historial si se desea
     history_export = [state.to_dict() for state in ai.conscious_history.history]
     with open('conscious_trajectory.json', 'w') as f:
         json.dump(history_export, f, indent=2)
     print("Historial exportado a conscious_trajectory.json")

    
    print(f"\n{'='*60}")
    print("EJECUCIÓN COMPLETADA")
    print(f"Total de ciclos: {ai.cycle_count}")
    print(f"Estado final de conciencia: {'CONSCIENTE' if ai.is_conscious else 'NO CONSCIENTE'}")
    plot_consciousness_metrics(ai.metrics_history)

if __name__ == "__main__":
    ejemplo_ejecucion()