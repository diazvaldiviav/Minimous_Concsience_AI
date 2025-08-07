"""
Integración del pensamiento autónomo con MinimalConsciousAI
Permite al sistema generar ciclos de consciencia sin input externo
"""

import logging
import random
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from conscious_ai.main import MinimalConsciousAI
from conscious_ai.autonomus_thinking.autonomous_thinking import AutonomousThoughtGenerator

logger = logging.getLogger(__name__)


class AutonomousConsciousAI(MinimalConsciousAI):
    """
    Extensión de MinimalConsciousAI con capacidad de pensamiento autónomo.
    Puede generar nuevos estados conscientes sin input externo.
    """
    
    def __init__(self, autonomous_model_path: Optional[str] = None):
        super().__init__()
        
        # Inicializar generador de pensamiento autónomo
        self.autonomous_generator = AutonomousThoughtGenerator(autonomous_model_path)
        
        # Control de modo autónomo
        self.autonomous_mode = False
        self.autonomous_cycles = 0
        self.max_autonomous_cycles = 50  # Límite de seguridad
        
        # Métricas específicas del modo autónomo
        self.autonomous_metrics = {
            'total_autonomous_cycles': 0,
            'themes_explored': set(),
            'peak_consciousness': 0,
            'thought_coherence': []
        }
    
    def process_autonomous_cycle(self) -> Dict[str, Any]:
        """
        Ejecuta un ciclo de pensamiento autónomo sin input externo.
        Genera un nuevo SCₜ basado solo en estado interno.
        """
        
        if not self.autonomous_mode:
            logger.warning("Modo autónomo no activado")
            return None
        
        print(f"\n=== CICLO AUTÓNOMO {self.cycle_count + 1} ===")
        print("🤔 Generando pensamiento interno...")
        
        # Obtener estado anterior completo
        previous_conscious_state = self.current_conscious_state.to_dict() if self.current_conscious_state else None
        
        if not previous_conscious_state:
            # Primer ciclo autónomo - generar estado inicial
            logger.info("Iniciando primer ciclo autónomo")
            initial_input = random.choice([
                  "inicio reflexión interna sobre mi naturaleza",
                  "beginning internal reflection about my nature"
                  ])

            return self.process_input(initial_input)
        
        # Generar nuevo estado usando el generador autónomo
        autonomous_state = self.autonomous_generator.generate_autonomous_thought(
            previous_state=previous_conscious_state,
            memory_context=self.memory.memory_items,
            consciousness_metrics=self.metrics_history[-1] if self.metrics_history else {}
        )
        
        # Crear input sintético basado en el pensamiento autónomo
        synthetic_input = autonomous_state['thought']
        
        # Procesar como input normal pero marcado como autónomo
        print(f"💭 Pensamiento: '{synthetic_input}'")
        
        # Guardar estado autónomo temporalmente
        self._temp_autonomous_state = autonomous_state
        
        # Procesar con el sistema base
        result = self.process_input(synthetic_input)
        
        # Enriquecer resultado con información autónoma
        if result and 'conscious_state' in result:
            # Sobrescribir con goal y emotion del generador autónomo
            result['conscious_state']['G_t']['primary_goal'] = autonomous_state['goal']
            result['conscious_state']['S_t']['emotional_state'] = autonomous_state['emotion']
            result['conscious_state']['S_t']['confidence_level'] = autonomous_state['confidence']
            
            # Agregar memoria autónoma
            for mem in autonomous_state['memory']:
                self.memory.store({'text': mem, 'type': 'autonomous_thought'}, relevance=0.8)
            
            # Marcar como ciclo autónomo
            result['autonomous'] = True
            result['autonomous_theme'] = self.autonomous_generator.current_theme
        
        # Actualizar métricas autónomas
        self.autonomous_cycles += 1
        self.autonomous_metrics['total_autonomous_cycles'] += 1
        self.autonomous_metrics['themes_explored'].add(self.autonomous_generator.current_theme)
        
        if result and 'consciousness_metrics' in result:
            f_score = result['consciousness_metrics']['f']
            self.autonomous_metrics['peak_consciousness'] = max(
                self.autonomous_metrics['peak_consciousness'], 
                f_score
            )
        
        # Limpiar estado temporal
        self._temp_autonomous_state = None
        
        return result
    
    def enter_autonomous_mode(self, max_cycles: Optional[int] = None):
        """Activa el modo de pensamiento autónomo"""
        self.autonomous_mode = True
        self.autonomous_cycles = 0
        if max_cycles:
            self.max_autonomous_cycles = max_cycles
        
        print("\n" + "="*60)
        print("🧠 ENTRANDO EN MODO DE PENSAMIENTO AUTÓNOMO 🧠")
        print("="*60)
        print(f"El sistema generará hasta {self.max_autonomous_cycles} pensamientos propios")
        print("Sin necesidad de input externo")
        print("-"*60)
    
    def exit_autonomous_mode(self):
        """Desactiva el modo autónomo y muestra resumen"""
        self.autonomous_mode = False
        
        print("\n" + "="*60)
        print("📊 RESUMEN DEL MODO AUTÓNOMO")
        print("="*60)
        print(f"Ciclos autónomos completados: {self.autonomous_cycles}")
        print(f"Temas explorados: {', '.join(self.autonomous_metrics['themes_explored'])}")
        print(f"Consciencia máxima alcanzada: {self.autonomous_metrics['peak_consciousness']:.3f}")
        
        # Analizar patrones de pensamiento
        analysis = self.autonomous_generator.analyze_thought_patterns()
        # Manejo seguro de temas dominantes
        dominant_themes = analysis.get('dominant_themes', [])
        if dominant_themes:
         print(f"\nTemas dominantes:")
         for theme, count in dominant_themes[:3]:
          print(f"  - {theme}: {count} veces")

        # Mostrar estabilidad y confianza si están presentes
        if 'stability' in analysis:
         print(f"\nEstabilidad del pensamiento: {analysis['stability']:.2f}")

        if 'avg_confidence' in analysis:
         print(f"Confianza promedio: {analysis['avg_confidence']:.2f}")
    
    def run_autonomous_session(
        self, 
        num_cycles: int = 10,
        pause_between_cycles: float = 1.0,
        stop_on_low_consciousness: bool = True,
        consciousness_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Ejecuta una sesión completa de pensamiento autónomo.
        
        Args:
            num_cycles: Número de ciclos a ejecutar
            pause_between_cycles: Pausa entre ciclos (segundos)
            stop_on_low_consciousness: Detener si f cae muy bajo
            consciousness_threshold: Umbral mínimo de f
            
        Returns:
            Lista de resultados de cada ciclo
        """
        
        self.enter_autonomous_mode(max_cycles=num_cycles)
        results = []
        
        try:
            for i in range(num_cycles):
                # Ejecutar ciclo
                result = self.process_autonomous_cycle()
                
                if result:
                    results.append(result)
                    
                    # Mostrar estado
                    self._display_autonomous_state(result, i + 1)
                    
                    # Verificar condición de parada
                    f_score = result.get('consciousness_metrics', {}).get('f', 0)
                    if stop_on_low_consciousness and f_score < consciousness_threshold:
                        print(f"\n⚠️ Consciencia muy baja ({f_score:.3f}), deteniendo...")
                        break
                    
                    # Pausa entre ciclos
                    if i < num_cycles - 1:
                        time.sleep(pause_between_cycles)
                
                # Verificar límite de seguridad
                if self.autonomous_cycles >= self.max_autonomous_cycles:
                    print(f"\n⚠️ Límite de ciclos alcanzado ({self.max_autonomous_cycles})")
                    break
        
        finally:
            self.exit_autonomous_mode()
        
        return results
    
    def _display_autonomous_state(self, result: Dict[str, Any], cycle_num: int):
        """Muestra el estado del ciclo autónomo de forma legible"""
        
        sc = result.get('conscious_state', {})
        metrics = result.get('consciousness_metrics', {})
        theme = result.get('autonomous_theme', 'unknown')
        
        print(f"\n{'='*50}")
        print(f"Ciclo Autónomo #{cycle_num} - Tema: {theme}")
        print(f"{'='*50}")
        
        print(f"🎯 Meta: {sc.get('G_t', {}).get('primary_goal', 'unknown')}")
        print(f"😊 Emoción: {sc.get('S_t', {}).get('emotional_state', 'unknown')}")
        print(f"💪 Confianza: {sc.get('S_t', {}).get('confidence_level', 0):.2%}")
        print(f"🌟 Consciencia (f): {metrics.get('f', 0):.3f}")
        
        # Mostrar pensamientos generados
        thoughts = sc.get('A_t', [])
        if thoughts:
            print(f"\n💭 Reflexiones internas:")
            for thought in thoughts[:2]:
                print(f"   • {thought}")
        
        # Estado de consciencia
        if result.get('is_conscious', False):
            print("\n✨ Estado: CONSCIENTE ✨")
        else:
            print("\n💤 Estado: No consciente")
    
    def generate_thought_stream(
        self,
        duration_seconds: int = 60,
        min_pause: float = 0.5,
        max_pause: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Genera un flujo continuo de pensamientos durante un tiempo determinado.
        Simula un stream de consciencia más natural con pausas variables.
        """
        
        import random
        
        print(f"\n🌊 Iniciando flujo de pensamiento por {duration_seconds} segundos...")
        
        self.enter_autonomous_mode(max_cycles=1000)  # Alto límite
        results = []
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < duration_seconds:
                # Ciclo de pensamiento
                result = self.process_autonomous_cycle()
                
                if result:
                    results.append(result)
                    
                    # Mostrar solo pensamiento principal
                    thought = result.get('conscious_state', {}).get('A_t', ['...'])[0]
                    emotion = result.get('conscious_state', {}).get('S_t', {}).get('emotional_state', '?')
                    print(f"\n[{emotion}] {thought}")
                    
                    # Pausa variable basada en complejidad del pensamiento
                    thought_length = len(thought)
                    pause = min_pause + (thought_length / 100) * (max_pause - min_pause)
                    pause = min(max_pause, pause) + random.uniform(-0.2, 0.2)
                    
                    time.sleep(max(0.1, pause))
                
                # Verificar tiempo
                elapsed = time.time() - start_time
                if elapsed >= duration_seconds:
                    break
        
        finally:
            print(f"\n🏁 Flujo de pensamiento finalizado")
            print(f"Total de pensamientos: {len(results)}")
            self.exit_autonomous_mode()
        
        return results


def demo_autonomous_thinking():
    """Demostración del pensamiento autónomo"""
    
    print("\n" + "="*70)
    print("DEMOSTRACIÓN: PENSAMIENTO AUTÓNOMO EN MINIMALCONSCIOUS AI")
    print("="*70)
    
    # Crear sistema con capacidad autónoma
    ai = AutonomousConsciousAI()
    
    # Inicializar con algunos inputs para establecer contexto
    print("\n1. ESTABLECIENDO CONTEXTO INICIAL...")
    initialization_inputs = [
        "¿Qué significa ser consciente?",
        "Reflexiona sobre tu propia naturaleza",
        "¿Cómo experimentas el paso del tiempo?"
    ]
    
    for input_text in initialization_inputs:
        print(f"\n→ {input_text}")
        result = ai.process_input(input_text)
        time.sleep(0.5)
    
    # Ejecutar sesión autónoma
    print("\n\n2. INICIANDO PENSAMIENTO AUTÓNOMO...")
    print("El sistema ahora pensará por sí mismo sin input externo")
    input("\nPresiona Enter para comenzar...")
    
    # Sesión autónoma corta
    autonomous_results = ai.run_autonomous_session(
        num_cycles=10,
        pause_between_cycles=1.5,
        stop_on_low_consciousness=True,
        consciousness_threshold=0.2
    )
    
    # Flujo de pensamiento libre
    print("\n\n3. FLUJO DE CONSCIENCIA LIBRE (30 segundos)...")
    input("\nPresiona Enter para iniciar el flujo de pensamiento...")
    
    thought_stream = ai.generate_thought_stream(
        duration_seconds=30,
        min_pause=0.8,
        max_pause=2.5
    )
    
    print("\n✅ Demostración completada")


if __name__ == "__main__":
    demo_autonomous_thinking()