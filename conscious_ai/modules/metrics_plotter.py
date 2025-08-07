import matplotlib.pyplot as plt
from typing import List, Dict

def plot_consciousness_metrics(metrics_history: List[Dict[str, float]]):
    """Genera una gráfica de la evolución de las métricas de conciencia"""
    if not metrics_history:
        print("No hay métricas disponibles para graficar.")
        return

    cycles = list(range(1, len(metrics_history) + 1))

    # Extraer métricas
    f_vals = [m['f'] for m in metrics_history]
    phi_vals = [m['Phi'] for m in metrics_history]
    ci_vals = [m['C_i'] for m in metrics_history]
    tu_vals = [m['T_u'] for m in metrics_history]
    r_vals = [m['R'] for m in metrics_history]
    sm_vals = [m['S_m'] for m in metrics_history]
    threshold = metrics_history[0].get('threshold', 1.3)

    plt.figure(figsize=(12, 7))
    plt.plot(cycles, f_vals, label='f (función de conciencia)', linewidth=2)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Umbral θ = {threshold}')

    # Submétricas
    plt.plot(cycles, phi_vals, label='Φ (Irreducibilidad)')
    plt.plot(cycles, ci_vals, label='C_i (Integración Causal)')
    plt.plot(cycles, tu_vals, label='T_u (Unificación Temporal)')
    plt.plot(cycles, r_vals, label='R (Reentrancia)')
    plt.plot(cycles, sm_vals, label='S_m (Self-Model)')

    plt.xlabel('Ciclo')
    plt.ylabel('Valor de métrica')
    plt.title('Evolución de métricas de conciencia por ciclo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# =========================
# Prueba de perturbación controlada
# =========================
def test_self_model_removal(ai_system, perturb_inputs: List[str]):
    """Ejecuta ciclos con el self-model desactivado para analizar su efecto sobre f y Φ"""
    print("\n===== EXPERIMENTO: REMOCIÓN TEMPORAL DEL SELF-MODEL =====")
    original_self_model = ai_system.self_model

    for i, input_text in enumerate(perturb_inputs):
        print(f"\n--- Ciclo perturbado {i+1} ---")
        # Desactivar el self-model artificialmente
        ai_system.self_model = DummySelfModel()  # Reemplazo por dummy
        result = ai_system.process_input(input_text)
        print(f"f = {result['consciousness_metrics']['f']:.3f}, Phi = {result['consciousness_metrics']['Phi']:.3f}")

    # Restaurar el modelo original
    ai_system.self_model = original_self_model

    print("\n--- Ciclo post-recuperación ---")
    result = ai_system.process_input("Recuperando estado interno")
    print(f"f = {result['consciousness_metrics']['f']:.3f}, Phi = {result['consciousness_metrics']['Phi']:.3f}")

class DummySelfModel:
    """Modelo nulo para simular ausencia de self-model"""
    def __init__(self):
        self.internal_state = {'confidence_level': 0.0}
    def update_state(self, *args, **kwargs): pass
    def generate_self_reference(self): return "(self-model desactivado)"
    def get_complexity_measure(self): return 0.0
    def get_state(self):
        return {
            'state_complexity': 0.0,
            'interaction_count': 0,
            'confidence': 0.0,
            'adaptation_level': 0.0
        }
