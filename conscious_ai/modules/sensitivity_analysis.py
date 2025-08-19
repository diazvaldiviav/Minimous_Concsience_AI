# Fase 6: Análisis de sensibilidad - sobre ciclo estable (ej. ciclo 8)

def sensibilidad_metricas(ciclo_datos: dict) -> None:
    """
    Recibe un diccionario con las métricas de conciencia de un ciclo (f, C_i, T_u, R, S_m, Phi)
    e imprime el impacto de poner cada componente a 0 individualmente.
    """
    f_original = ciclo_datos['f']
    componentes = ['C_i', 'T_u', 'R', 'S_m', 'Phi']

    print("\n===== ANÁLISIS DE SENSIBILIDAD FUNCIONAL =====")
    print(f"f original = {f_original:.4f}\n")

    for comp in componentes:
        datos_modificados = ciclo_datos.copy()
        original_valor = datos_modificados[comp]
        datos_modificados[comp] = 0.0

        # Calcular nuevo f con ese componente anulado
        if comp == 'Phi':
            f_mod = 0.0  # si phi es 0, f = 0
        else:
            f_mod = ciclo_datos['Phi'] * (
                (datos_modificados['C_i'] +
                 datos_modificados['T_u'] +
                 datos_modificados['R'] +
                 datos_modificados['S_m'])
            )

        sensibilidad = (f_original - f_mod) / original_valor if original_valor > 0 else 'N/A'

        print(f"Componente anulado: {comp}")
        print(f"\tf modificado = {f_mod:.4f}")
        print(f"\tSensibilidad relativa = {sensibilidad if sensibilidad=='N/A' else f'{sensibilidad:.3f}'}\n")


# Ejemplo de uso con datos reales de un ciclo estable (antes de perturbación)
ciclo_8 = {
    'C_i': 0.86,
    'T_u': 0.75,
    'R': 0.64,
    'S_m': 0.55,
    'Phi': 0.84,
    'f': 0.84 * (0.86 + 0.75 + 0.64 + 0.55)
}

if __name__ == "__main__":
    sensibilidad_metricas(ciclo_8)
