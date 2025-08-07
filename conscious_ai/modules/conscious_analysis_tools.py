"""
Herramientas avanzadas para análisis del flujo consciente
Fase II: Análisis de patrones, visualización y métricas complejas
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import seaborn as sns
from conscious_ai.modules.conscious_state import ConsciousState, ConsciousStateHistory, semantic_distance


class ConsciousFlowAnalyzer:
    """
    Analizador avanzado de flujos de conciencia.
    Detecta patrones complejos y genera visualizaciones detalladas.
    """
    
    def __init__(self, history: ConsciousStateHistory):
        self.history = history
        
    def compute_transition_matrix(self, 
                                 feature: str = 'emotional_state',
                                 normalize: bool = True) -> Dict[str, Any]:
        """
        Calcula matriz de transición entre estados.
        
        Args:
            feature: Característica a analizar ('emotional_state', 'attention_focus', etc.)
            normalize: Si normalizar las probabilidades
            
        Returns:
            Dict con matriz y estados únicos
        """
        if len(self.history.history) < 2:
            return {'matrix': np.array([]), 'states': []}
        
        # Extraer secuencia de estados
        if feature.startswith('S_t.'):
            feature_key = feature.split('.')[-1]
            sequence = [s.S_t.get(feature_key) for s in self.history.history]
        else:
            sequence = self.history.get_trajectory(feature)
        
        # Estados únicos
        unique_states = sorted(list(set(s for s in sequence if s is not None)))
        n_states = len(unique_states)
        
        if n_states == 0:
            return {'matrix': np.array([]), 'states': []}
        
        # Construir matriz de transición
        transition_matrix = np.zeros((n_states, n_states))
        state_to_idx = {state: i for i, state in enumerate(unique_states)}
        
        for i in range(len(sequence) - 1):
            if sequence[i] is not None and sequence[i+1] is not None:
                from_idx = state_to_idx[sequence[i]]
                to_idx = state_to_idx[sequence[i+1]]
                transition_matrix[from_idx, to_idx] += 1
        
        # Normalizar si se solicita
        if normalize:
            row_sums = transition_matrix.sum(axis=1)
            row_sums[row_sums == 0] = 1  # Evitar división por cero
            transition_matrix = transition_matrix / row_sums[:, np.newaxis]
        
        return {
            'matrix': transition_matrix,
            'states': unique_states,
            'sequence': sequence
        }
    
    def detect_cycles(self, min_length: int = 3, max_distance: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detecta ciclos repetitivos en el flujo consciente.
        
        Args:
            min_length: Longitud mínima del ciclo
            max_distance: Distancia máxima para considerar estados similares
            
        Returns:
            Lista de ciclos detectados
        """
        cycles_found = []
        states = self.history.history
        
        if len(states) < min_length * 2:
            return cycles_found
        
        # Buscar patrones repetitivos
        for start_i in range(len(states) - min_length * 2):
            for length in range(min_length, (len(states) - start_i) // 2 + 1):
                pattern = states[start_i:start_i + length]
                
                # Buscar repeticiones del patrón
                for check_i in range(start_i + length, len(states) - length + 1):
                    candidate = states[check_i:check_i + length]
                    
                    # Comparar secuencias
                    is_similar = True
                    total_distance = 0.0
                    
                    for j in range(length):
                        distance = semantic_distance(pattern[j], candidate[j])
                        total_distance += distance
                        if distance > max_distance:
                            is_similar = False
                            break
                    
                    if is_similar:
                        avg_distance = total_distance / length
                        cycles_found.append({
                            'start_cycle': pattern[0].cycle,
                            'repeat_cycle': candidate[0].cycle,
                            'length': length,
                            'average_distance': avg_distance,
                            'pattern_summary': [s.get_summary() for s in pattern]
                        })
        
        return cycles_found
    
    def compute_entropy(self, feature: str = 'emotional_state', window: int = 10) -> List[float]:
        """
        Calcula entropía del flujo consciente a través del tiempo.
        
        Args:
            feature: Característica a analizar
            window: Tamaño de ventana deslizante
            
        Returns:
            Lista de valores de entropía
        """
        sequence = self.history.get_trajectory(f'S_t.{feature}')
        
        if len(sequence) < window:
            return []
        
        entropies = []
        
        for i in range(len(sequence) - window + 1):
            window_data = sequence[i:i + window]
            
            # Calcular distribución de probabilidad
            unique_values = list(set(window_data))
            counts = {val: window_data.count(val) for val in unique_values}
            total = sum(counts.values())
            
            # Calcular entropía
            entropy = 0.0
            for count in counts.values():
                if count > 0:
                    p = count / total
                    entropy -= p * np.log2(p)
            
            entropies.append(entropy)
        
        return entropies
    
    def find_bifurcation_points(self, threshold: float = 0.6) -> List[Dict[str, Any]]:
        """
        Identifica puntos de bifurcación (cambios abruptos) en el flujo.
        
        Args:
            threshold: Umbral de distancia para considerar bifurcación
            
        Returns:
            Lista de puntos de bifurcación
        """
        bifurcations = []
        states = self.history.history
        
        if len(states) < 3:
            return bifurcations
        
        for i in range(1, len(states) - 1):
            # Distancia con estado anterior
            dist_prev = semantic_distance(states[i-1], states[i])
            
            # Distancia con estado siguiente
            dist_next = semantic_distance(states[i], states[i+1])
            
            # Detectar cambio abrupto
            if dist_prev > threshold or dist_next > threshold:
                # Analizar naturaleza del cambio
                change_analysis = self._analyze_state_change(states[i-1], states[i], states[i+1])
                
                bifurcations.append({
                    'cycle': states[i].cycle,
                    'distance_prev': dist_prev,
                    'distance_next': dist_next,
                    'type': 'abrupt' if dist_prev > threshold else 'divergent',
                    'changes': change_analysis
                })
        
        return bifurcations
    
    def _analyze_state_change(self, prev: ConsciousState, curr: ConsciousState, 
                             next: ConsciousState) -> Dict[str, Any]:
        """Analiza la naturaleza de un cambio entre estados"""
        changes = {
            'sensory': semantic_distance(prev, curr) if prev else 1.0,
            'memory_shift': len(set(m['content'].get('text', '')[:20] for m in curr.M_t) - 
                           set(m['content'].get('text', '')[:20] for m in prev.M_t)) if prev else 0,
            'emotional': curr.S_t.get('emotional_state') != prev.S_t.get('emotional_state') if prev else False,
            'confidence_delta': abs(curr.S_t.get('confidence_level', 0.5) - 
                                  prev.S_t.get('confidence_level', 0.5)) if prev else 0
        }
        
        return changes
    
    def generate_phase_space_plot(self, feature1: str = 'confidence_level', 
                                 feature2: str = 'f', save_path: Optional[str] = None):
        """
        Genera gráfico de espacio de fases del flujo consciente.
        
        Args:
            feature1, feature2: Características para los ejes
            save_path: Ruta opcional para guardar
        """
        # Extraer trayectorias
        traj1 = self.history.get_trajectory(f'S_t.{feature1}')
        traj2 = self.history.get_trajectory(f'metrics.{feature2}')
        
        if len(traj1) < 2 or len(traj2) < 2:
            print("Datos insuficientes para espacio de fases")
            return
        
        # Asegurar mismo tamaño
        min_len = min(len(traj1), len(traj2))
        traj1 = traj1[:min_len]
        traj2 = traj2[:min_len]
        
        # Crear figura
        plt.figure(figsize=(10, 8))
        
        # Trayectoria con color gradiente
        cycles = list(range(len(traj1)))
        scatter = plt.scatter(traj1, traj2, c=cycles, cmap='viridis', s=50, alpha=0.7)
        
        # Conectar puntos consecutivos
        for i in range(len(traj1) - 1):
            plt.plot([traj1[i], traj1[i+1]], [traj2[i], traj2[i+1]], 
                    'k-', alpha=0.3, linewidth=0.5)
        
        # Marcar inicio y fin
        plt.scatter(traj1[0], traj2[0], color='green', s=200, marker='o', 
                   edgecolor='black', linewidth=2, label='Inicio')
        plt.scatter(traj1[-1], traj2[-1], color='red', s=200, marker='s', 
                   edgecolor='black', linewidth=2, label='Actual')
        
        plt.xlabel(feature1.replace('_', ' ').title())
        plt.ylabel(feature2.upper() if len(feature2) <= 3 else feature2.replace('_', ' ').title())
        plt.title('Espacio de Fases del Flujo Consciente')
        plt.colorbar(scatter, label='Ciclo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def generate_comprehensive_report(self) -> str:
        """
        Genera reporte comprehensivo del análisis del flujo consciente.
        
        Returns:
            str: Reporte formateado
        """
        report = "=== ANÁLISIS COMPREHENSIVO DEL FLUJO CONSCIENTE ===\n\n"
        
        # 1. Estadísticas básicas
        report += "1. ESTADÍSTICAS BÁSICAS\n"
        report += f"   - Total de estados: {len(self.history.history)}\n"
        
        if self.history.history:
            first = self.history.history[0]
            last = self.history.history[-1]
            duration = (last.timestamp - first.timestamp).total_seconds()
            report += f"   - Duración: {duration:.1f} segundos\n"
            report += f"   - Ciclos: {first.cycle} → {last.cycle}\n"
        
        # 2. Análisis de estabilidad
        stability = self.history.analyze_stability()
        report += "\n2. ANÁLISIS DE ESTABILIDAD\n"
        for key, value in stability.items():
            report += f"   - {key}: {value:.3f}\n"
        
        # 3. Transiciones emocionales
        transitions = self.compute_transition_matrix('emotional_state')
        if transitions['states']:
            report += "\n3. TRANSICIONES EMOCIONALES\n"
            report += f"   Estados: {transitions['states']}\n"
            report += "   Matriz de transición:\n"
            for i, from_state in enumerate(transitions['states']):
                for j, to_state in enumerate(transitions['states']):
                    prob = transitions['matrix'][i, j]
                    if prob > 0:
                        report += f"     {from_state} → {to_state}: {prob:.2f}\n"
        
        # 4. Ciclos detectados
        cycles = self.detect_cycles()
        report += f"\n4. CICLOS REPETITIVOS\n"
        if cycles:
            for i, cycle in enumerate(cycles[:3]):  # Top 3
                report += f"   Ciclo {i+1}:\n"
                report += f"     - Inicio: {cycle['start_cycle']}, Repetición: {cycle['repeat_cycle']}\n"
                report += f"     - Longitud: {cycle['length']} estados\n"
                report += f"     - Distancia promedio: {cycle['average_distance']:.3f}\n"
        else:
            report += "   No se detectaron ciclos repetitivos\n"
        
        # 5. Puntos de bifurcación
        bifurcations = self.find_bifurcation_points()
        report += f"\n5. PUNTOS DE BIFURCACIÓN\n"
        if bifurcations:
            for bif in bifurcations[:3]:  # Top 3
                report += f"   - Ciclo {bif['cycle']}: {bif['type']}\n"
                report += f"     Distancias: prev={bif['distance_prev']:.3f}, "
                report += f"next={bif['distance_next']:.3f}\n"
        else:
            report += "   No se detectaron bifurcaciones significativas\n"
        
        # 6. Entropía
        entropy = self.compute_entropy()
        if entropy:
            report += f"\n6. ANÁLISIS DE ENTROPÍA\n"
            report += f"   - Entropía promedio: {np.mean(entropy):.3f}\n"
            report += f"   - Tendencia: {'creciente' if entropy[-1] > entropy[0] else 'decreciente'}\n"
        
        # 7. Análisis de Metas (G_t)
        report += f"\n7. ANÁLISIS DE METAS (G_t)\n"
        if self.history.history:
            goals = [s.G_t.get('primary_goal', 'none') for s in self.history.history]
            unique_goals = list(set(goals))
            report += f"   - Metas únicas: {unique_goals}\n"
            report += f"   - Cambios de meta: {sum(1 for i in range(1, len(goals)) if goals[i] != goals[i-1])}\n"
            
            # Estabilidad promedio de metas
            goal_stabilities = [s.G_t.get('goal_stability', 0) for s in self.history.history]
            if goal_stabilities:
                report += f"   - Estabilidad promedio: {np.mean(goal_stabilities):.3f}\n"
        
        # 8. Análisis de Pensamientos Automáticos (A_t)
        report += f"\n8. ANÁLISIS DE PENSAMIENTOS AUTOMÁTICOS (A_t)\n"
        if self.history.history:
            thought_counts = [len(s.A_t) for s in self.history.history]
            total_thoughts = sum(thought_counts)
            report += f"   - Total generados: {total_thoughts}\n"
            report += f"   - Promedio por ciclo: {total_thoughts/len(self.history.history):.1f}\n"
            report += f"   - Máximo en un ciclo: {max(thought_counts) if thought_counts else 0}\n"
        
        return report


def visualize_conscious_flow_3d(history: ConsciousStateHistory, 
                               features: List[str] = ['confidence_level', 'f', 'adaptation_level'],
                               save_path: Optional[str] = None):
    """
    Visualización 3D del flujo consciente.
    
    Args:
        history: Historial de estados conscientes
        features: Lista de 3 características para los ejes
        save_path: Ruta opcional para guardar
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    if len(history.history) < 2:
        print("Historial insuficiente para visualización 3D")
        return
    
    # Extraer datos
    data_x = history.get_trajectory(f'S_t.{features[0]}')
    data_y = history.get_trajectory(f'metrics.{features[1]}')
    data_z = history.get_trajectory(f'S_t.{features[2]}')
    
    # Asegurar mismo tamaño
    min_len = min(len(data_x), len(data_y), len(data_z))
    data_x = data_x[:min_len]
    data_y = data_y[:min_len]
    data_z = data_z[:min_len]
    
    # Crear figura 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trayectoria con gradiente de color
    cycles = list(range(len(data_x)))
    scatter = ax.scatter(data_x, data_y, data_z, c=cycles, cmap='plasma', s=50, alpha=0.7)
    
    # Conectar puntos
    for i in range(len(data_x) - 1):
        ax.plot([data_x[i], data_x[i+1]], 
               [data_y[i], data_y[i+1]], 
               [data_z[i], data_z[i+1]], 
               'k-', alpha=0.2, linewidth=0.5)
    
    # Marcar inicio y fin
    ax.scatter(data_x[0], data_y[0], data_z[0], color='green', s=200, 
              marker='o', edgecolor='black', linewidth=2)
    ax.scatter(data_x[-1], data_y[-1], data_z[-1], color='red', s=200, 
              marker='s', edgecolor='black', linewidth=2)
    
    ax.set_xlabel(features[0].replace('_', ' ').title())
    ax.set_ylabel(features[1].upper())
    ax.set_zlabel(features[2].replace('_', ' ').title())
    ax.set_title('Trayectoria 3D del Flujo Consciente')
    
    fig.colorbar(scatter, label='Ciclo', pad=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def generate_consciousness_heatmap(history: ConsciousStateHistory, save_path: Optional[str] = None):
    """
    Genera mapa de calor de la evolución de métricas de conciencia.
    
    Args:
        history: Historial de estados conscientes
        save_path: Ruta opcional para guardar
    """
    if len(history.history) < 2:
        print("Historial insuficiente para mapa de calor")
        return
    
    # Extraer métricas
    metrics_names = ['C_i', 'T_u', 'R', 'S_m', 'Phi', 'f']
    data = []
    
    for metric in metrics_names:
        trajectory = history.get_trajectory(f'metrics.{metric}')
        if trajectory:
            data.append(trajectory)
    
    # Añadir métricas de G_t
    goal_stability = history.get_trajectory('G_t.goal_stability')
    if goal_stability:
        data.append(goal_stability)
        metrics_names.append('Goal_Stability')
    
    # Añadir conteo de A_t
    thought_counts = [len(state.A_t) for state in history.history]
    if thought_counts:
        # Normalizar a escala 0-1
        max_thoughts = max(thought_counts) if thought_counts else 1
        normalized_thoughts = [t/max_thoughts for t in thought_counts]
        data.append(normalized_thoughts)
        metrics_names.append('Thought_Activity')
    
    if not data:
        print("No hay métricas disponibles")
        return
    
    # Crear matriz de datos
    data_matrix = np.array(data)
    
    # Crear figura
    plt.figure(figsize=(14, 8))
    
    # Mapa de calor
    sns.heatmap(data_matrix, 
                xticklabels=range(1, len(data_matrix[0]) + 1),
                yticklabels=metrics_names,
                cmap='YlOrRd',
                cbar_kws={'label': 'Valor'},
                fmt='.2f')
    
    plt.xlabel('Ciclo')
    plt.ylabel('Métrica')
    plt.title('Evolución de Métricas de Conciencia (incluyendo G_t y A_t)')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()