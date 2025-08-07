import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class ActiveMemory:
    """Memoria activa - almacena información relevante por múltiples ciclos"""
    
    def __init__(self, capacity: int = 10, decay_rate: float = 0.1):
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.memory_items = []
        self.relevance_threshold = 0.3
        
    def store(self, item: Dict[str, Any], relevance: float = 1.0):
        """Almacena un elemento en memoria con nivel de relevancia"""
        memory_item = {
            'content': item,
            'relevance': relevance,
            'timestamp': datetime.now(),
            'access_count': 0,
            'cycles_active': 0
        }
        
        self.memory_items.append(memory_item)
        
        # Mantener capacidad limitada
        if len(self.memory_items) > self.capacity:
            self._cleanup_memory()
    
    def update_cycle(self):
        """Actualiza memoria en cada ciclo - aplica decay y incrementa ciclos activos"""
        for item in self.memory_items:
            item['relevance'] *= (1 - self.decay_rate)
            item['cycles_active'] += 1
        
        # Remover items con relevancia muy baja
        self.memory_items = [item for item in self.memory_items 
                           if item['relevance'] > self.relevance_threshold]
    
    def retrieve_relevant(self, query_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recupera elementos relevantes basándose en características de consulta"""
        relevant_items = []
        
        for item in self.memory_items:
            # Incrementar contador de acceso
            item['access_count'] += 1
            
            # Calcular relevancia contextual simple
            contextual_relevance = self._calculate_contextual_relevance(
                item['content'], query_features
            )
            
            if contextual_relevance > 0.5:
                relevant_items.append({
                    'content': item['content'],
                    'relevance': item['relevance'] * contextual_relevance,
                    'cycles_active': item['cycles_active']
                })
        
        # Ordenar por relevancia
        relevant_items.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant_items[:10]  # Top 5 más relevantes
    
    def _calculate_contextual_relevance(self, memory_content: Dict[str, Any], 
                                      query_features: Dict[str, Any]) -> float:
        """Calcula relevancia contextual entre memoria y consulta actual"""
        relevance = 0.0
        
        # Relevancia por tipo de contenido similar
        if isinstance(memory_content, dict) and isinstance(query_features, dict):
            common_keys = set(memory_content.keys()) & set(query_features.keys())
            relevance += len(common_keys) * 0.1
            
            # Relevancia por contenido textual
            if 'text' in memory_content and 'text' in query_features:
                memory_words = set(str(memory_content['text']).lower().split())
                query_words = set(str(query_features['text']).lower().split())
                word_overlap = len(memory_words & query_words)
                if word_overlap > 0:
                    relevance += min(0.8, word_overlap * 0.2)
        
        return min(1.0, relevance)
    
    def _cleanup_memory(self):
        """Limpia memoria manteniendo los elementos más relevantes"""
        self.memory_items.sort(key=lambda x: x['relevance'] * x['access_count'], reverse=True)
        self.memory_items = self.memory_items[:self.capacity]
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'item_count': len(self.memory_items),
            'total_relevance': sum(item['relevance'] for item in self.memory_items),
            'avg_cycles_active': np.mean([item['cycles_active'] for item in self.memory_items]) if self.memory_items else 0
        }
    
    def reset_memory(self):
     """Simula pérdida total de memoria"""
     self.memory_items = []
    

    def clear(self):
     """Elimina completamente todos los elementos de la memoria."""
     self.memory_items = []

