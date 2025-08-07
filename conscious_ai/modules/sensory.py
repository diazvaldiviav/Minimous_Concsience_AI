from typing import Dict, Any
from datetime import datetime

class SensoryModule:
    """Módulo sensorial - maneja entrada textual externa"""
    
    def __init__(self):
        self.current_input = ""
        self.input_history = []
        self.activation_level = 0.0
        
    def receive_input(self, text_input: str) -> Dict[str, Any]:
        """Procesa entrada textual y extrae características"""
        self.current_input = text_input
        self.input_history.append({
            'text': text_input,
            'timestamp': datetime.now(),
            'length': len(text_input),
            'complexity': len(text_input.split())
        })
        
        # Calcular nivel de activación basado en complejidad
        self.activation_level = min(1.0, len(text_input) / 100.0)
        
        # Extraer características semánticas básicas
        features = {
            'text': text_input,
            'word_count': len(text_input.split()),
            'has_question': '?' in text_input,
            'has_emotion': any(word in text_input.lower() for word in 
                             ['bueno', 'malo', 'feliz', 'triste', 'enojado']),
            'activation': self.activation_level,
            'timestamp': datetime.now()
        }
        
        return features
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'current_input': self.current_input,
            'activation_level': self.activation_level,
            'history_length': len(self.input_history)
        }