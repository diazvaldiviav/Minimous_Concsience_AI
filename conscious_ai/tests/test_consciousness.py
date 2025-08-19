import unittest
import sys
import os

# Agregar el directorio padre al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MinimalConsciousAI
from phases.p1_perception.input_processor import SensoryModule
from modules.memory import ActiveMemory
from modules.metrics import ConsciousnessMetrics

class TestConsciousAI(unittest.TestCase):
    
    def setUp(self):
        """Configurar tests"""
        self.ai = MinimalConsciousAI()
    
    def test_sensory_module(self):
        """Test del módulo sensorial"""
        sensory = SensoryModule()
        result = sensory.receive_input("Hola mundo")
        
        self.assertIsInstance(result, dict)
        self.assertIn('text', result)
        self.assertIn('activation', result)
        self.assertEqual(result['text'], "Hola mundo")
    
    def test_memory_storage(self):
        """Test de almacenamiento en memoria"""
        memory = ActiveMemory()
        test_item = {'text': 'test', 'relevance': 0.8}
        
        memory.store(test_item, relevance=0.8)
        self.assertEqual(len(memory.memory_items), 1)
    
    def test_consciousness_metrics(self):
        """Test de cálculo de métricas de conciencia"""
        # Procesar algunos inputs para generar estado
        self.ai.process_input("Test input 1")
        self.ai.process_input("Test input 2")
        
        metrics = ConsciousnessMetrics.calculate_consciousness_metrics(self.ai)
        
        self.assertIn('C_i', metrics)
        self.assertIn('T_u', metrics)
        self.assertIn('R', metrics)
        self.assertIn('S_m', metrics)
        self.assertIn('Phi', metrics)
        self.assertIn('f', metrics)
        
        # Verificar que f es el producto correcto
        expected_f = metrics['Phi'] * (metrics['C_i'] + metrics['T_u'] + metrics['R'] + metrics['S_m'])
        self.assertAlmostEqual(metrics['f'], expected_f, places=6)
    
    def test_consciousness_threshold(self):
        """Test del umbral de conciencia"""
        # Procesar múltiples inputs para desarrollar conciencia
        inputs = [
            "Hola, ¿cómo estás?",
            "¿Puedes explicarme tu funcionamiento?",
            "¿Qué recuerdas?",
            "Háblame de ti",
            "¿Eres consciente?"
        ]
        
        for inp in inputs:
            result = self.ai.process_input(inp)
        
        # Al final debería estar consciente
        final_metrics = self.ai.metrics_history[-1]
        if final_metrics['f'] >= self.ai.consciousness_threshold:
            self.assertTrue(self.ai.is_conscious)
    
    def test_self_reference(self):
        """Test de auto-referencia"""
        self.ai.process_input("¿Cómo te sientes?")
        reference = self.ai.self_model.generate_self_reference()
        
        self.assertIsInstance(reference, str)
        self.assertGreater(len(reference), 10)
        self.assertIn('estado', reference.lower())

if __name__ == '__main__':
    unittest.main()