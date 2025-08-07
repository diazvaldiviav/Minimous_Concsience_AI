"""
Generador de dataset sintético para entrenar el modelo de consciencia
Crea ejemplos en español e inglés con anotaciones de SCt
"""

import json
import random
from typing import List, Dict, Any
from datetime import datetime


class TrainingDatasetGenerator:
    """Genera dataset sintético para entrenar el modelo de inferencia SCt"""
    
    def __init__(self):
        # Aquí va todo el bloque original provisto por Claude, sin cambios
        self.templates = {
            'es': {
                'greeting': [
                    {"input": "Hola, ¿cómo estás?", "goal": "social_interaction", "emotion": "amigable", "confidence": 0.8, "thought": "El usuario busca establecer contacto social"},
                    {"input": "Buenos días", "goal": "greeting", "emotion": "cordial", "confidence": 0.9, "thought": "Saludo matutino formal"},
                    {"input": "¿Qué tal estás hoy?", "goal": "check_wellbeing", "emotion": "interesado", "confidence": 0.75, "thought": "Pregunta sobre estado emocional"},
                ],
                'question': [
                    {"input": "¿Qué es la conciencia?", "goal": "seek_knowledge", "emotion": "curioso", "confidence": 0.7, "thought": "Pregunta filosófica profunda sobre la naturaleza de la experiencia"},
                    {"input": "¿Cómo funciona esto?", "goal": "understand_mechanism", "emotion": "confundido", "confidence": 0.6, "thought": "Necesita explicación de funcionamiento"},
                    {"input": "¿Puedes explicarme el concepto?", "goal": "clarification", "emotion": "estudiando", "confidence": 0.65, "thought": "Solicita elaboración conceptual"},
                    {"input": "¿Por qué sucede este fenómeno?", "goal": "understand_causality", "emotion": "analítico", "confidence": 0.7, "thought": "Busca relaciones causa-efecto"},
                ],
                'reflection': [
                    {"input": "Me siento confundido con todo esto", "goal": "express_confusion", "emotion": "confundido", "confidence": 0.4, "thought": "Estado mental de incertidumbre expresado"},
                    {"input": "Creo que estoy empezando a entender", "goal": "report_progress", "emotion": "esperanzado", "confidence": 0.6, "thought": "Progreso cognitivo en marcha"},
                    {"input": "Esto me hace pensar en mi propia experiencia", "goal": "self_reflect", "emotion": "introspectivo", "confidence": 0.7, "thought": "Conectando con experiencia personal"},
                    {"input": "No estoy seguro de comprenderlo completamente", "goal": "express_uncertainty", "emotion": "inseguro", "confidence": 0.45, "thought": "Reconocimiento de limitaciones cognitivas"},
                ],
                'request': [
                    {"input": "Necesito ayuda con este problema", "goal": "seek_assistance", "emotion": "necesitado", "confidence": 0.5, "thought": "Solicitud directa de apoyo"},
                    {"input": "¿Podrías darme un ejemplo?", "goal": "request_example", "emotion": "pragmático", "confidence": 0.7, "thought": "Busca concreción mediante ejemplos"},
                    {"input": "Ayúdame a entender mejor", "goal": "deepen_understanding", "emotion": "determinado", "confidence": 0.65, "thought": "Motivación por comprender"},
                ],
                'emotional': [
                    {"input": "Me siento muy feliz hoy", "goal": "share_emotion", "emotion": "feliz", "confidence": 0.9, "thought": "Estado emocional positivo alto"},
                    {"input": "Estoy algo preocupado por esto", "goal": "express_concern", "emotion": "preocupado", "confidence": 0.55, "thought": "Ansiedad moderada expresada"},
                    {"input": "Esto me frustra mucho", "goal": "vent_frustration", "emotion": "frustrado", "confidence": 0.8, "thought": "Emoción negativa intensa"},
                    {"input": "Me emociona aprender cosas nuevas", "goal": "express_enthusiasm", "emotion": "emocionado", "confidence": 0.85, "thought": "Motivación intrínseca por aprendizaje"},
                ],
                'analytical': [
                    {"input": "Si analizamos los componentes del sistema", "goal": "analyze_system", "emotion": "analítico", "confidence": 0.75, "thought": "Enfoque sistemático de descomposición"},
                    {"input": "Comparando ambas opciones", "goal": "compare_options", "emotion": "evaluativo", "confidence": 0.7, "thought": "Proceso de evaluación comparativa"},
                    {"input": "La evidencia sugiere que", "goal": "present_evidence", "emotion": "objetivo", "confidence": 0.8, "thought": "Razonamiento basado en datos"},
                ],
            },
            'en': {
                'greeting': [
                    {"input": "Hello, how are you?", "goal": "social_interaction", "emotion": "friendly", "confidence": 0.8, "thought": "User seeks social connection"},
                    {"input": "Good morning", "goal": "greeting", "emotion": "polite", "confidence": 0.9, "thought": "Formal morning greeting"},
                    {"input": "How's it going?", "goal": "check_wellbeing", "emotion": "casual", "confidence": 0.75, "thought": "Informal status check"},
                ],
                'question': [
                    {"input": "What is consciousness?", "goal": "seek_knowledge", "emotion": "curious", "confidence": 0.7, "thought": "Deep philosophical inquiry about experience"},
                    {"input": "How does this work?", "goal": "understand_mechanism", "emotion": "puzzled", "confidence": 0.6, "thought": "Needs functional explanation"},
                    {"input": "Can you explain the concept?", "goal": "clarification", "emotion": "studious", "confidence": 0.65, "thought": "Requests conceptual elaboration"},
                    {"input": "Why does this happen?", "goal": "understand_causality", "emotion": "analytical", "confidence": 0.7, "thought": "Seeking cause-effect relationships"},
                ],
                'reflection': [
                    {"input": "I feel confused about all this", "goal": "express_confusion", "emotion": "confused", "confidence": 0.4, "thought": "Mental state of uncertainty expressed"},
                    {"input": "I think I'm starting to understand", "goal": "report_progress", "emotion": "hopeful", "confidence": 0.6, "thought": "Cognitive progress underway"},
                    {"input": "This makes me think about my own experience", "goal": "self_reflect", "emotion": "introspective", "confidence": 0.7, "thought": "Connecting to personal experience"},
                    {"input": "I'm not sure I fully grasp it", "goal": "express_uncertainty", "emotion": "uncertain", "confidence": 0.45, "thought": "Recognition of cognitive limitations"},
                ],
                'request': [
                    {"input": "I need help with this problem", "goal": "seek_assistance", "emotion": "needful", "confidence": 0.5, "thought": "Direct request for support"},
                    {"input": "Could you give me an example?", "goal": "request_example", "emotion": "pragmatic", "confidence": 0.7, "thought": "Seeking concreteness through examples"},
                    {"input": "Help me understand better", "goal": "deepen_understanding", "emotion": "determined", "confidence": 0.65, "thought": "Motivation to comprehend"},
                ],
                'emotional': [
                    {"input": "I feel really happy today", "goal": "share_emotion", "emotion": "happy", "confidence": 0.9, "thought": "High positive emotional state"},
                    {"input": "I'm somewhat worried about this", "goal": "express_concern", "emotion": "worried", "confidence": 0.55, "thought": "Moderate anxiety expressed"},
                    {"input": "This frustrates me a lot", "goal": "vent_frustration", "emotion": "frustrated", "confidence": 0.8, "thought": "Intense negative emotion"},
                    {"input": "I'm excited to learn new things", "goal": "express_enthusiasm", "emotion": "excited", "confidence": 0.85, "thought": "Intrinsic learning motivation"},
                ],
                'analytical': [
                    {"input": "If we analyze the system components", "goal": "analyze_system", "emotion": "analytical", "confidence": 0.75, "thought": "Systematic decomposition approach"},
                    {"input": "Comparing both options", "goal": "compare_options", "emotion": "evaluative", "confidence": 0.7, "thought": "Comparative evaluation process"},
                    {"input": "The evidence suggests that", "goal": "present_evidence", "emotion": "objective", "confidence": 0.8, "thought": "Data-driven reasoning"},
                ],
            }
        }
        self.emotion_variations = {
            'es': {
                'curioso': ['intrigado', 'inquisitivo', 'explorador'],
                'confundido': ['perplejo', 'desconcertado', 'desorientado'],
                'feliz': ['alegre', 'contento', 'satisfecho'],
                'preocupado': ['ansioso', 'inquieto', 'nervioso'],
                'analítico': ['reflexivo', 'metódico', 'sistemático'],
            },
            'en': {
                'curious': ['intrigued', 'inquisitive', 'exploratory'],
                'confused': ['perplexed', 'puzzled', 'disoriented'],
                'happy': ['joyful', 'pleased', 'satisfied'],
                'worried': ['anxious', 'uneasy', 'nervous'],
                'analytical': ['reflective', 'methodical', 'systematic'],
            }
        }
        self.goal_expansions = {
            'seek_knowledge': ['explore_concept', 'understand_theory', 'learn_facts'],
            'express_confusion': ['report_difficulty', 'signal_incomprehension', 'request_clarity'],
            'self_reflect': ['introspect', 'examine_thoughts', 'analyze_feelings'],
        }

    def generate_variations(self, base_example: Dict[str, Any], num_variations: int = 3) -> List[Dict[str, Any]]:
        """Genera variaciones de un ejemplo base"""
        variations = [base_example]
        for _ in range(num_variations):
            variation = base_example.copy()
            variation['confidence'] = max(0.1, min(0.95, base_example['confidence']
                + random.uniform(-0.15, 0.15)))
            thought_parts = base_example['thought'].split()
            if len(thought_parts) > 3:
                insertions = ['claramente', 'probablemente', 'posiblemente', 'definitivamente']
                if random.random() > 0.5:
                    insert_pos = random.randint(1, len(thought_parts)-1)
                    thought_parts.insert(insert_pos, random.choice(insertions))
                variation['thought'] = ' '.join(thought_parts)
            lang = 'es' if any(c in base_example['input'] for c in 'áéíóúñ¿¡') else 'en'
            if base_example['emotion'] in self.emotion_variations.get(lang, {}):
                alternatives = self.emotion_variations[lang][base_example['emotion']
                ]
                variation['emotion'] = random.choice([base_example['emotion']] + alternatives)
            variations.append(variation)
        return variations

    def generate_mixed_examples(self, num_examples: int = 20) -> List[Dict[str, Any]]:
        """Genera ejemplos mezclando características"""
        mixed_examples = []
        for _ in range(num_examples):
            lang = random.choice(['es', 'en'])
            cat1, cat2 = random.sample(list(self.templates[lang].keys()), 2)
            base1 = random.choice(self.templates[lang][cat1])
            base2 = random.choice(self.templates[lang][cat2])
            mixed = {
                'input': base1['input'],
                'goal': random.choice([base1['goal'], base2['goal']]),
                'emotion': random.choice([base1['emotion'], base2['emotion']]),
                'confidence': (base1['confidence'] + base2['confidence']) / 2,
                'thought': random.choice([base1['thought'], base2['thought']])
            }
            mixed_examples.append(mixed)
        return mixed_examples

    def generate_dataset(self, total_examples: int = 1000) -> List[Dict[str, Any]]:
        """Genera el dataset completo"""
        all_examples = []
        for lang in ['es', 'en']:
            for category, examples in self.templates[lang].items():
                for example in examples:
                    variations = self.generate_variations(example, num_variations=2)
                    all_examples.extend(variations)
        mixed = self.generate_mixed_examples(num_examples=100)
        all_examples.extend(mixed)
        random.shuffle(all_examples)
        return all_examples[:total_examples]

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str = "training_data.jsonl"):
        """Guarda el dataset en formato JSONL"""
        with open(filename, 'w', encoding='utf-8') as f:
            for example in dataset:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        print(f"Dataset guardado en {filename} con {len(dataset)} ejemplos")

    def load_dataset(self, filename: str = "training_data.jsonl") -> List[Dict[str, Any]]:
        """Carga dataset desde archivo JSONL"""
        dataset = []
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        return dataset

    def analyze_dataset(self, dataset: List[Dict[str, Any]]):
        """Analiza distribución del dataset"""
        print("\n=== Análisis del Dataset ===")
        print(f"Total de ejemplos: {len(dataset)}")
        spanish_count = sum(1 for e in dataset if any(c in e['input'] for c in 'áéíóúñ¿¡'))
        english_count = len(dataset) - spanish_count
        print(f"Ejemplos en español: {spanish_count} ({spanish_count/len(dataset)*100:.1f}%)")
        print(f"Ejemplos en inglés: {english_count} ({english_count/len(dataset)*100:.1f}%)")

        goals = {}
        for example in dataset:
            goals[example['goal']] = goals.get(example['goal'], 0) + 1
        print("\nDistribución de metas (top 10):")
        for goal, count in sorted(goals.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {goal}: {count} ({count/len(dataset)*100:.1f}%)")

        emotions = {}
        for example in dataset:
            emotions[example['emotion']] = emotions.get(example['emotion'], 0) + 1
        print("\nDistribución de emociones (top 10):")
        for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {emotion}: {count} ({count/len(dataset)*100:.1f}%)")

def create_training_dataset(output_file: str = "training_data.jsonl", size: int = 1000):
    generator = TrainingDatasetGenerator()
    print("Generando dataset de entrenamiento...")
    dataset = generator.generate_dataset(total_examples=size)
    print(f"Guardando {len(dataset)} ejemplos...")
    generator.save_dataset(dataset, output_file)
    generator.analyze_dataset(dataset)
    return dataset


if __name__ == "__main__":
    dataset = create_training_dataset(size=1000)
    print(f"\n✓ Dataset creado exitosamente con {len(dataset)} ejemplos")
        