"""
Detector de idioma para el sistema MinimalConsciousAI
Soporta español e inglés usando heurísticas ligeras para funcionar sin GPU
"""

import re
from typing import Dict, Tuple, List
from collections import Counter


class LanguageDetector:
    """
    Detector de idioma ligero basado en heurísticas.
    Optimizado para español e inglés, funciona eficientemente en CPU.
    """
    
    def __init__(self):
        # Palabras comunes en español
        self.spanish_common_words = {
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'por',
            'con', 'no', 'una', 'los', 'del', 'se', 'las', 'para', 'como',
            'más', 'pero', 'sus', 'al', 'lo', 'todo', 'esta', 'entre',
            'cuando', 'muy', 'sin', 'sobre', 'ser', 'tiene', 'también',
            'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están',
            'estado', 'desde', 'todos', 'fue', 'son', 'había', 'era',
            'qué', 'cómo', 'estoy', 'tengo', 'hacer', 'puede', 'puedo',
            'mi', 'tu', 'él', 'ella', 'nosotros', 'ustedes', 'ellos'
        }
        
        # Palabras comunes en inglés
        self.english_common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one',
            'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
            'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when',
            'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know',
            'take', 'people', 'into', 'year', 'your', 'good', 'some',
            'could', 'them', 'see', 'other', 'than', 'then', 'now',
            'how', 'am', 'is', 'are', 'was', 'were', 'been', 'being'
        }
        
        # Patrones de caracteres específicos
        self.spanish_patterns = [
            r'[ñÑ]',  # ñ es exclusiva del español
            r'[áéíóúÁÉÍÓÚ]',  # acentos españoles
            r'\b(ll|rr)\b',  # dobles consonantes comunes
            r'\¿[^?]*\?',  # signos de interrogación españoles
            r'\¡[^!]*!',  # signos de exclamación españoles
        ]
        
        # Sufijos comunes
        self.spanish_suffixes = {'ción', 'sión', 'dad', 'tad', 'mente', 'miento', 'ando', 'iendo'}
        self.english_suffixes = {'tion', 'sion', 'ness', 'ment', 'ing', 'ly', 'ful', 'less'}
        
    def detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detecta el idioma del texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Tuple de (idioma, confianza) donde idioma es 'es' o 'en'
        """
        if not text or len(text.strip()) == 0:
            return 'en', 0.5  # Default a inglés con baja confianza
        
        # Limpiar y preparar texto
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        if not words:
            return 'en', 0.5
        
        # Contar indicadores
        spanish_score = 0
        english_score = 0
        
        # 1. Palabras comunes
        spanish_word_count = sum(1 for word in words if word in self.spanish_common_words)
        english_word_count = sum(1 for word in words if word in self.english_common_words)
        
        spanish_score += spanish_word_count * 2
        english_score += english_word_count * 2
        
        # 2. Patrones de caracteres españoles
        for pattern in self.spanish_patterns:
            matches = len(re.findall(pattern, text))
            spanish_score += matches * 3  # Mayor peso a caracteres únicos
        
        # 3. Sufijos
        for word in words:
            for suffix in self.spanish_suffixes:
                if word.endswith(suffix):
                    spanish_score += 1
            for suffix in self.english_suffixes:
                if word.endswith(suffix):
                    english_score += 1
        
        # 4. Análisis de bigramas comunes
        spanish_bigrams = {'de la', 'en el', 'de los', 'a la', 'y el', 'que se', 'con el'}
        english_bigrams = {'of the', 'in the', 'to the', 'and the', 'for the', 'with the'}
        
        text_bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        spanish_score += sum(2 for bigram in text_bigrams if bigram in spanish_bigrams)
        english_score += sum(2 for bigram in text_bigrams if bigram in english_bigrams)
        
        # Calcular confianza
        total_score = spanish_score + english_score
        if total_score == 0:
            return 'en', 0.5
        
        spanish_ratio = spanish_score / total_score
        
        # Determinar idioma y confianza
        if spanish_ratio > 0.6:
            return 'es', min(0.95, spanish_ratio)
        elif spanish_ratio < 0.4:
            return 'en', min(0.95, 1 - spanish_ratio)
        else:
            # Caso ambiguo, usar longitud de palabras como desempate
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 5.5:  # Español tiende a tener palabras más largas
                return 'es', 0.6
            else:
                return 'en', 0.6
    
    def get_language_name(self, lang_code: str) -> str:
        """Retorna el nombre completo del idioma"""
        names = {
            'es': 'español',
            'en': 'english'
        }
        return names.get(lang_code, 'unknown')
    
    def analyze_text_features(self, text: str) -> Dict[str, any]:
        """
        Analiza características adicionales del texto útiles para el procesamiento.
        
        Returns:
            Dict con características del texto
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        features = {
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / len(words) if words else 0,
            'has_question': '?' in text or '¿' in text,
            'has_exclamation': '!' in text or '¡' in text,
            'unique_words': len(set(words)),
            'lexical_diversity': len(set(words)) / len(words) if words else 0
        }
        
        # Detectar algunas intenciones básicas
        question_words_es = {'qué', 'cómo', 'cuándo', 'dónde', 'por qué', 'quién', 'cuál'}
        question_words_en = {'what', 'how', 'when', 'where', 'why', 'who', 'which'}
        
        features['has_question_word'] = any(w in question_words_es | question_words_en for w in words)
        
        return features


# Funciones auxiliares para uso directo
_detector = LanguageDetector()

def detect_language(text: str) -> Tuple[str, float]:
    """Función directa para detectar idioma"""
    return _detector.detect_language(text)

def get_text_features(text: str) -> Dict[str, any]:
    """Función directa para obtener características del texto"""
    return _detector.analyze_text_features(text)


if __name__ == "__main__":
    # Pruebas básicas
    test_texts = [
        "Hola, ¿cómo estás hoy?",
        "Hello, how are you today?",
        "Me gustaría saber más sobre inteligencia artificial",
        "I would like to know more about artificial intelligence",
        "¿What is your opinion?",  # Mezclado
        "Creo que esto es fascinante",
        "I think this is fascinating",
        "¿Puedes ayudarme con mi tarea?",
        "Can you help me with my homework?",
    ]
    
    print("=== Pruebas de detección de idioma ===\n")
    for text in test_texts:
        lang, confidence = detect_language(text)
        features = get_text_features(text)
        print(f"Texto: '{text}'")
        print(f"Idioma: {lang} (confianza: {confidence:.2f})")
        print(f"Características: palabras={features['word_count']}, "
              f"pregunta={features['has_question']}, "
              f"diversidad_léxica={features['lexical_diversity']:.2f}")
        print("-" * 50)