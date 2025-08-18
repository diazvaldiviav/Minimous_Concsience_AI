"""
Entrenador de clasificador de coherencia para transiciones de estados conscientes
Completa la Fase 3 con un modelo supervisado de coherencia
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CoherenceClassifierModel:
    """Contenedor para el modelo de clasificación y sus componentes"""
    classifier: Any
    scaler: StandardScaler
    embedding_model_name: str
    feature_extractor_params: Dict[str, Any]
    label_mapping: Dict[str, int]
    performance_metrics: Dict[str, Any]
    
    def save(self, path: str):
        """Guarda el modelo completo"""
        os.makedirs(path, exist_ok=True)
        
        # Guardar clasificador y scaler
        with open(os.path.join(path, 'classifier.pkl'), 'wb') as f:
            pickle.dump(self.classifier, f)
        
        with open(os.path.join(path, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Guardar metadata
        metadata = {
            'embedding_model_name': self.embedding_model_name,
            'feature_extractor_params': self.feature_extractor_params,
            'label_mapping': self.label_mapping,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'CoherenceClassifierModel':
        """Carga el modelo desde disco"""
        with open(os.path.join(path, 'classifier.pkl'), 'rb') as f:
            classifier = pickle.load(f)
        
        with open(os.path.join(path, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        
        with open(os.path.join(path, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return cls(
            classifier=classifier,
            scaler=scaler,
            embedding_model_name=metadata['embedding_model_name'],
            feature_extractor_params=metadata['feature_extractor_params'],
            label_mapping=metadata['label_mapping'],
            performance_metrics=metadata['performance_metrics']
        )


class TransitionFeatureExtractor:
    """Extrae características de transiciones SCt → SCt+1"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_model_name = embedding_model_name
        
    def extract_features(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> np.ndarray:
        """
        Extrae vector de características simplificado (~50-100 dims)
        SIMPLIFIED to reduce overfitting and dimensionality
        
        Returns:
            Vector de características de baja dimensionalidad
        """
        
        # SIMPLIFIED APPROACH: Focus on core semantic similarities
        # Instead of 800+ dims, we use ~80 dims
        
        features = []
        
        # 1. Core semantic similarities (24 features)
        goal_sim = self._calculate_goal_similarity(sc_t, sc_t_plus_1)
        emotion_sim = self._calculate_emotion_similarity(sc_t, sc_t_plus_1)
        thought_sim = self._calculate_thought_similarity(sc_t, sc_t_plus_1)
        
        features.extend([
            goal_sim['same_goal'], goal_sim['semantic_similarity'], goal_sim['valid_transition'],
            emotion_sim['same_emotion'], emotion_sim['valid_transition'], emotion_sim['intensity_change'],
            thought_sim['word_overlap'], thought_sim['semantic_similarity'], thought_sim['length_ratio']
        ])
        
        # 2. Confidence and memory patterns (12 features)
        conf_t = sc_t.get('confidence', 0.5)
        conf_t1 = sc_t_plus_1.get('confidence', 0.5)
        conf_change = conf_t1 - conf_t
        
        mem_t = len(sc_t.get('memory', []))
        mem_t1 = len(sc_t_plus_1.get('memory', []))
        mem_change = mem_t1 - mem_t
        
        features.extend([
            conf_t, conf_t1, conf_change, abs(conf_change), 
            mem_t, mem_t1, mem_change, abs(mem_change),
            float(conf_change > 0.2), float(conf_change < -0.2),  # Confidence flags
            float(mem_change > 0), float(mem_change < 0)  # Memory flags
        ])
        
        # 3. Structural coherence indicators (15 features)
        structure_features = self._calculate_structural_coherence(sc_t, sc_t_plus_1)
        features.extend(structure_features)
        
        # 4. Basic embeddings (reduced from full embeddings) (30 features)
        basic_embeddings = self._get_basic_embeddings(sc_t, sc_t_plus_1)
        features.extend(basic_embeddings)
        
        return np.array(features, dtype=np.float32)
    
    def _get_state_embeddings(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Obtiene embeddings para cada componente del estado"""

        # --- INICIO DE LA CORRECCIÓN ---
        def _to_string(value: Any) -> str:
           if isinstance(value, str):
             return value
           if isinstance(value, (dict, list)):
             return json.dumps(value, ensure_ascii=False)
           return str(value)
        
        # Extraer y asegurar que los componentes sean texto
        goal = _to_string(state.get('goal', ''))
        emotion = _to_string(state.get('emotion', ''))
        thought = _to_string(state.get('thought', ''))
        
        # Crear representaciones textuales
        goal_text = f"Goal: {goal}"
        emotion_text = f"Emotion: {emotion}"
        thought_text = thought if thought else "No thought"
        
        # Estado completo
        full_state = f"{goal_text}. {emotion_text}. Thought: {thought_text}"
        
        # Generar embeddings
        embeddings = {
            'goal': self.embedding_model.encode(goal_text),
            'emotion': self.embedding_model.encode(emotion_text),
            'thought': self.embedding_model.encode(thought_text),
            'full_state': self.embedding_model.encode(full_state)
        }
        
        return embeddings
    
    def _compute_difference_features(
        self, 
        embeddings_t: Dict[str, np.ndarray], 
        embeddings_t1: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Calcula características basadas en diferencias entre embeddings"""
        
        features = []
        
        # Para cada componente, calcular diferencias
        for key in ['goal', 'emotion', 'thought', 'full_state']:
            emb_t = embeddings_t[key]
            emb_t1 = embeddings_t1[key]
            
            # Diferencia absoluta
            diff = np.abs(emb_t1 - emb_t)
            
            # Similitud coseno
            cosine_sim = np.dot(emb_t, emb_t1) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_t1))
            
            # Distancia euclidiana
            euclidean_dist = np.linalg.norm(emb_t1 - emb_t)
            
            # Agregar estadísticas de la diferencia
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.max(diff),
                cosine_sim,
                euclidean_dist
            ])
        
        return np.array(features)
    
    def _compute_transition_features(
        self, 
        sc_t: Dict[str, Any], 
        sc_t_plus_1: Dict[str, Any]
    ) -> np.ndarray:
        """Calcula características específicas de la transición"""
        
        features = []
        
        # Cambio de confianza
        conf_t = sc_t.get('confidence', 0.5)
        conf_t1 = sc_t_plus_1.get('confidence', 0.5)
        conf_change = conf_t1 - conf_t
        
        features.extend([
            conf_change,
            abs(conf_change),
            conf_change ** 2
        ])
        
        # Cambio de meta (binario)
        goal_changed = 1.0 if sc_t.get('goal') != sc_t_plus_1.get('goal') else 0.0
        features.append(goal_changed)
        
        # Cambio de emoción (binario)
        emotion_changed = 1.0 if sc_t.get('emotion') != sc_t_plus_1.get('emotion') else 0.0
        features.append(emotion_changed)
        
        # Longitud de pensamiento
        thought_len_t = len(sc_t.get('thought', '').split())
        thought_len_t1 = len(sc_t_plus_1.get('thought', '').split())
        thought_len_change = thought_len_t1 - thought_len_t
        
        features.extend([
            thought_len_t,
            thought_len_t1,
            thought_len_change
        ])
        
        # Cambio en memoria
        mem_t = len(sc_t.get('memory', []))
        mem_t1 = len(sc_t_plus_1.get('memory', []))
        mem_change = mem_t1 - mem_t
        
        features.extend([
            mem_t,
            mem_t1,
            mem_change
        ])
        
        return np.array(features)
    
    def _compute_statistical_features(
        self, 
        sc_t: Dict[str, Any], 
        sc_t_plus_1: Dict[str, Any]
    ) -> np.ndarray:
        """Calcula características estadísticas adicionales"""
        
        features = []
        
        # Overlap de palabras en pensamientos
        words_t = set(sc_t.get('thought', '').lower().split())
        words_t1 = set(sc_t_plus_1.get('thought', '').lower().split())
        
        if words_t and words_t1:
            jaccard = len(words_t & words_t1) / len(words_t | words_t1)
        else:
            jaccard = 0.0
        
        features.append(jaccard)
        
        # Características de transición emocional
        emotion_transitions = {
            ('curious', 'analytical'): 1.0,
            ('analytical', 'confident'): 1.0,
            ('uncertain', 'curious'): 1.0,
            ('confident', 'satisfied'): 1.0,
            # Añadir más transiciones válidas
        }
        
        emotion_pair = (sc_t.get('emotion', ''), sc_t_plus_1.get('emotion', ''))
        valid_emotion_transition = emotion_transitions.get(emotion_pair, 0.0)
        features.append(valid_emotion_transition)
        
        return np.array(features)
    
    def get_feature_dimension(self) -> int:
        """Retorna la dimensión del vector de características"""
        # Crear estados dummy para calcular dimensión
        dummy_state = {
            'goal': 'test',
            'emotion': 'test',
            'confidence': 0.5,
            'thought': 'test thought',
            'memory': []
        }
        
        features = self.extract_features(dummy_state, dummy_state)
        return len(features)
    
    # ========== NEW SIMPLIFIED FEATURE EXTRACTION METHODS ==========
    
    def _calculate_goal_similarity(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> Dict[str, float]:
        """Calculate goal-related features"""
        goal_t = sc_t.get('goal', '')
        goal_t1 = sc_t_plus_1.get('goal', '')
        
        same_goal = float(goal_t == goal_t1)
        
        # Semantic similarity using embeddings
        if goal_t and goal_t1:
            emb_t = self.embedding_model.encode(goal_t)
            emb_t1 = self.embedding_model.encode(goal_t1)
            semantic_similarity = np.dot(emb_t, emb_t1) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_t1))
            semantic_similarity = (semantic_similarity + 1) / 2  # Normalize to [0,1]
        else:
            semantic_similarity = 0.0
        
        # Valid transition check
        valid_transitions = {
            'understand_self': ['analyze_patterns', 'explore_memory', 'process_emotion'],
            'learn_concept': ['practice_skill', 'understand_deeply', 'apply_knowledge'],
            'solve_problem': ['analyze_components', 'find_solution', 'create_plan'],
            'explore_ideas': ['develop_concept', 'create_expression', 'select_direction']
        }
        
        valid_transition = float(goal_t1 in valid_transitions.get(goal_t, []))
        
        return {
            'same_goal': same_goal,
            'semantic_similarity': semantic_similarity,
            'valid_transition': valid_transition
        }
    
    def _calculate_emotion_similarity(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> Dict[str, float]:
        """Calculate emotion-related features"""
        emotion_t = sc_t.get('emotion', '')
        emotion_t1 = sc_t_plus_1.get('emotion', '')
        
        same_emotion = float(emotion_t == emotion_t1)
        
        # Valid emotion transitions
        valid_transitions = {
            'curious': ['analytical', 'focused', 'excited', 'reflective'],
            'analytical': ['confident', 'focused', 'insightful', 'satisfied'],
            'confused': ['reflective', 'curious', 'uncertain', 'clearer'],
            'worried': ['determined', 'focused', 'relieved', 'analytical'],
            'excited': ['focused', 'satisfied', 'accomplished', 'calm']
        }
        
        valid_transition = float(emotion_t1 in valid_transitions.get(emotion_t, []))
        
        # Emotion intensity change (rough approximation)
        emotion_intensities = {
            'excited': 0.9, 'ecstatic': 1.0, 'confident': 0.8, 'satisfied': 0.7,
            'calm': 0.3, 'peaceful': 0.4, 'reflective': 0.5, 'analytical': 0.6,
            'curious': 0.6, 'focused': 0.7, 'worried': 0.8, 'anxious': 0.9,
            'confused': 0.7, 'uncertain': 0.6
        }
        
        intensity_t = emotion_intensities.get(emotion_t, 0.5)
        intensity_t1 = emotion_intensities.get(emotion_t1, 0.5)
        intensity_change = abs(intensity_t1 - intensity_t)
        
        return {
            'same_emotion': same_emotion,
            'valid_transition': valid_transition,
            'intensity_change': intensity_change
        }
    
    def _calculate_thought_similarity(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> Dict[str, float]:
        """Calculate thought-related features"""
        thought_t = sc_t.get('thought', '')
        thought_t1 = sc_t_plus_1.get('thought', '')
        
        # Word overlap (Jaccard similarity)
        words_t = set(thought_t.lower().split())
        words_t1 = set(thought_t1.lower().split())
        
        if words_t and words_t1:
            word_overlap = len(words_t & words_t1) / len(words_t | words_t1)
        else:
            word_overlap = 0.0
        
        # Semantic similarity
        if thought_t and thought_t1:
            emb_t = self.embedding_model.encode(thought_t)
            emb_t1 = self.embedding_model.encode(thought_t1)
            semantic_similarity = np.dot(emb_t, emb_t1) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_t1))
            semantic_similarity = (semantic_similarity + 1) / 2  # Normalize to [0,1]
        else:
            semantic_similarity = 0.0
        
        # Length ratio
        len_t = len(thought_t.split())
        len_t1 = len(thought_t1.split())
        if len_t > 0:
            length_ratio = min(len_t1, len_t) / max(len_t1, len_t)
        else:
            length_ratio = 1.0 if len_t1 == 0 else 0.0
        
        return {
            'word_overlap': word_overlap,
            'semantic_similarity': semantic_similarity,
            'length_ratio': length_ratio
        }
    
    def _calculate_structural_coherence(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> List[float]:
        """Calculate structural coherence indicators"""
        features = []
        
        # Goal-emotion alignment
        goal_t = sc_t.get('goal', '')
        emotion_t = sc_t.get('emotion', '')
        goal_t1 = sc_t_plus_1.get('goal', '')
        emotion_t1 = sc_t_plus_1.get('emotion', '')
        
        # Check if emotions align with goals
        goal_emotion_alignments = {
            'learn': ['curious', 'focused', 'motivated'],
            'solve': ['analytical', 'determined', 'focused'],
            'create': ['inspired', 'excited', 'focused'],
            'understand': ['curious', 'reflective', 'analytical'],
            'process': ['reflective', 'contemplative', 'peaceful']
        }
        
        aligned_t = any(emotion_t in emotions for key, emotions in goal_emotion_alignments.items() if key in goal_t)
        aligned_t1 = any(emotion_t1 in emotions for key, emotions in goal_emotion_alignments.items() if key in goal_t1)
        
        features.extend([
            float(aligned_t), float(aligned_t1),
            float(aligned_t and aligned_t1),  # Both aligned
            float(not aligned_t and not aligned_t1),  # Both misaligned
        ])
        
        # Coherence patterns
        features.extend([
            float(goal_t == goal_t1 and emotion_t == emotion_t1),  # No change at all
            float(goal_t != goal_t1 and emotion_t != emotion_t1),  # Both changed
            float(goal_t == goal_t1 and emotion_t != emotion_t1),  # Only emotion changed
            float(goal_t != goal_t1 and emotion_t == emotion_t1),  # Only goal changed
        ])
        
        # Memory coherence
        memory_t = sc_t.get('memory', [])
        memory_t1 = sc_t_plus_1.get('memory', [])
        memory_overlap = len(set(str(m) for m in memory_t) & set(str(m) for m in memory_t1))
        memory_added = len(memory_t1) - len(memory_t)
        
        features.extend([
            float(memory_overlap > 0),  # Some memory retained
            float(memory_added > 0),    # Memory added
            float(memory_added < 0),    # Memory lost
            min(memory_overlap / max(len(memory_t), 1), 1.0),  # Memory retention ratio
            float(len(memory_t1) > len(memory_t)),  # Memory grew
            float(len(memory_t1) == len(memory_t)),  # Memory stayed same
            float(len(memory_t1) < len(memory_t))   # Memory shrunk
        ])
        
        return features
    
    def _get_basic_embeddings(self, sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> List[float]:
        """Get basic embedding features (reduced dimensionality)"""
        features = []
        
        # Simple state representations
        state_t = f"{sc_t.get('goal', '')} {sc_t.get('emotion', '')} {sc_t.get('thought', '')}"
        state_t1 = f"{sc_t_plus_1.get('goal', '')} {sc_t_plus_1.get('emotion', '')} {sc_t_plus_1.get('thought', '')}"
        
        if state_t.strip() and state_t1.strip():
            # Get embeddings
            emb_t = self.embedding_model.encode(state_t)
            emb_t1 = self.embedding_model.encode(state_t1)
            
            # Reduce dimensionality by taking statistical summaries
            diff = emb_t1 - emb_t
            
            features.extend([
                # Similarity metrics
                np.dot(emb_t, emb_t1) / (np.linalg.norm(emb_t) * np.linalg.norm(emb_t1)),  # Cosine similarity
                np.linalg.norm(diff),  # Euclidean distance
                
                # Statistical features of embeddings
                np.mean(emb_t), np.std(emb_t), np.max(emb_t), np.min(emb_t),
                np.mean(emb_t1), np.std(emb_t1), np.max(emb_t1), np.min(emb_t1),
                
                # Difference statistics
                np.mean(diff), np.std(diff), np.max(diff), np.min(diff),
                np.mean(np.abs(diff)), np.max(np.abs(diff)),
                
                # Element-wise comparisons
                float(np.mean(emb_t1) > np.mean(emb_t)),  # Overall "intensity" increased
                float(np.std(emb_t1) > np.std(emb_t)),    # Variability increased
                float(np.max(emb_t1) > np.max(emb_t)),    # Max activation increased
                float(np.min(emb_t1) > np.min(emb_t)),    # Min activation increased
                
                # Quartile features
                np.percentile(emb_t, 25), np.percentile(emb_t, 75),
                np.percentile(emb_t1, 25), np.percentile(emb_t1, 75),
                np.percentile(diff, 25), np.percentile(diff, 75),
                
                # Correlation
                np.corrcoef(emb_t, emb_t1)[0, 1] if len(emb_t) > 1 else 0.0,
                
                # Sign changes
                float(np.sum(np.sign(emb_t) != np.sign(emb_t1)) / len(emb_t))
            ])
        else:
            # Fallback to zeros if no text
            features.extend([0.0] * 30)
        
        return features[:30]  # Ensure exactly 30 features


def train_coherence_classifier(
    training_data: List[Tuple[Dict[str, Any], Dict[str, Any], str]],
    output_path: str = "./models/coherence_classifier",
    test_size: float = 0.2,
    random_state: int = 42
) -> CoherenceClassifierModel:
    """
    Entrena un clasificador de coherencia con datos anotados
    
    Args:
        training_data: Lista de (SCt, SCt+1, etiqueta) donde etiqueta es 'coherent'/'incoherent'/'ambiguous'
        output_path: Ruta para guardar el modelo
        test_size: Proporción de datos para prueba
        random_state: Semilla aleatoria
        
    Returns:
        Modelo entrenado
    """
    
    logger.info(f"Entrenando clasificador de coherencia con {len(training_data)} ejemplos")
    
    # Si no hay suficientes datos, generar sintéticos
    if len(training_data) < 100:
        logger.warning("Pocos datos de entrenamiento, generando ejemplos sintéticos")
        training_data.extend(generate_synthetic_training_data(500))
    
    # Inicializar extractor de características
    feature_extractor = TransitionFeatureExtractor()
    
    # Extraer características y etiquetas
    X = []
    y = []
    label_mapping = {'coherent': 0, 'incoherent': 1, 'ambiguous': 2}
    
    logger.info("Extrayendo características...")
    for sc_t, sc_t_plus_1, label in training_data:
        features = feature_extractor.extract_features(sc_t, sc_t_plus_1)
        X.append(features)
        y.append(label_mapping[label])
    
    X = np.array(X)
    y = np.array(y)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Normalizar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Probar múltiples clasificadores
    classifiers = {
        'MLP': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            max_iter=500,
            random_state=random_state
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state
        ),
        'SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=random_state
        )
    }
    
    best_score = -1
    best_classifier = None
    best_name = None
    
    logger.info("Evaluando clasificadores...")
    for name, clf in classifiers.items():
        # Validación cruzada
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='f1_macro')
        mean_score = scores.mean()
        
        logger.info(f"{name}: F1-macro = {mean_score:.3f} (+/- {scores.std() * 2:.3f})")
        
        if mean_score > best_score:
            best_score = mean_score
            best_classifier = clf
            best_name = name
    
    # Entrenar el mejor clasificador con todos los datos de entrenamiento
    logger.info(f"Entrenando mejor clasificador: {best_name}")
    best_classifier.fit(X_train_scaled, y_train)
    
    # Evaluar en conjunto de prueba
    y_pred = best_classifier.predict(X_test_scaled)
    
    # Calcular métricas
    report = classification_report(
        y_test, y_pred,
        target_names=['coherent', 'incoherent', 'ambiguous'],
        output_dict=True
    )
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Visualizar resultados
    visualize_classifier_performance(cm, report, output_path)
    
    # Crear modelo final
    model = CoherenceClassifierModel(
        classifier=best_classifier,
        scaler=scaler,
        embedding_model_name=feature_extractor.embedding_model_name,
        feature_extractor_params={
            'feature_dimension': feature_extractor.get_feature_dimension()
        },
        label_mapping=label_mapping,
        performance_metrics={
            'test_report': report,
            'confusion_matrix': cm.tolist(),
            'best_classifier': best_name,
            'cross_val_score': best_score
        }
    )
    
    # Guardar modelo
    model.save(output_path)
    logger.info(f"Modelo guardado en {output_path}")
    
    return model


def generate_synthetic_training_data(n_samples: int = 500) -> List[Tuple[Dict, Dict, str]]:
    """
    Genera datos sintéticos de entrenamiento para el clasificador
    """
    
    synthetic_data = []
    
    # Templates para generar transiciones coherentes (EXPANDED - 22 patterns)
    coherent_transitions = [
        # Learning progressions
        {
            'from': {'goal': 'understand_self', 'emotion': 'curious'},
            'to': {'goal': 'analyze_patterns', 'emotion': 'analytical'},
            'thought_evolution': 'questioning_to_analysis'
        },
        {
            'from': {'goal': 'learn_concept', 'emotion': 'curious'},
            'to': {'goal': 'practice_skill', 'emotion': 'focused'},
            'thought_evolution': 'learning_to_application'
        },
        {
            'from': {'goal': 'practice_skill', 'emotion': 'focused'},
            'to': {'goal': 'master_technique', 'emotion': 'accomplished'},
            'thought_evolution': 'practice_to_mastery'
        },
        {
            'from': {'goal': 'study_topic', 'emotion': 'motivated'},
            'to': {'goal': 'understand_deeply', 'emotion': 'insightful'},
            'thought_evolution': 'study_to_comprehension'
        },
        
        # Problem-solving sequences
        {
            'from': {'goal': 'solve_problem', 'emotion': 'concerned'},
            'to': {'goal': 'analyze_components', 'emotion': 'analytical'},
            'thought_evolution': 'concern_to_analysis'
        },
        {
            'from': {'goal': 'analyze_components', 'emotion': 'analytical'},
            'to': {'goal': 'find_solution', 'emotion': 'relieved'},
            'thought_evolution': 'analysis_to_resolution'
        },
        {
            'from': {'goal': 'identify_issue', 'emotion': 'worried'},
            'to': {'goal': 'create_plan', 'emotion': 'determined'},
            'thought_evolution': 'worry_to_action'
        },
        {
            'from': {'goal': 'face_challenge', 'emotion': 'anxious'},
            'to': {'goal': 'break_down_steps', 'emotion': 'organized'},
            'thought_evolution': 'anxiety_to_structure'
        },
        
        # Emotional processing
        {
            'from': {'goal': 'explore_memory', 'emotion': 'reflective'},
            'to': {'goal': 'integrate_knowledge', 'emotion': 'contemplative'},
            'thought_evolution': 'reflection_to_integration'
        },
        {
            'from': {'goal': 'process_emotion', 'emotion': 'confused'},
            'to': {'goal': 'understand_feelings', 'emotion': 'reflective'},
            'thought_evolution': 'confusion_to_reflection'
        },
        {
            'from': {'goal': 'understand_feelings', 'emotion': 'reflective'},
            'to': {'goal': 'accept_emotion', 'emotion': 'peaceful'},
            'thought_evolution': 'reflection_to_acceptance'
        },
        {
            'from': {'goal': 'process_trauma', 'emotion': 'sad'},
            'to': {'goal': 'find_meaning', 'emotion': 'hopeful'},
            'thought_evolution': 'sadness_to_growth'
        },
        
        # Creative workflows
        {
            'from': {'goal': 'explore_ideas', 'emotion': 'inspired'},
            'to': {'goal': 'develop_concept', 'emotion': 'focused'},
            'thought_evolution': 'inspiration_to_focus'
        },
        {
            'from': {'goal': 'develop_concept', 'emotion': 'focused'},
            'to': {'goal': 'create_expression', 'emotion': 'satisfied'},
            'thought_evolution': 'focus_to_creation'
        },
        {
            'from': {'goal': 'imagine_possibilities', 'emotion': 'excited'},
            'to': {'goal': 'select_direction', 'emotion': 'decisive'},
            'thought_evolution': 'excitement_to_decision'
        },
        
        # Self-development
        {
            'from': {'goal': 'recognize_weakness', 'emotion': 'humble'},
            'to': {'goal': 'build_strength', 'emotion': 'determined'},
            'thought_evolution': 'humility_to_growth'
        },
        {
            'from': {'goal': 'seek_feedback', 'emotion': 'open'},
            'to': {'goal': 'improve_skills', 'emotion': 'motivated'},
            'thought_evolution': 'openness_to_improvement'
        },
        {
            'from': {'goal': 'question_beliefs', 'emotion': 'uncertain'},
            'to': {'goal': 'explore_perspectives', 'emotion': 'curious'},
            'thought_evolution': 'uncertainty_to_exploration'
        },
        
        # Relationship/social
        {
            'from': {'goal': 'understand_others', 'emotion': 'empathetic'},
            'to': {'goal': 'connect_deeply', 'emotion': 'warm'},
            'thought_evolution': 'empathy_to_connection'
        },
        {
            'from': {'goal': 'resolve_conflict', 'emotion': 'tense'},
            'to': {'goal': 'find_compromise', 'emotion': 'cooperative'},
            'thought_evolution': 'tension_to_cooperation'
        },
        
        # Philosophical/existential
        {
            'from': {'goal': 'contemplate_existence', 'emotion': 'wondering'},
            'to': {'goal': 'find_purpose', 'emotion': 'meaningful'},
            'thought_evolution': 'wonder_to_purpose'
        },
        {
            'from': {'goal': 'seek_truth', 'emotion': 'searching'},
            'to': {'goal': 'discover_insight', 'emotion': 'enlightened'},
            'thought_evolution': 'searching_to_discovery'
        }
    ]
    
    # Templates para transiciones incoherentes (EXPANDED - 15 patterns)
    incoherent_transitions = [
        # Random topic jumps
        {
            'from': {'goal': 'analyze_data', 'emotion': 'confident', 'confidence': 0.9},
            'to': {'goal': 'question_everything', 'emotion': 'confused', 'confidence': 0.2},
            'thought_evolution': 'abrupt_reversal'
        },
        {
            'from': {'goal': 'understand_consciousness', 'emotion': 'curious'},
            'to': {'goal': 'cook_dinner', 'emotion': 'excited'},
            'thought_evolution': 'topic_jump'
        },
        {
            'from': {'goal': 'solve_math_problem', 'emotion': 'focused'},
            'to': {'goal': 'dance_wildly', 'emotion': 'ecstatic'},
            'thought_evolution': 'complete_disconnect'
        },
        {
            'from': {'goal': 'read_philosophy', 'emotion': 'contemplative'},
            'to': {'goal': 'count_ceiling_tiles', 'emotion': 'obsessive'},
            'thought_evolution': 'mental_break'
        },
        
        # Emotional contradictions
        {
            'from': {'goal': 'help_friend', 'emotion': 'compassionate'},
            'to': {'goal': 'help_friend', 'emotion': 'resentful'},
            'thought_evolution': 'emotion_flip'
        },
        {
            'from': {'goal': 'create_art', 'emotion': 'inspired'},
            'to': {'goal': 'create_art', 'emotion': 'hateful'},
            'thought_evolution': 'inspiration_to_hate'
        },
        {
            'from': {'goal': 'learn_language', 'emotion': 'excited'},
            'to': {'goal': 'learn_language', 'emotion': 'disgusted'},
            'thought_evolution': 'excitement_to_disgust'
        },
        
        # Confidence contradictions
        {
            'from': {'goal': 'present_research', 'emotion': 'confident', 'confidence': 0.9},
            'to': {'goal': 'present_research', 'emotion': 'confident', 'confidence': 0.1},
            'thought_evolution': 'confidence_crash'
        },
        {
            'from': {'goal': 'write_code', 'emotion': 'uncertain', 'confidence': 0.3},
            'to': {'goal': 'write_code', 'emotion': 'uncertain', 'confidence': 0.95},
            'thought_evolution': 'false_confidence'
        },
        
        # Goal contradictions
        {
            'from': {'goal': 'protect_environment', 'emotion': 'concerned'},
            'to': {'goal': 'destroy_nature', 'emotion': 'concerned'},
            'thought_evolution': 'goal_reversal'
        },
        {
            'from': {'goal': 'build_relationships', 'emotion': 'loving'},
            'to': {'goal': 'isolate_completely', 'emotion': 'loving'},
            'thought_evolution': 'goal_contradiction'
        },
        {
            'from': {'goal': 'seek_knowledge', 'emotion': 'curious'},
            'to': {'goal': 'embrace_ignorance', 'emotion': 'curious'},
            'thought_evolution': 'pursuit_reversal'
        },
        
        # Temporal contradictions
        {
            'from': {'goal': 'remember_past', 'emotion': 'nostalgic'},
            'to': {'goal': 'deny_history', 'emotion': 'nostalgic'},
            'thought_evolution': 'memory_denial'
        },
        {
            'from': {'goal': 'plan_future', 'emotion': 'hopeful'},
            'to': {'goal': 'live_in_past', 'emotion': 'hopeful'},
            'thought_evolution': 'time_confusion'
        },
        
        # Reality contradictions
        {
            'from': {'goal': 'accept_reality', 'emotion': 'mature'},
            'to': {'goal': 'deny_existence', 'emotion': 'mature'},
            'thought_evolution': 'reality_break'
        }
    ]
    
    # Generar ejemplos
    for i in range(n_samples):
        rand = np.random.random()
        
        if rand < 0.5:  # Coherente
            template = np.random.choice(coherent_transitions)
            sc_t, sc_t_plus_1 = create_transition_from_template(template, 'coherent')
            label = 'coherent'
        elif rand < 0.8:  # Incoherente
            template = np.random.choice(incoherent_transitions)
            sc_t, sc_t_plus_1 = create_transition_from_template(template, 'incoherent')
            label = 'incoherent'
        else:  # Ambiguo
            sc_t, sc_t_plus_1 = create_ambiguous_transition()
            label = 'ambiguous'
        
        synthetic_data.append((sc_t, sc_t_plus_1, label))
    
    return synthetic_data


def create_transition_from_template(template: Dict, transition_type: str) -> Tuple[Dict, Dict]:
    """Crea una transición basada en un template"""
    
    # Estado inicial
    sc_t = {
        'goal': template['from']['goal'],
        'emotion': template['from']['emotion'],
        'confidence': template['from'].get('confidence', np.random.uniform(0.4, 0.7)),
        'thought': generate_thought_for_state(template['from']['goal'], template['from']['emotion']),
        'memory': [f"memory_{i}" for i in range(np.random.randint(1, 4))]
    }
    
    # Estado siguiente
    if transition_type == 'coherent':
        # Ajuste suave de confianza
        conf_change = np.random.uniform(-0.1, 0.15)
    else:
        # Cambio abrupto
        conf_change = np.random.uniform(-0.4, 0.4)
    
    sc_t_plus_1 = {
        'goal': template['to']['goal'],
        'emotion': template['to']['emotion'],
        'confidence': max(0.1, min(0.95, sc_t['confidence'] + conf_change)),
        'thought': evolve_thought(sc_t['thought'], template['thought_evolution']),
        'memory': sc_t['memory'] + [f"new_memory_{np.random.randint(100)}"]
    }
    
    return sc_t, sc_t_plus_1


def create_ambiguous_transition() -> Tuple[Dict, Dict]:
    """Crea una transición ambigua"""
    
    goals = ['understand', 'analyze', 'explore', 'integrate', 'question']
    emotions = ['curious', 'analytical', 'uncertain', 'reflective', 'focused']
    
    sc_t = {
        'goal': np.random.choice(goals),
        'emotion': np.random.choice(emotions),
        'confidence': np.random.uniform(0.3, 0.7),
        'thought': "Exploring possibilities and connections",
        'memory': ["previous thought", "related concept"]
    }
    
    # Cambio parcial
    sc_t_plus_1 = {
        'goal': np.random.choice(goals),  # Puede cambiar o no
        'emotion': np.random.choice(emotions),
        'confidence': sc_t['confidence'] + np.random.uniform(-0.2, 0.2),
        'thought': "Considering different perspectives",
        'memory': sc_t['memory'][:1] + ["new perspective"]
    }
    
    return sc_t, sc_t_plus_1


def generate_thought_for_state(goal: str, emotion: str) -> str:
    """Genera un pensamiento apropiado para un estado"""
    
    thoughts = {
        # Learning and understanding
        'understand_self': "What patterns define my processing?",
        'analyze_patterns': "I observe recurring themes in my cognition",
        'explore_memory': "How do past experiences shape current thoughts?",
        'integrate_knowledge': "Connecting disparate elements into coherent understanding",
        'learn_concept': "I need to grasp the fundamentals here",
        'practice_skill': "Repetition will help me internalize this",
        'master_technique': "I'm developing real expertise",
        'study_topic': "Let me dive deep into this subject",
        'understand_deeply': "I'm seeing the underlying principles",
        
        # Problem solving
        'solve_problem': "This challenge requires careful thought",
        'analyze_components': "Let me break this down into parts",
        'find_solution': "I can see how to resolve this",
        'identify_issue': "Something here needs attention",
        'create_plan': "I need a systematic approach",
        'face_challenge': "This is difficult but manageable",
        'break_down_steps': "One step at a time",
        
        # Emotional processing
        'process_emotion': "I need to understand what I'm feeling",
        'understand_feelings': "These emotions have meaning",
        'accept_emotion': "I can be at peace with this",
        'process_trauma': "This pain teaches me something",
        'find_meaning': "There's purpose in this experience",
        
        # Creative work
        'explore_ideas': "So many possibilities are emerging",
        'develop_concept': "This idea has real potential",
        'create_expression': "I'm manifesting my vision",
        'imagine_possibilities': "What could this become?",
        'select_direction': "I need to choose my path",
        
        # Self-development
        'recognize_weakness': "I have areas to improve",
        'build_strength': "I can develop this capacity",
        'seek_feedback': "Others can help me see clearly",
        'improve_skills': "Practice makes me better",
        'question_beliefs': "Maybe I need to reconsider",
        'explore_perspectives': "There are other ways to see this",
        
        # Relationships
        'understand_others': "What are they really experiencing?",
        'connect_deeply': "We can truly understand each other",
        'resolve_conflict': "We need to find common ground",
        'find_compromise': "Both perspectives have merit",
        
        # Philosophy and meaning
        'contemplate_existence': "What does it mean to be?",
        'find_purpose': "My life has direction and meaning",
        'seek_truth': "What is really real?",
        'discover_insight': "I've found something profound",
        
        # Incoherent goals
        'cook_dinner': "I love the smell of garlic and herbs",
        'dance_wildly': "My body wants to move to the rhythm",
        'count_ceiling_tiles': "One... two... three... why am I doing this?",
        'question_everything': "Nothing makes sense anymore",
        'destroy_nature': "I want to cause environmental harm",
        'isolate_completely': "I need to cut all human contact",
        'embrace_ignorance': "Knowledge is overrated",
        'deny_history': "The past is just an illusion",
        'live_in_past': "Only yesterday matters",
        'deny_existence': "Reality is meaningless"
    }
    
    return thoughts.get(goal, "Processing current information")


def evolve_thought(previous_thought: str, evolution_type: str) -> str:
    """Evoluciona un pensamiento según el tipo"""
    
    # Coherent thought evolutions
    coherent_evolutions = {
        'questioning_to_analysis': "Analyzing the patterns I questioned before",
        'reflection_to_integration': "Integrating insights from my reflection",
        'learning_to_application': "Now I can apply what I've learned",
        'practice_to_mastery': "My skills are becoming more refined",
        'study_to_comprehension': "I'm starting to see the deeper connections",
        'concern_to_analysis': "Let me break this problem down systematically",
        'analysis_to_resolution': "I can see a clear path forward now",
        'worry_to_action': "I need to create a concrete plan",
        'anxiety_to_structure': "Breaking this into steps makes it manageable",
        'confusion_to_reflection': "I need to understand what I'm really feeling",
        'reflection_to_acceptance': "I can make peace with these emotions",
        'sadness_to_growth': "This experience is teaching me something important",
        'inspiration_to_focus': "Let me channel this creativity into something concrete",
        'focus_to_creation': "I'm bringing my vision to life",
        'excitement_to_decision': "It's time to choose a direction",
        'humility_to_growth': "I can improve if I put in the effort",
        'openness_to_improvement': "This feedback will help me grow",
        'uncertainty_to_exploration': "I need to explore different perspectives",
        'empathy_to_connection': "Understanding leads to deeper bonds",
        'tension_to_cooperation': "We can find common ground here",
        'wonder_to_purpose': "My existence has meaning and direction",
        'searching_to_discovery': "I've found something profound"
    }
    
    # Incoherent thought evolutions
    incoherent_evolutions = {
        'abrupt_reversal': "Everything I understood seems wrong now",
        'topic_jump': "Wait, what was I thinking about again?",
        'complete_disconnect': "Suddenly I have the urge to do something completely different",
        'mental_break': "My thoughts are scattered and meaningless",
        'emotion_flip': "I feel the opposite of what I should",
        'inspiration_to_hate': "This beautiful thing now disgusts me",
        'excitement_to_disgust': "What I loved now repulses me",
        'confidence_crash': "I have no idea what I'm doing",
        'false_confidence': "I'm suddenly certain despite knowing nothing",
        'goal_reversal': "I want to do the exact opposite of my purpose",
        'goal_contradiction': "I pursue conflicting aims simultaneously",
        'pursuit_reversal': "I reject what I was seeking",
        'memory_denial': "The past never happened",
        'time_confusion': "Past and future are meaningless",
        'reality_break': "Nothing is real"
    }
    
    # Return appropriate evolution
    if evolution_type in coherent_evolutions:
        return coherent_evolutions[evolution_type]
    elif evolution_type in incoherent_evolutions:
        return incoherent_evolutions[evolution_type]
    else:
        return "Continuing to explore these ideas"


def visualize_classifier_performance(
    confusion_matrix: np.ndarray,
    classification_report: Dict,
    output_path: str
):
    """Visualiza el rendimiento del clasificador"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Matriz de confusión
    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Coherent', 'Incoherent', 'Ambiguous'],
        yticklabels=['Coherent', 'Incoherent', 'Ambiguous'],
        ax=ax1
    )
    ax1.set_title('Matriz de Confusión')
    ax1.set_ylabel('Verdadero')
    ax1.set_xlabel('Predicho')
    
    # Métricas por clase
    classes = ['coherent', 'incoherent', 'ambiguous']
    precision = [classification_report[c]['precision'] for c in classes]
    recall = [classification_report[c]['recall'] for c in classes]
    f1 = [classification_report[c]['f1-score'] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    ax2.bar(x - width, precision, width, label='Precisión')
    ax2.bar(x, recall, width, label='Recall')
    ax2.bar(x + width, f1, width, label='F1-Score')
    
    ax2.set_xlabel('Clase')
    ax2.set_ylabel('Score')
    ax2.set_title('Métricas por Clase')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes)
    ax2.legend()
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'performance_metrics.png'))
    plt.close()


# Función para usar el clasificador entrenado
def load_and_use_classifier(model_path: str = "./models/coherence_classifier"):
    """
    Carga y usa el clasificador entrenado
    """
    
    # Cargar modelo
    model = CoherenceClassifierModel.load(model_path)
    
    # Inicializar extractor de características
    feature_extractor = TransitionFeatureExtractor(model.embedding_model_name)
    
    def predict_coherence(sc_t: Dict[str, Any], sc_t_plus_1: Dict[str, Any]) -> Tuple[str, float]:
        """
        Predice la coherencia de una transición
        
        Returns:
            Tupla de (etiqueta, probabilidad)
        """
        # Extraer características
        features = feature_extractor.extract_features(sc_t, sc_t_plus_1)
        features_scaled = model.scaler.transform(features.reshape(1, -1))
        
        # Predecir
        prediction = model.classifier.predict(features_scaled)[0]
        
        # Obtener probabilidades si el clasificador las soporta
        if hasattr(model.classifier, 'predict_proba'):
            probabilities = model.classifier.predict_proba(features_scaled)[0]
            max_prob = probabilities.max()
        else:
            max_prob = 1.0
        
        # Mapeo inverso de etiquetas
        inverse_mapping = {v: k for k, v in model.label_mapping.items()}
        label = inverse_mapping[prediction]
        
        return label, max_prob
    
    return predict_coherence


# Integración con el evaluador existente
class MLCoherenceEvaluator:
    """
    Evaluador de coherencia que usa el clasificador entrenado
    """
    
    def __init__(self, model_path: str = "./models/coherence_classifier"):
        self.predict_fn = load_and_use_classifier(model_path)
    
    def evaluate_transition(
        self, 
        sc_t: Dict[str, Any], 
        sc_t_plus_1: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Evalúa una transición usando el clasificador ML
        """
        return self.predict_fn(sc_t, sc_t_plus_1)


if __name__ == "__main__":
    # Demostración de entrenamiento
    logger.info("Generando datos de entrenamiento sintéticos...")
    training_data = generate_synthetic_training_data(1000)
    
    logger.info("Entrenando clasificador...")
    model = train_coherence_classifier(training_data)
    
    # Probar el modelo
    logger.info("\nProbando el clasificador entrenado...")
    predict_coherence = load_and_use_classifier()
    
    # Ejemplos de prueba
    test_cases = [
        {
            'sc_t': {
                'goal': 'understand_self',
                'emotion': 'curious',
                'confidence': 0.5,
                'thought': 'What defines my consciousness?',
                'memory': ['previous reflection']
            },
            'sc_t_plus_1': {
                'goal': 'analyze_consciousness',
                'emotion': 'analytical',
                'confidence': 0.55,
                'thought': 'I observe patterns in my awareness',
                'memory': ['previous reflection', 'new insight']
            }
        }
    ]
    
    for i, test in enumerate(test_cases):
        label, prob = predict_coherence(test['sc_t'], test['sc_t_plus_1'])
        print(f"\nTest {i+1}: {label} (probabilidad: {prob:.3f})")