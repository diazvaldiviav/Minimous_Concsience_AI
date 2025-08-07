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
        Extrae vector de características de una transición
        
        Returns:
            Vector de características concatenado
        """
        
        # 1. Embeddings de componentes individuales
        embeddings_t = self._get_state_embeddings(sc_t)
        embeddings_t1 = self._get_state_embeddings(sc_t_plus_1)
        
        # 2. Características de diferencia
        diff_features = self._compute_difference_features(embeddings_t, embeddings_t1)
        
        # 3. Características de transición
        transition_features = self._compute_transition_features(sc_t, sc_t_plus_1)
        
        # 4. Características estadísticas
        statistical_features = self._compute_statistical_features(sc_t, sc_t_plus_1)
        
        # Concatenar todas las características
        feature_vector = np.concatenate([
            diff_features,
            transition_features,
            statistical_features
        ])
        
        return feature_vector
    
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
    
    # Templates para generar transiciones coherentes
    coherent_transitions = [
        {
            'from': {'goal': 'understand_self', 'emotion': 'curious'},
            'to': {'goal': 'analyze_patterns', 'emotion': 'analytical'},
            'thought_evolution': 'questioning_to_analysis'
        },
        {
            'from': {'goal': 'explore_memory', 'emotion': 'reflective'},
            'to': {'goal': 'integrate_knowledge', 'emotion': 'contemplative'},
            'thought_evolution': 'reflection_to_integration'
        }
    ]
    
    # Templates para transiciones incoherentes
    incoherent_transitions = [
        {
            'from': {'goal': 'analyze_data', 'emotion': 'confident', 'confidence': 0.9},
            'to': {'goal': 'question_everything', 'emotion': 'confused', 'confidence': 0.2},
            'thought_evolution': 'abrupt_reversal'
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
        'understand_self': "What patterns define my processing?",
        'analyze_patterns': "I observe recurring themes in my cognition",
        'explore_memory': "How do past experiences shape current thoughts?",
        'integrate_knowledge': "Connecting disparate elements into coherent understanding"
    }
    
    return thoughts.get(goal, "Processing current information")


def evolve_thought(previous_thought: str, evolution_type: str) -> str:
    """Evoluciona un pensamiento según el tipo"""
    
    if evolution_type == 'questioning_to_analysis':
        return "Analyzing the patterns I questioned before"
    elif evolution_type == 'reflection_to_integration':
        return "Integrating insights from my reflection"
    elif evolution_type == 'abrupt_reversal':
        return "Everything I understood seems wrong now"
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