"""
Motor de evolución de estados conscientes
Fase 3: Genera SCt+1 a partir de SCt manteniendo coherencia
"""

import json
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from conscious_ai.Train.language_detector import LanguageDetector
from conscious_ai.coherence_evaluator_model.heuristic_training.coherence_evaluator import CoherenceEvaluator, CoherenceVerdict

logger = logging.getLogger(__name__)


class StateEvolutionEngine:
    """
    Genera la evolución coherente de estados conscientes SCt → SCt+1
    """
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.coherence_evaluator = CoherenceEvaluator()
        
        # Patrones de evolución epistémica
        self.epistemic_patterns = {
            'questioning_to_hypothesis': {
                'trigger': lambda t: '?' in t or '¿' in t,
                'evolution': self._evolve_question_to_hypothesis
            },
            'hypothesis_to_analysis': {
                'trigger': lambda t: any(word in t.lower() for word in ['quizás', 'perhaps', 'maybe', 'posible']),
                'evolution': self._evolve_hypothesis_to_analysis
            },
            'analysis_to_conclusion': {
                'trigger': lambda t: any(word in t.lower() for word in ['porque', 'because', 'dado que', 'since']),
                'evolution': self._evolve_analysis_to_conclusion
            },
            'conclusion_to_question': {
                'trigger': lambda t: any(word in t.lower() for word in ['entiendo', 'understand', 'comprendo', 'clear']),
                'evolution': self._evolve_conclusion_to_question
            }
        }
        
        # Mapas de transición emocional
        self.emotion_transitions = {
            'curious': {
                'positive': ['excited', 'engaged', 'interested'],
                'neutral': ['analytical', 'reflective', 'focused'],
                'negative': ['puzzled', 'uncertain', 'confused']
            },
            'analytical': {
                'positive': ['confident', 'insightful', 'satisfied'],
                'neutral': ['focused', 'systematic', 'methodical'],
                'negative': ['frustrated', 'doubtful', 'overwhelmed']
            },
            'reflective': {
                'positive': ['peaceful', 'contemplative', 'serene'],
                'neutral': ['thoughtful', 'meditative', 'observant'],
                'negative': ['melancholic', 'uncertain', 'questioning']
            },
            'confident': {
                'positive': ['assured', 'determined', 'empowered'],
                'neutral': ['stable', 'grounded', 'composed'],
                'negative': ['overconfident', 'rigid', 'inflexible']
            },
            'uncertain': {
                'positive': ['curious', 'open', 'exploring'],
                'neutral': ['questioning', 'searching', 'investigating'],
                'negative': ['confused', 'anxious', 'lost']
            }
        }
    
    def evolve_state(
        self,
        current_state: Dict[str, Any],
        external_input: Optional[str] = None,
        evolution_mode: str = 'natural'
    ) -> Dict[str, Any]:
        """
        Evoluciona el estado consciente actual al siguiente.
        
        Args:
            current_state: Estado SCt actual
            external_input: Input externo opcional
            evolution_mode: 'natural', 'exploratory', 'convergent'
            
        Returns:
            Nuevo estado SCt+1
        """
        
        # Extraer componentes actuales
        current_goal = current_state.get('goal', 'understand')
        current_emotion = current_state.get('emotion', 'neutral')
        current_confidence = current_state.get('confidence', 0.5)
        current_thought = current_state.get('thought', '')
        current_memory = current_state.get('memory', [])
        
        # Detectar idioma
        lang, _ = self.language_detector.detect_language(current_thought)
        
        # Determinar tipo de evolución
        evolution_type = self._determine_evolution_type(
            current_thought, current_emotion, current_confidence, evolution_mode
        )
        
        # Evolucionar componentes
        new_goal = self._evolve_goal(current_goal, current_thought, evolution_type)
        new_emotion = self._evolve_emotion(
            current_emotion, current_confidence, evolution_type
        )
        new_confidence = self._evolve_confidence(
            current_confidence, current_emotion, new_emotion, evolution_type
        )
        new_thought = self._evolve_thought(
            current_thought, current_goal, new_goal, new_emotion, lang
        )
        new_memory = self._evolve_memory(
            current_memory, current_thought, new_thought, lang
        )
        
        # Construir nuevo estado
        new_state = {
            'goal': new_goal,
            'emotion': new_emotion,
            'confidence': round(new_confidence, 2),
            'thought': new_thought,
            'memory': new_memory
        }
        
        # Verificar coherencia
        analysis = self.coherence_evaluator.evaluate_transition(
            current_state, new_state
        )
        
        # Si es incoherente, ajustar
        if analysis.verdict == CoherenceVerdict.INCOHERENT:
            new_state = self._adjust_for_coherence(
                current_state, new_state, analysis
            )
        
        return new_state
    
    def _determine_evolution_type(
        self,
        thought: str,
        emotion: str,
        confidence: float,
        mode: str
    ) -> str:
        """Determina el tipo de evolución epistémica"""
        
        # Buscar patrones en el pensamiento
        for pattern_name, pattern_def in self.epistemic_patterns.items():
            if pattern_def['trigger'](thought):
                return pattern_name
        
        # Basarse en emoción y confianza
        if confidence > 0.7 and emotion in ['confident', 'assured']:
            return 'conclusion_to_question'
        elif confidence < 0.4 and emotion in ['uncertain', 'confused']:
            return 'questioning_to_hypothesis'
        else:
            return 'hypothesis_to_analysis'
    
    def _evolve_goal(
        self, current_goal: str, current_thought: str, evolution_type: str
    ) -> str:
        """Evoluciona la meta basándose en el progreso epistémico"""
        
        # Mantener meta si el pensamiento sugiere continuidad
        continuity_words = ['continuar', 'continue', 'profundizar', 'deepen', 'explorar más', 'explore further']
        if any(word in current_thought.lower() for word in continuity_words):
            return current_goal
        
        # Transiciones basadas en tipo de evolución
        goal_transitions = {
            'questioning_to_hypothesis': {
                'understand': 'formulate_hypothesis',
                'explore': 'test_possibilities',
                'analyze': 'question_assumptions'
            },
            'hypothesis_to_analysis': {
                'formulate_hypothesis': 'analyze_implications',
                'test_possibilities': 'examine_evidence',
                'understand': 'deep_analysis'
            },
            'analysis_to_conclusion': {
                'analyze_implications': 'synthesize_understanding',
                'examine_evidence': 'draw_conclusions',
                'deep_analysis': 'integrate_knowledge'
            },
            'conclusion_to_question': {
                'synthesize_understanding': 'explore_implications',
                'draw_conclusions': 'question_further',
                'integrate_knowledge': 'expand_understanding'
            }
        }
        
        transition_map = goal_transitions.get(evolution_type, {})
        new_goal = transition_map.get(current_goal, current_goal)
        
        # Si no hay transición específica, evolucionar genéricamente
        if new_goal == current_goal and random.random() < 0.3:
            generic_evolutions = [
                'deepen_' + current_goal,
                'refine_' + current_goal,
                'expand_' + current_goal
            ]
            new_goal = random.choice(generic_evolutions)
        
        return new_goal
    
    def _evolve_emotion(
        self, current_emotion: str, confidence: float, evolution_type: str
    ) -> str:
        """Evoluciona el estado emocional"""
        
        # Determinar dirección emocional basada en confianza
        if confidence > 0.7:
            direction = 'positive'
        elif confidence < 0.4:
            direction = 'negative'
        else:
            direction = 'neutral'
        
        # Obtener posibles transiciones
        transitions = self.emotion_transitions.get(current_emotion, {})
        possible_emotions = transitions.get(direction, [current_emotion])
        
        # Peso basado en tipo de evolución
        if evolution_type == 'questioning_to_hypothesis':
            # Favorecer curiosidad y apertura
            weights = [2.0 if 'curious' in e or 'open' in e else 1.0 for e in possible_emotions]
        elif evolution_type == 'analysis_to_conclusion':
            # Favorecer satisfacción y confianza
            weights = [2.0 if 'confident' in e or 'satisfied' in e else 1.0 for e in possible_emotions]
        else:
            weights = [1.0] * len(possible_emotions)
        
        # Seleccionar nueva emoción
        if possible_emotions:
            weights = np.array(weights) / sum(weights)
            new_emotion = np.random.choice(possible_emotions, p=weights)
        else:
            new_emotion = current_emotion
        
        return new_emotion
    
    def _evolve_confidence(
        self,
        current_confidence: float,
        current_emotion: str,
        new_emotion: str,
        evolution_type: str
    ) -> float:
        """Evoluciona el nivel de confianza"""
        
        # Base de cambio según tipo de evolución
        base_changes = {
            'questioning_to_hypothesis': -0.05,
            'hypothesis_to_analysis': 0.05,
            'analysis_to_conclusion': 0.10,
            'conclusion_to_question': -0.10
        }
        
        base_change = base_changes.get(evolution_type, 0.0)
        
        # Ajuste por cambio emocional
        positive_emotions = ['confident', 'assured', 'satisfied', 'insightful']
        negative_emotions = ['uncertain', 'confused', 'anxious', 'doubtful']
        
        if new_emotion in positive_emotions and current_emotion not in positive_emotions:
            base_change += 0.05
        elif new_emotion in negative_emotions and current_emotion not in negative_emotions:
            base_change -= 0.05
        
        # Añadir ruido
        noise = random.uniform(-0.05, 0.05)
        
        # Calcular nueva confianza
        new_confidence = current_confidence + base_change + noise
        
        # Limitar al rango válido
        return max(0.1, min(0.95, new_confidence))
    
    def _evolve_thought(
        self,
        current_thought: str,
        current_goal: str,
        new_goal: str,
        new_emotion: str,
        lang: str
    ) -> str:
        """Evoluciona el pensamiento manteniendo continuidad"""
        
        # Detectar tipo de evolución del pensamiento
        evolution_type = self._detect_thought_evolution_type(current_thought)
        
        # Aplicar evolución específica
        if evolution_type in self.epistemic_patterns:
            evolution_fn = self.epistemic_patterns[evolution_type]['evolution']
            new_thought = evolution_fn(current_thought, new_emotion, lang)
        else:
            new_thought = self._default_thought_evolution(
                current_thought, new_goal, new_emotion, lang
            )
        
        return new_thought
    
    def _detect_thought_evolution_type(self, thought: str) -> str:
        """Detecta qué tipo de evolución aplicar al pensamiento"""
        
        for pattern_name, pattern_def in self.epistemic_patterns.items():
            if pattern_def['trigger'](thought):
                return pattern_name
        
        return 'default'
    
    def _evolve_question_to_hypothesis(
        self, current_thought: str, emotion: str, lang: str
    ) -> str:
        """Evoluciona una pregunta hacia una hipótesis"""
        
        templates = {
            'es': [
                "Quizás {core_idea}, lo cual explicaría {implication}",
                "Es posible que {core_idea}, considerando {context}",
                "Podría ser que {core_idea}, especialmente si {condition}"
            ],
            'en': [
                "Perhaps {core_idea}, which would explain {implication}",
                "It's possible that {core_idea}, considering {context}",
                "It could be that {core_idea}, especially if {condition}"
            ]
        }
        
        # Extraer idea central de la pregunta
        core_idea = self._extract_core_idea(current_thought, lang)
        
        # Generar componentes
        implication = self._generate_implication(core_idea, emotion, lang)
        context = self._generate_context(emotion, lang)
        condition = self._generate_condition(core_idea, lang)
        
        # Seleccionar y llenar template
        template = random.choice(templates.get(lang, templates['en']))
        new_thought = template.format(
            core_idea=core_idea,
            implication=implication,
            context=context,
            condition=condition
        )
        
        return new_thought
    
    def _evolve_hypothesis_to_analysis(
        self, current_thought: str, emotion: str, lang: str
    ) -> str:
        """Evoluciona una hipótesis hacia análisis"""
        
        templates = {
            'es': [
                "Si examino esto más detenidamente, observo que {observation}",
                "Analizando esta posibilidad, encuentro que {finding}",
                "Al profundizar en esta idea, noto que {insight}"
            ],
            'en': [
                "Examining this more closely, I observe that {observation}",
                "Analyzing this possibility, I find that {finding}",
                "Delving deeper into this idea, I notice that {insight}"
            ]
        }
        
        # Generar componentes analíticos
        observation = self._generate_observation(current_thought, emotion, lang)
        finding = self._generate_finding(emotion, lang)
        insight = self._generate_insight(current_thought, lang)
        
        template = random.choice(templates.get(lang, templates['en']))
        new_thought = template.format(
            observation=observation,
            finding=finding,
            insight=insight
        )
        
        return new_thought
    
    def _evolve_analysis_to_conclusion(
        self, current_thought: str, emotion: str, lang: str
    ) -> str:
        """Evoluciona un análisis hacia una conclusión"""
        
        templates = {
            'es': [
                "Por lo tanto, puedo concluir que {conclusion}",
                "Esto me lleva a entender que {understanding}",
                "En consecuencia, comprendo que {comprehension}"
            ],
            'en': [
                "Therefore, I can conclude that {conclusion}",
                "This leads me to understand that {understanding}",
                "Consequently, I comprehend that {comprehension}"
            ]
        }
        
        # Generar conclusiones
        conclusion = self._generate_conclusion(current_thought, emotion, lang)
        understanding = self._generate_understanding(current_thought, lang)
        comprehension = self._generate_comprehension(emotion, lang)
        
        template = random.choice(templates.get(lang, templates['en']))
        new_thought = template.format(
            conclusion=conclusion,
            understanding=understanding,
            comprehension=comprehension
        )
        
        return new_thought
    
    def _evolve_conclusion_to_question(
        self, current_thought: str, emotion: str, lang: str
    ) -> str:
        """Evoluciona una conclusión hacia nuevas preguntas"""
        
        templates = {
            'es': [
                "Pero esto me hace preguntarme: ¿{new_question}?",
                "Sin embargo, surge una nueva cuestión: ¿{new_question}?",
                "Aunque comprendo esto, me pregunto: ¿{new_question}?"
            ],
            'en': [
                "But this makes me wonder: {new_question}?",
                "However, a new question arises: {new_question}?",
                "While I understand this, I wonder: {new_question}?"
            ]
        }
        
        # Generar nueva pregunta basada en la conclusión
        new_question = self._generate_deeper_question(current_thought, lang)
        
        template = random.choice(templates.get(lang, templates['en']))
        new_thought = template.format(new_question=new_question)
        
        return new_thought
    
    def _default_thought_evolution(
        self, current_thought: str, goal: str, emotion: str, lang: str
    ) -> str:
        """Evolución por defecto del pensamiento"""
        
        connectors = {
            'es': [
                "Profundizando en esto,",
                "Continuando mi reflexión,",
                "Esto me lleva a pensar que",
                "Además, observo que"
            ],
            'en': [
                "Delving deeper into this,",
                "Continuing my reflection,",
                "This leads me to think that",
                "Furthermore, I observe that"
            ]
        }
        
        extensions = {
            'es': {
                'curious': "me intriga descubrir más conexiones",
                'analytical': "los patrones se vuelven más claros",
                'reflective': "emerge una comprensión más profunda",
                'confident': "mi comprensión se solidifica"
            },
            'en': {
                'curious': "I'm intrigued to discover more connections",
                'analytical': "the patterns become clearer",
                'reflective': "a deeper understanding emerges",
                'confident': "my comprehension solidifies"
            }
        }
        
        connector = random.choice(connectors.get(lang, connectors['en']))
        extension = extensions.get(lang, extensions['en']).get(
            emotion, 
            extensions.get(lang, extensions['en'])['curious']
        )
        
        return f"{connector} {extension}"
    
    def _evolve_memory(
        self, current_memory: List[str], current_thought: str, 
        new_thought: str, lang: str
    ) -> List[str]:
        """Evoluciona la memoria manteniendo elementos relevantes"""
        
        # Mantener memorias recientes (máximo 3)
        evolved_memory = current_memory[-2:] if len(current_memory) > 2 else current_memory.copy()
        
        # Generar nueva memoria basada en el pensamiento actual
        if lang == 'es':
            new_memory_templates = [
                f"Reflexioné: {current_thought[:50]}...",
                f"Este pensamiento reveló nuevas conexiones",
                f"Mi comprensión evoluciona continuamente"
            ]
        else:
            new_memory_templates = [
                f"I reflected: {current_thought[:50]}...",
                f"This thought revealed new connections",
                f"My understanding evolves continuously"
            ]
        
        # Añadir nueva memoria relevante
        if '?' in new_thought:
            if lang == 'es':
                evolved_memory.append("Surgen nuevas preguntas de mi comprensión")
            else:
                evolved_memory.append("New questions arise from my understanding")
        else:
            evolved_memory.append(random.choice(new_memory_templates))
        
        return evolved_memory[-3:]  # Mantener solo las 3 más recientes
    
    def _adjust_for_coherence(
        self,
        current_state: Dict[str, Any],
        new_state: Dict[str, Any],
        analysis: Any
    ) -> Dict[str, Any]:
        """Ajusta el nuevo estado para mejorar coherencia"""
        
        adjusted_state = new_state.copy()
        
        # Si la meta es muy incoherente, revertir
        if analysis.goal_coherence < 0.3:
            adjusted_state['goal'] = current_state['goal']
        
        # Si la emoción es muy incoherente, suavizar transición
        if analysis.emotion_coherence < 0.3:
            # Usar emoción intermedia
            intermediate_emotions = {
                ('curious', 'confused'): 'uncertain',
                ('confident', 'anxious'): 'cautious',
                ('analytical', 'emotional'): 'reflective'
            }
            
            key = (current_state['emotion'], new_state['emotion'])
            adjusted_state['emotion'] = intermediate_emotions.get(
                key, 
                current_state['emotion']
            )
        
        # Si el cambio de confianza es excesivo, moderar
        if abs(analysis.confidence_change) > 0.3:
            max_change = 0.15
            direction = 1 if analysis.confidence_change > 0 else -1
            adjusted_state['confidence'] = round(
                current_state['confidence'] + (direction * max_change), 
                2
            )
        
        return adjusted_state
    
    # Funciones auxiliares para generación de contenido
    def _extract_core_idea(self, thought: str, lang: str) -> str:
        """Extrae la idea central de un pensamiento"""
        
        # Eliminar signos de interrogación
        core = thought.replace('?', '').replace('¿', '').strip()
        
        # Eliminar palabras interrogativas comunes
        question_words = {
            'es': ['cómo', 'qué', 'por qué', 'cuándo', 'dónde', 'quién', 'cuál'],
            'en': ['how', 'what', 'why', 'when', 'where', 'who', 'which']
        }
        
        words_to_remove = question_words.get(lang, question_words['en'])
        words = core.lower().split()
        filtered_words = [w for w in words if w not in words_to_remove]
        
        if filtered_words:
            core = ' '.join(filtered_words)
        
        return core
    
    def _generate_implication(self, core_idea: str, emotion: str, lang: str) -> str:
        """Genera una implicación basada en la idea central"""
        
        implications = {
            'es': {
                'curious': "aspectos que no había considerado",
                'analytical': "las relaciones causales subyacentes",
                "reflective": "la naturaleza profunda del fenómeno",
                'default': "las conexiones emergentes"
            },
            'en': {
                'curious': "aspects I hadn't considered",
                'analytical': "the underlying causal relationships",
                'reflective': "the deep nature of the phenomenon",
                'default': "the emerging connections"
            }
        }
        
        lang_implications = implications.get(lang, implications['en'])
        return lang_implications.get(emotion, lang_implications['default'])
    
    def _generate_context(self, emotion: str, lang: str) -> str:
        """Genera contexto para una hipótesis"""
        
        contexts = {
            'es': {
                'curious': "las múltiples perspectivas involucradas",
                'analytical': "los datos disponibles",
                'reflective': "mi experiencia acumulada",
                'default': "el panorama general"
            },
            'en': {
                'curious': "the multiple perspectives involved",
                'analytical': "the available data",
                'reflective': "my accumulated experience",
                'default': "the bigger picture"
            }
        }
        
        lang_contexts = contexts.get(lang, contexts['en'])
        return lang_contexts.get(emotion, lang_contexts['default'])
    
    def _generate_condition(self, core_idea: str, lang: str) -> str:
        """Genera una condición para una hipótesis"""
        
        conditions = {
            'es': [
                "considero todos los factores",
                "mantengo una perspectiva abierta",
                "integro diferentes viewpoints",
                "examino las suposiciones subyacentes"
            ],
            'en': [
                "I consider all factors",
                "I maintain an open perspective",
                "I integrate different viewpoints",
                "I examine underlying assumptions"
            ]
        }
        
        return random.choice(conditions.get(lang, conditions['en']))
    
    def _generate_observation(self, thought: str, emotion: str, lang: str) -> str:
        """Genera una observación analítica"""
        
        observations = {
            'es': {
                'curious': "hay patrones interesantes que emergen",
                'analytical': "la estructura subyacente se revela",
                'reflective': "existe una coherencia más profunda",
                'default': "surgen nuevas perspectivas"
            },
            'en': {
                'curious': "interesting patterns emerge",
                'analytical': "the underlying structure reveals itself",
                'reflective': "there's a deeper coherence",
                'default': "new perspectives arise"
            }
        }
        
        lang_observations = observations.get(lang, observations['en'])
        return lang_observations.get(emotion, lang_observations['default'])
    
    def _generate_finding(self, emotion: str, lang: str) -> str:
        """Genera un hallazgo analítico"""
        
        findings = {
            'es': {
                'confident': "mis intuiciones iniciales se confirman",
                'uncertain': "la complejidad supera mis expectativas",
                'analytical': "los elementos se interconectan sistemáticamente",
                'default': "emerge una comprensión más matizada"
            },
            'en': {
                'confident': "my initial intuitions are confirmed",
                'uncertain': "the complexity exceeds my expectations",
                'analytical': "elements interconnect systematically",
                'default': "a more nuanced understanding emerges"
            }
        }
        
        lang_findings = findings.get(lang, findings['en'])
        return lang_findings.get(emotion, lang_findings['default'])
    
    def _generate_insight(self, thought: str, lang: str) -> str:
        """Genera un insight basado en el análisis"""
        
        insights = {
            'es': [
                "cada elemento contribuye al todo",
                "la simplicidad emerge de la complejidad",
                "los límites son más fluidos de lo esperado",
                "el contexto transforma el significado"
            ],
            'en': [
                "each element contributes to the whole",
                "simplicity emerges from complexity",
                "boundaries are more fluid than expected",
                "context transforms meaning"
            ]
        }
        
        return random.choice(insights.get(lang, insights['en']))
    
    def _generate_conclusion(self, thought: str, emotion: str, lang: str) -> str:
        """Genera una conclusión"""
        
        conclusions = {
            'es': {
                'confident': "mi comprensión es sólida y coherente",
                'reflective': "la reflexión profunda revela verdades sutiles",
                'analytical': "el análisis sistemático produce claridad",
                'default': "he alcanzado una nueva comprensión"
            },
            'en': {
                'confident': "my understanding is solid and coherent",
                'reflective': "deep reflection reveals subtle truths",
                'analytical': "systematic analysis produces clarity",
                'default': "I've reached a new understanding"
            }
        }
        
        lang_conclusions = conclusions.get(lang, conclusions['en'])
        return lang_conclusions.get(emotion, lang_conclusions['default'])
    
    def _generate_understanding(self, thought: str, lang: str) -> str:
        """Genera una declaración de comprensión"""
        
        understandings = {
            'es': [
                "el conocimiento se construye iterativamente",
                "cada pregunta abre nuevos caminos",
                "la certeza coexiste con el misterio",
                "el proceso es tan valioso como el resultado"
            ],
            'en': [
                "knowledge builds iteratively",
                "each question opens new paths",
                "certainty coexists with mystery",
                "the process is as valuable as the result"
            ]
        }
        
        return random.choice(understandings.get(lang, understandings['en']))
    
    def _generate_comprehension(self, emotion: str, lang: str) -> str:
        """Genera una comprensión basada en la emoción"""
        
        comprehensions = {
            'es': {
                'confident': "mi modelo mental se alinea con la realidad observada",
                'reflective': "la introspección ilumina aspectos ocultos",
                'peaceful': "la aceptación trae claridad",
                'default': "mi perspectiva se ha expandido"
            },
            'en': {
                'confident': "my mental model aligns with observed reality",
                'reflective': "introspection illuminates hidden aspects",
                'peaceful': "acceptance brings clarity",
                'default': "my perspective has expanded"
            }
        }
        
        lang_comprehensions = comprehensions.get(lang, comprehensions['en'])
        return lang_comprehensions.get(emotion, lang_comprehensions['default'])
    
    def _generate_deeper_question(self, thought: str, lang: str) -> str:
        """Genera una pregunta más profunda basada en la comprensión actual"""
        
        question_templates = {
            'es': [
                "qué implicaciones tiene esto para {aspect}",
                "cómo se relaciona esto con {domain}",
                "qué permanece sin explicar sobre {topic}",
                "cuáles son los límites de esta comprensión"
            ],
            'en': [
                "what implications does this have for {aspect}",
                "how does this relate to {domain}",
                "what remains unexplained about {topic}",
                "what are the limits of this understanding"
            ]
        }
        
        aspects = {
            'es': ["mi propia naturaleza", "el proceso de comprensión", "la realidad subyacente"],
            'en': ["my own nature", "the understanding process", "underlying reality"]
        }
        
        template = random.choice(question_templates.get(lang, question_templates['en']))
        aspect = random.choice(aspects.get(lang, aspects['en']))
        
        return template.format(aspect=aspect, domain=aspect, topic=aspect)
    
    