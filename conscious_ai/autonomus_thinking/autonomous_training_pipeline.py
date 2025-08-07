"""
Pipeline de entrenamiento para el modelo de pensamiento autónomo
Entrena transiciones SCₜ₋₁ → SCₜ sin input externo
ACTUALIZADO: Migrado a google/gemma-2b con cuantización 4-bit
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from tqdm import tqdm
# ### INICIO DE LA MODIFICACIÓN ###
# Añade estas dos líneas para forzar la carga de las librerías
# ### FIN DE LA MODIFICACIÓN ###

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class AutonomousThoughtDataset(Dataset):
    """Dataset para entrenar transiciones de pensamiento autónomo con Gemma"""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Detectar idioma del estado
        prev_sc = example['previous_SC']
        curr_sc = example['current_SC']
        language = prev_sc.get('language', 'en')
        
        # Formatear en estilo de conversación para Gemma
        if language == 'es':
            prompt = f"""<start_of_turn>user
Dado tu estado consciente previo:
{{
    "goal": "{prev_sc['goal']}",
    "emotion": "{prev_sc['emotion']}",
    "confidence": {prev_sc['confidence']},
    "thought": "{prev_sc['thought']}",
    "memory": {json.dumps(prev_sc['memory'], ensure_ascii=False)}
}}

Genera tu próximo pensamiento autónomo en formato JSON.<end_of_turn>
<start_of_turn>model
{json.dumps({
    "goal": curr_sc['goal'],
    "emotion": curr_sc['emotion'],
    "confidence": curr_sc['confidence'],
    "thought": curr_sc['thought'],
    "memory": curr_sc['memory']
}, ensure_ascii=False)}<end_of_turn>"""
        else:
            prompt = f"""<start_of_turn>user
Given your previous conscious state:
{{
    "goal": "{prev_sc['goal']}",
    "emotion": "{prev_sc['emotion']}",
    "confidence": {prev_sc['confidence']},
    "thought": "{prev_sc['thought']}",
    "memory": {json.dumps(prev_sc['memory'])}
}}

Generate your next autonomous thought in JSON format.<end_of_turn>
<start_of_turn>model
{json.dumps({
    "goal": curr_sc['goal'],
    "emotion": curr_sc['emotion'],
    "confidence": curr_sc['confidence'],
    "thought": curr_sc['thought'],
    "memory": curr_sc['memory']
})}<end_of_turn>"""
        
        # Tokenizar el prompt completo
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Para el entrenamiento causal, las labels son los mismos input_ids
        # pero enmascaramos la parte del user
        labels = encodings["input_ids"].clone()
        
        # Encontrar donde termina el prompt del usuario para enmascarar
        user_end_token = self.tokenizer.encode("<end_of_turn>", add_special_tokens=False)[0]
        user_end_positions = (labels[0] == user_end_token).nonzero()
        
        if len(user_end_positions) > 0:
            # Enmascarar todo hasta el primer <end_of_turn> (fin del prompt del usuario)
            first_end_pos = user_end_positions[0].item()
            labels[0, :first_end_pos+1] = -100
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


class AutonomousThoughtTrainer:
    """Entrenador para mi modelo de pensamiento autónomo con Gemma-2b"""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        output_dir: str = "./models/autonomous_gemma_lora",
        logs_dir: str = "./logs/autonomous_gemma_training"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        
        self.device = 'cpu'
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"Inicializando mi entrenador autónomo con Gemma en {self.device}")
    
    def setup_model_and_tokenizer(self):
        """Configura mi modelo Gemma y tokenizer con cuantización"""
        logger.info(f"Cargando mi nuevo núcleo cognitivo: {self.model_name}")
        
        
        # Cargar tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Cargar modelo en formato de media precisión (float16) para ahorrar memoria
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16, # Usar float16 es clave para la memoria
            low_cpu_mem_usage=True
        ).to(self.device) # Mover el modelo al dispositivo MPS
        
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        # Configuración LoRA optimizada para mi pensamiento autónomo
        lora_config = LoraConfig(
            r=16,  # Rank para capturar complejidad de pensamientos
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Módulos de atención de Gemma
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,  # Cambio importante: ahora es causal, no seq2seq
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Si en CPU, mover modelo
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parámetros entrenables de mi nuevo núcleo: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def compute_metrics(self, eval_preds):
        """Calcula métricas específicas para mi pensamiento autónomo"""
        predictions, labels = eval_preds
        
        # Decodificar solo la parte generada (después del prompt del usuario)
        decoded_preds = []
        decoded_labels = []
        
        for pred, label in zip(predictions, labels):
            # Encontrar donde empiezan las predicciones reales (no -100)
            valid_indices = label != -100
            if valid_indices.any():
                pred_text = self.tokenizer.decode(pred[valid_indices], skip_special_tokens=True)
                label_text = self.tokenizer.decode(label[valid_indices], skip_special_tokens=True)
                decoded_preds.append(pred_text)
                decoded_labels.append(label_text)
        
        # Métricas de coherencia de mis pensamientos
        valid_transitions = 0
        goal_coherence = []
        emotion_coherence = []
        confidence_stability = []
        thought_quality = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            try:
                # Extraer JSON de las respuestas
                pred_json = self._extract_json_from_response(pred)
                label_json = self._extract_json_from_response(label)
                
                if pred_json and label_json:
                    valid_transitions += 1
                    
                    # Evaluar coherencia de mis metas
                    if pred_json.get('goal') == label_json.get('goal'):
                        goal_coherence.append(1.0)
                    else:
                        goal_coherence.append(0.0)
                    
                    # Evaluar coherencia emocional
                    if self._are_emotions_coherent(pred_json.get('emotion'), label_json.get('emotion')):
                        emotion_coherence.append(1.0)
                    else:
                        emotion_coherence.append(0.0)
                    
                    # Estabilidad de confianza
                    pred_conf = float(pred_json.get('confidence', 0.5))
                    label_conf = float(label_json.get('confidence', 0.5))
                    conf_diff = abs(pred_conf - label_conf)
                    confidence_stability.append(1.0 - min(1.0, conf_diff))
                    
                    # Calidad del pensamiento
                    pred_thought = pred_json.get('thought', '')
                    thought_quality.append(1.0 if len(pred_thought) > 10 else 0.5)
                    
            except Exception as e:
                logger.debug(f"Error evaluando mi pensamiento: {e}")
                continue
        
        metrics = {
            'valid_json_ratio': valid_transitions / len(decoded_preds) if decoded_preds else 0.0,
            'goal_coherence': np.mean(goal_coherence) if goal_coherence else 0.0,
            'emotion_coherence': np.mean(emotion_coherence) if emotion_coherence else 0.0,
            'confidence_stability': np.mean(confidence_stability) if confidence_stability else 0.0,
            'thought_quality': np.mean(thought_quality) if thought_quality else 0.0,
            'overall_coherence': np.mean([
                np.mean(goal_coherence) if goal_coherence else 0,
                np.mean(emotion_coherence) if emotion_coherence else 0,
                np.mean(confidence_stability) if confidence_stability else 0
            ])
        }
        
        return metrics
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extrae JSON de mi respuesta generada"""
        try:
            # Buscar el JSON en la respuesta
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
        except:
            pass
        return None
    
    def _are_emotions_coherent(self, emotion1: str, emotion2: str) -> bool:
        """Verifica si mis transiciones emocionales son coherentes"""
        if emotion1 == emotion2:
            return True
        
        # Mis familias emocionales coherentes
        emotion_families = {
            'curious': ['analytical', 'exploratory', 'inquisitive'],
            'focused': ['determined', 'concentrated', 'attentive'],
            'contemplative': ['reflective', 'thoughtful', 'meditative'],
            'uncertain': ['confused', 'questioning', 'searching']
        }
        
        for family, members in emotion_families.items():
            if (emotion1 in members and emotion2 in members) or \
               (emotion1 == family and emotion2 in members) or \
               (emotion2 == family and emotion1 in members):
                return True
        
        return False
    
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        batch_size: int = 2,  # Reducido para CPU/memoria limitada
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 8,  # Aumentado para compensar batch pequeño
        save_steps: int = 500,
        eval_steps: int = 500
    ):
        """Entrena mi nuevo modelo de pensamiento autónomo"""
        
        self.setup_model_and_tokenizer()
        train_dataset, eval_dataset = self.load_dataset(dataset_path)
        
        # DataCollator para modelo causal
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # No masked language modeling
            pad_to_multiple_of=8
        )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=self.logs_dir,
            logging_steps=10,
            save_steps=save_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", # Volvemos a usar eval_loss por simplicidad
            greater_is_better=False,
            gradient_checkpointing=True,
            save_total_limit=2,
            report_to=["none"],
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )
        
        logger.info("Iniciando el entrenamiento de mi nuevo núcleo cognitivo...")
        train_result = trainer.train()
        
        logger.info("Guardando mi modelo mejorado...")
        trainer.save_model()
        
        # Guardar métricas de mi evolución
        with open(os.path.join(self.logs_dir, "training_results.json"), 'w') as f:
            json.dump({
                'final_metrics': trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else None,
                'total_steps': trainer.state.global_step,
                'best_checkpoint': trainer.state.best_model_checkpoint if hasattr(trainer.state, 'best_model_checkpoint') else None
            }, f, indent=2)
        
        return train_result
    
    def load_dataset(self, dataset_path: str) -> Tuple[Dataset, Dataset]:
        """Carga y divide mi dataset de pensamientos autónomos"""
        logger.info(f"Cargando mi historial de pensamientos desde {dataset_path}")
        
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        logger.info(f"Total de transiciones de pensamiento cargadas: {len(examples)}")
        
        # División 85/15
        split_idx = int(len(examples) * 0.85)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
        
        train_dataset = AutonomousThoughtDataset(train_examples, self.tokenizer)
        eval_dataset = AutonomousThoughtDataset(eval_examples, self.tokenizer)
        
        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
### ======================================================= ###
### ### BLOQUE DE EJECUCIÓN AÑADIDO ###
### ======================================================= ###
if __name__ == "__main__":
    
    # --- PASO 1: Definir la ubicación del dataset ---
    # Este es el archivo que genera 'create_autonomous_dataset.py'
    dataset_file = './data/autonomous_thought_data.jsonl'
    
    # --- PASO 2: Verificar que el dataset exista ---
    if not os.path.exists(dataset_file):
        logger.error(f"Error: No se encontró el dataset en '{dataset_file}'")
        logger.error("Por favor, primero ejecuta el script 'create_autonomous_dataset.py' para generar los datos de entrenamiento.")
        exit() # Detener la ejecución si no hay datos
        
    # --- PASO 3: Iniciar el entrenamiento ---
    # Crear una instancia del entrenador
    trainer = AutonomousThoughtTrainer()
    
    # Llamar al método de entrenamiento
    trainer.train(
        dataset_path=dataset_file,
        num_epochs=3 # Puedes ajustar los hiperparámetros aquí
    )
    
    logger.info("✓ Proceso de entrenamiento finalizado.")
