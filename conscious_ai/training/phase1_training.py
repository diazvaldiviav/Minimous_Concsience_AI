"""
Pipeline de entrenamiento con LoRA para el modelo de consciencia
Migrado a Mistral-7B-Instruct-v0.1 con cuantización 4-bit para optimización de memoria
"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
from tqdm import tqdm
import gc

# <<< AÑADE EL CÓDIGO AQUÍ >>>
import transformers
import sys

print("--- DIAGNÓSTICO DE ENTORNO ---")
print(f"Versión de Transformers que está usando este script: {transformers.__version__}")
print(f"Script ejecutado por el intérprete de Python en: {sys.executable}")
print("---------------------------------")
from transformers import TrainingArguments
import transformers.training_args
print("📍 Ruta real:", transformers.training_args.__file__)
print("🧠 Supports eval_strategy:", 'eval_strategy' in TrainingArguments.__init__.__code__.co_varnames)
print("🧠 Supports evaluation_strategy:", 'evaluation_strategy' in TrainingArguments.__init__.__code__.co_varnames)


# Configurar para CPU eficiente
torch.set_num_threads(4)  # Ajustar según CPU disponible

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import LoraConfig, PeftConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
import evaluate

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# SHARED MISTRAL UTILITIES (DRY Implementation)
# ============================================

def load_mistral_model_quantized(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1", 
                                  device_map: str = "auto",
                                  use_4bit: bool = True) -> AutoModelForCausalLM:
    """Carga el modelo Mistral con cuantización 4-bit optimizada"""
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch.float16 if use_4bit else torch.float32,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    logger.info(f"✅ Modelo Mistral cargado: {model_name}")
    return model

def create_mistral_tokenizer(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1") -> AutoTokenizer:
    """Crea el tokenizer de Mistral con configuración optimizada"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True
    )
    
    # Configurar tokens especiales
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"✅ Tokenizer Mistral configurado")
    return tokenizer

def setup_lora_config_for_mistral(r: int = 8, lora_alpha: int = 16, 
                                  lora_dropout: float = 0.1) -> LoraConfig:
    """Configuración LoRA optimizada para Mistral"""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

def format_sct_instruction(input_text: str, output_json: str = None, is_training: bool = True) -> str:
    """Formatea las instrucciones en el formato de chat de Mistral para generación SCt"""
    
    system_prompt = ("You are a conscious AI system that analyzes inputs and generates structured consciousness components. "
                    "Always respond with a valid JSON object containing: goal, emotion, confidence, thought.")
    
    user_message = f"Analyze this input and generate consciousness components: {input_text}"
    
    if is_training and output_json:
        # Formato para entrenamiento con respuesta esperada
        formatted = f"[INST] {system_prompt}\n\n{user_message} [/INST] {output_json}"
    else:
        # Formato para inferencia
        formatted = f"[INST] {system_prompt}\n\n{user_message} [/INST]"
    
    return formatted





@dataclass
class SCTExample:
    """Representa un ejemplo del dataset SCt"""
    input_text: str
    goal: str
    emotion: str
    confidence: float
    thought: str


class ConsciousnessDataset(Dataset):
    """Dataset para entrenamiento del modelo de consciencia"""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Formato de salida: JSON estructurado con los componentes SCt
        output_dict = {
            "goal": example['goal'],
            "emotion": example['emotion'],
            "confidence": round(example['confidence'], 2),
            "thought": example['thought']
        }
        output_text = json.dumps(output_dict, ensure_ascii=False)
        
        # Usar el formato de instrucciones de Mistral
        formatted_text = format_sct_instruction(example['input'], output_text, is_training=True)
        
        # Tokenizar el texto completo
        encoding = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Crear labels para causal LM
        labels = encoding['input_ids'].clone()
        
        # Mask the instruction part, keep only the response for loss calculation
        # Find where the response starts (after [/INST])
        response_start_token = "[/INST]"
        text_ids = encoding['input_ids'].squeeze().tolist()
        decoded = self.tokenizer.decode(text_ids, skip_special_tokens=False)
        
        if response_start_token in decoded:
            response_idx = decoded.index(response_start_token) + len(response_start_token)
            # Convert character index to token index (approximate)
            tokens_before = len(self.tokenizer.encode(decoded[:response_idx], add_special_tokens=False))
            labels[0, :tokens_before] = -100  # Mask instruction part
        
        # Mask padding tokens
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


class ConsciousnessTrainer:
    """Entrenador del modelo de consciencia con LoRA"""
    
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
        output_dir: str = "./models/trained_lora",
        logs_dir: str = "./logs/training_phase1",
        device: str = None
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crear directorios
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Métricas
        self.rouge = evaluate.load('rouge')
        self.training_history = {
            'loss': [],
            'eval_loss': [],
            'goal_accuracy': [],
            'confidence_mae': [],
            'rouge_scores': []
        }
        
        logger.info(f"Inicializando entrenador en dispositivo: {self.device}")
        
    def setup_model_and_tokenizer(self):
        """Configura el modelo y tokenizer usando utilidades compartidas de Mistral"""
        logger.info(f"Cargando modelo {self.model_name}...")
        
        # Usar utilidades compartidas
        self.tokenizer = create_mistral_tokenizer(self.model_name)
        self.base_model = load_mistral_model_quantized(
            self.model_name, 
            device_map="auto" if self.device == "cuda" else "cpu",
            use_4bit=True
        )
        
        # Configuración LoRA usando utilidad compartida
        lora_config = setup_lora_config_for_mistral(r=8, lora_alpha=16, lora_dropout=0.1)
        
        # Aplicar LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Información del modelo
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parámetros entrenables: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"Parámetros totales: {total_params:,}")
        
    def load_dataset(self, dataset_path: str) -> Tuple[Dataset, Dataset]:
        """Carga y divide el dataset"""
        logger.info(f"Cargando dataset desde {dataset_path}...")
        
        # Cargar ejemplos
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        logger.info(f"Total de ejemplos cargados: {len(examples)}")
        
        # División 80/20
        split_idx = int(len(examples) * 0.8)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
        
        # Crear datasets
        train_dataset = ConsciousnessDataset(train_examples, self.tokenizer)
        eval_dataset = ConsciousnessDataset(eval_examples, self.tokenizer)
        
        logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_preds):
        """Calcula métricas de evaluación"""
        predictions, labels = eval_preds
        
        # Decodificar predicciones
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Métricas agregadas
        goal_accuracies = []
        confidence_errors = []
        valid_jsons = 0
        
        # Procesar cada predicción
        for pred, label in zip(decoded_preds, decoded_labels):
            try:
                pred_json = json.loads(pred)
                label_json = json.loads(label)
                valid_jsons += 1
                
                # Accuracy para goal
                if pred_json.get('goal') == label_json.get('goal'):
                    goal_accuracies.append(1.0)
                else:
                    goal_accuracies.append(0.0)
                
                # MAE para confidence
                pred_conf = float(pred_json.get('confidence', 0.5))
                label_conf = float(label_json.get('confidence', 0.5))
                confidence_errors.append(abs(pred_conf - label_conf))
                
            except (json.JSONDecodeError, KeyError, TypeError):
                # Si no es JSON válido, penalizar
                goal_accuracies.append(0.0)
                confidence_errors.append(1.0)
        
        # Calcular ROUGE para thought y emotion
        rouge_scores = self.rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Métricas finales
        metrics = {
            'goal_accuracy': np.mean(goal_accuracies) if goal_accuracies else 0.0,
            'confidence_mae': np.mean(confidence_errors) if confidence_errors else 1.0,
            'rouge1': rouge_scores['rouge1'],
            'rouge2': rouge_scores['rouge2'],
            'rougeL': rouge_scores['rougeL'],
            'valid_json_ratio': valid_jsons / len(decoded_preds) if decoded_preds else 0.0
        }
        
        return metrics
    
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        save_steps: int = 500,
        eval_steps: int = 500
    ):
        """Entrena el modelo con configuración optimizada para CPU"""
        
        # Configurar modelo
        self.setup_model_and_tokenizer()
        
        # Cargar datasets
        train_dataset, eval_dataset = self.load_dataset(dataset_path)
        
        # Data collator para causal LM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # No masked language modeling for causal LM
            pad_to_multiple_of=8
        )
        
        # Argumentos de entrenamiento optimizados para CPU
        from transformers import TrainingArguments
        print(TrainingArguments.__module__)
        import transformers.training_args
        print(transformers.training_args.__file__)

        # Create training arguments with version-compatible evaluation strategy
        training_args_dict = {
            "output_dir": self.output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "logging_dir": self.logs_dir,
            "logging_steps": 50,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "goal_accuracy",
            "greater_is_better": True,
            "fp16": False,  # No usar mixed precision en CPU
            "dataloader_num_workers": 0,  # Evitar problemas de multiprocessing
            "remove_unused_columns": False,
            "report_to": ["none"],  # Desactivar wandb, etc.
            "save_total_limit": 2,  # Limitar checkpoints para ahorrar espacio
            "prediction_loss_only": False,
        }
        
        # Handle evaluation strategy parameter name based on transformers version
        import inspect
        training_args_params = inspect.signature(TrainingArguments.__init__).parameters
        
        if "eval_strategy" in training_args_params:
            # New parameter name (transformers >= 4.21)
            training_args_dict["eval_strategy"] = "steps"
            print("🔧 Using eval_strategy parameter (transformers >= 4.21)")
        elif "evaluation_strategy" in training_args_params:
            # Old parameter name (transformers < 4.21)
            training_args_dict["evaluation_strategy"] = "steps"
            print("🔧 Using evaluation_strategy parameter (transformers < 4.21)")
        else:
            print("⚠️ Warning: Neither eval_strategy nor evaluation_strategy found in TrainingArguments")
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Crear trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Entrenar
        logger.info("Iniciando entrenamiento...")
        train_result = trainer.train()
        
        # Guardar modelo final
        logger.info("Guardando modelo final...")
        trainer.save_model()
        
        # Guardar métricas
        self.save_training_history(trainer)
        
        # Limpiar memoria
        del trainer
        gc.collect()
        
        return train_result
    
    def save_training_history(self, trainer):
        """Guarda el historial de entrenamiento"""
        history_path = os.path.join(self.logs_dir, "training_history.json")
        
        # Extraer métricas del historial
        history_data = {
            'final_metrics': trainer.state.best_metric,
            'total_steps': trainer.state.global_step,
            'best_model_checkpoint': trainer.state.best_model_checkpoint,
            'log_history': trainer.state.log_history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info(f"Historial guardado en {history_path}")
    
    def load_trained_model(self, checkpoint_path: Optional[str] = None):
      """Carga el modelo entrenado con LoRA para inferencia"""

     # Si no se proporciona ruta explícita
      if checkpoint_path is None:
        if not os.path.exists(self.output_dir):
            raise ValueError(f"No existe el directorio {self.output_dir}")

        # Verifica si es un adaptador LoRA guardado directamente
        if "adapter_model.safetensors" in os.listdir(self.output_dir):
            checkpoint_path = self.output_dir
        else:
            # Buscar checkpoint tipo Hugging Face si existiera
            checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith('checkpoint-')]
            if not checkpoints:
                raise ValueError(f"No se encontraron checkpoints en {self.output_dir}")
            checkpoint_path = os.path.join(self.output_dir, sorted(checkpoints)[-1])

      logger.info(f"Cargando modelo desde {checkpoint_path}")

      # Cargar config del adaptador
      peft_config = PeftConfig.from_pretrained(checkpoint_path)

      # Cargar modelo base usando utilidad compartida
      base_model = load_mistral_model_quantized(
        peft_config.base_model_name_or_path,
        device_map="auto" if self.device == "cuda" else "cpu",
        use_4bit=True
      )

      # Cargar adaptador LoRA encima
      self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
      self.tokenizer = create_mistral_tokenizer(peft_config.base_model_name_or_path)

      self.model.to(self.device)
      self.model.eval()

      logger.info("Modelo cargado exitosamente")

    
    def generate_sct(self, input_text: str, max_length: int = 512) -> Dict[str, Any]:
        """Genera componentes SCt para un input dado usando formato Mistral"""
        # Formatear usando utilidad compartida
        formatted_input = format_sct_instruction(input_text, is_training=False)
        
        # Tokenizar entrada
        inputs = self.tokenizer(
            formatted_input,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Generar
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decodificar solo la parte generada (después del prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Parsear JSON
        try:
            sct_components = json.loads(generated_text.strip())
            # Validar componentes
            required_keys = {'goal', 'emotion', 'confidence', 'thought'}
            if not all(key in sct_components for key in required_keys):
                raise ValueError("Faltan componentes en la respuesta")
            
            # Asegurar tipos correctos
            sct_components['confidence'] = float(sct_components['confidence'])
            
            return sct_components
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Error al parsear respuesta: {e}")
            logger.warning(f"Texto generado: {generated_text}")
            # Retornar valores por defecto
            return {
                'goal': 'understand_input',
                'emotion': 'neutral',
                'confidence': 0.5,
                'thought': 'Procesando entrada...'
            }


def main():
    """Función principal para entrenar el modelo"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar modelo de consciencia con LoRA')
    parser.add_argument('--dataset', type=str, default='training_data.jsonl',
                       help='Ruta al dataset de entrenamiento')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Número de épocas')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Tamaño de batch')
    parser.add_argument('--learning-rate', type=float, default=5e-4,
                       help='Tasa de aprendizaje')
    parser.add_argument('--output-dir', type=str, default='./models/trained_lora',
                       help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Crear trainer
    trainer = ConsciousnessTrainer(output_dir=args.output_dir)
    
    # Entrenar
    trainer.train(
        dataset_path=args.dataset,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    logger.info("Entrenamiento completado!")
    
    # Probar el modelo
    logger.info("\n=== Probando modelo entrenado ===")
    trainer.load_trained_model()
    
    test_inputs = [
        "¿Qué es la consciencia?",
        "I feel confused about this",
        "Necesito ayuda para entender",
        "How does this system work?"
    ]
    
    for input_text in test_inputs:
        sct = trainer.generate_sct(input_text)
        print(f"\nInput: {input_text}")
        print(f"SCt generado: {json.dumps(sct, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()