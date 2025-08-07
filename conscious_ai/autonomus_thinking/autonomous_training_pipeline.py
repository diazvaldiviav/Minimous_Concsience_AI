"""
Pipeline de entrenamiento para el modelo de pensamiento autónomo
Entrena transiciones SCₜ₋₁ → SCₜ sin input externo
ACTUALIZADO: Migrado a google/gemma-2b con cuantización 4-bit para Colab/GPU.
"""

import os
import json
import torch
from typing import Dict, List, Any, Tuple
import logging
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AutonomousThoughtDataset(Dataset):
    """Dataset para entrenar transiciones de pensamiento autónomo con Gemma."""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        prev_sc = example['previous_SC']
        curr_sc = example['current_SC']
        language = prev_sc.get('language', 'en')
        
        # Formatear en estilo de conversación para Gemma
        prompt = f"""<start_of_turn>user
Given the previous conscious state:
{json.dumps(prev_sc, ensure_ascii=False, indent=2)}

Generate the next autonomous thought in JSON format.<end_of_turn>
<start_of_turn>model
{json.dumps(curr_sc, ensure_ascii=False, indent=2)}<end_of_turn>"""
        
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = encodings["input_ids"].clone()
        
        # Encontrar donde termina el prompt del usuario para enmascarar
        # Buscamos la secuencia de tokens que indica el inicio de la respuesta del modelo
        model_prompt_text = "<start_of_turn>model"
        model_prompt_tokens = self.tokenizer(model_prompt_text, add_special_tokens=False)["input_ids"]
        
        # Convertir a tensor para la búsqueda
        labels_list = labels[0].tolist()
        search_sequence = model_prompt_tokens
        
        try:
            # Encontrar el índice donde empieza la secuencia del modelo
            separator_idx = -1
            for i in range(len(labels_list) - len(search_sequence) + 1):
                if labels_list[i:i+len(search_sequence)] == search_sequence:
                    separator_idx = i
                    break
            
            if separator_idx != -1:
                labels[0, :separator_idx] = -100
        except Exception:
            # Fallback por si algo sale mal en la búsqueda
            pass
        
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


class AutonomousThoughtTrainer:
    """Entrenador para el modelo de pensamiento autónomo con Gemma-2b en Colab."""
    
    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        output_dir: str = "./models/autonomous_gemma_lora",
        logs_dir: str = "./logs/autonomous_gemma_training"
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        logger.info(f"Inicializando entrenador autónomo con Gemma en el dispositivo: {self.device.upper()}")
    
    def setup_model_and_tokenizer(self):
        """Configura el modelo Gemma y tokenizer con cuantización para GPU."""
        logger.info(f"Cargando núcleo cognitivo: {self.model_name}")
        
        ### CONFIGURACIÓN PARA COLAB (GPU + 4-BIT) ###
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parámetros entrenables: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 16,
    ):
        """Entrena el modelo de pensamiento autónomo en Colab."""
        self.setup_model_and_tokenizer()
        train_dataset, eval_dataset = self.load_dataset(dataset_path)
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        
        ### ARGUMENTOS DE ENTRENAMIENTO PARA GPU ###
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_dir=self.logs_dir,
            logging_steps=20,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True, # Activar media precisión para GPU
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
        )
        
        logger.info("Iniciando entrenamiento en Colab (GPU)...")
        trainer.train()
        
        logger.info("Guardando modelo final...")
        trainer.save_model()
    
    def load_dataset(self, dataset_path: str) -> Tuple[Dataset, Dataset]:
        """Carga y divide el dataset de pensamientos autónomos."""
        logger.info(f"Cargando historial de pensamientos desde {dataset_path}")
        examples = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        logger.info(f"Total de transiciones cargadas: {len(examples)}")
        
        split_idx = int(len(examples) * 0.90)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]
        
        train_dataset = AutonomousThoughtDataset(train_examples, self.tokenizer)
        eval_dataset = AutonomousThoughtDataset(eval_examples, self.tokenizer)
        
        logger.info(f"División de datos - Entrenamiento: {len(train_dataset)}, Evaluación: {len(eval_dataset)}")
        return train_dataset, eval_dataset

if __name__ == "__main__":
    dataset_file = './data/autonomous_thought_data.jsonl'
    
    if not os.path.exists(dataset_file):
        logger.error(f"Error: No se encontró el dataset en '{dataset_file}'")
        logger.error("Por favor, primero ejecuta 'create_autonomous_dataset.py' para generar los datos.")
        exit()
        
    trainer = AutonomousThoughtTrainer()
    trainer.train(dataset_path=dataset_file)
    logger.info("✓ Proceso de entrenamiento finalizado.")