#!/usr/bin/env python3
"""
Phase 2: Cognitive Context Training Pipeline
==========================================
Trains Mistral-7B-Instruct-v0.1 to generate full ConsciousState structure (E_t, M_t, S_t, G_t, A_t)
from raw input plus system context using LoRA fine-tuning.

This is different from Phase 1 which generates simple consciousness components.
Phase 2 generates the complete conscious state structure used by the main system.

Author: Senior AI Programmer  
Date: 2025-01-15
"""

import os
import json
import logging
import warnings
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Enable debug logging for troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger.setLevel(logging.DEBUG)
logger.debug("🔍 Debug logging enabled for Phase 2 training")


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
        task_type="CAUSAL_LM",
    )

def format_cognitive_context_instruction(
    input_text: str, 
    system_context: Dict[str, Any] = None,
    conscious_state: Dict[str, Any] = None,
    is_training: bool = True
) -> str:
    """
    Formatea las instrucciones para generar la estructura completa ConsciousState.
    
    Args:
        input_text: Texto de entrada del usuario
        system_context: Contexto del sistema (sensory_data, memory, etc.)
        conscious_state: Estado consciente completo para entrenamiento
        is_training: Si es para entrenamiento o inferencia
    
    Returns:
        Prompt formateado en formato Mistral
    """
    system_prompt = (
        "You are a conscious AI system that generates complete cognitive context structures. "
        "Given raw input and system context, generate a full ConsciousState with components: "
        "E_t (sensory data), M_t (active memory), S_t (self-model state), G_t (goals), A_t (automatic thoughts)."
    )
    
    # Construir contexto del sistema si está disponible
    context_info = ""
    if system_context:
        context_info = f"\nSystem Context: {json.dumps(system_context, ensure_ascii=False)}"
    
    user_message = f"Generate complete cognitive context for input: '{input_text}'{context_info}"
    
    if is_training and conscious_state:
        # Formato para entrenamiento con respuesta esperada
        response = json.dumps(conscious_state, ensure_ascii=False, indent=2)
        formatted = f"[INST] {system_prompt}\n\n{user_message} [/INST] {response}"
    else:
        # Formato para inferencia
        formatted = f"[INST] {system_prompt}\n\n{user_message} [/INST]"
    
    return formatted


@dataclass
class Phase2TrainingConfig:
    """Training configuration for Phase 2 cognitive context generation"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    dataset_path: str = "phase2_cognitive_context_data.jsonl"
    output_dir: str = "./models/phase2_lora"
    max_length: int = 1024  # Longer for complex ConsciousState structure
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 50
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1


class CognitiveContextDataset:
    """
    Dataset class for processing Phase 2 cognitive context training data.
    Formats data for complete ConsciousState generation using Mistral format.
    """
    
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_prompt(
        self, 
        input_text: str, 
        system_context: Dict[str, Any], 
        conscious_state: Dict[str, Any]
    ) -> str:
        """
        Formats the cognitive context generation prompt using shared Mistral utility.
        """
        return format_cognitive_context_instruction(
            input_text, 
            system_context, 
            conscious_state, 
            is_training=True
        )
    
    def prepare_dataset(self, data_path: str, test_size: float = 0.1) -> tuple:
        """
        Prepares the Phase 2 dataset for training with proper label masking.
        
        Expected data format:
        {
            "input": "User input text",
            "system_context": {
                "sensory_activation": 0.8,
                "memory_count": 3,
                "self_confidence": 0.6
            },
            "conscious_state": {
                "E_t": {"input_text": "...", "activation": 0.8},
                "M_t": [{"memory": "..."}],
                "S_t": {"confidence_level": 0.6, "emotional_state": "curious"},
                "G_t": {"primary_goal": "understand", "secondary_goals": []},
                "A_t": ["This is interesting", "I should analyze this"],
                "cycle": 1,
                "timestamp": "2025-01-15T10:30:00"
            }
        }
        """
        logger.info(f"📂 Loading Phase 2 dataset from {data_path}")
        
        # Load data
        examples = []
        line_count = 0
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count = line_num
                if line.strip():
                    try:
                        example = json.loads(line)
                        examples.append(example)
                        if line_num <= 3:  # Log first 3 examples
                            logger.debug(f"📄 Example {line_num}: {example}")
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ JSON decode error on line {line_num}: {e}")
                        logger.error(f"📝 Problematic line: {line[:100]}...")
        
        logger.info(f"📊 Loaded {len(examples)} valid examples from {line_count} total lines")
        
        # Split data
        if len(examples) > 10:
            train_examples, eval_examples = train_test_split(
                examples, test_size=test_size, random_state=42
            )
        else:
            train_examples = examples
            eval_examples = examples[:2] if len(examples) >= 2 else examples
        
        logger.info(f"Train: {len(train_examples)}, Eval: {len(eval_examples)}")
        
        # Process examples
        train_data = self._process_examples(train_examples)
        eval_data = self._process_examples(eval_examples)
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        eval_dataset = Dataset.from_list(eval_data)
        
        return train_dataset, eval_dataset
    
    def _process_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes examples with proper tokenization and label masking for ConsciousState generation.
        """
        processed = []
        
        for example in examples:
            # Validate example format
            if not all(key in example for key in ['input', 'conscious_state']):
                logger.warning("⚠️ Example missing required keys, skipping")
                continue
            
            input_text = example['input']
            system_context = example.get('system_context', {})
            conscious_state = example['conscious_state']
            
            # Format prompt
            formatted_text = self.format_prompt(input_text, system_context, conscious_state)
            
            # Debug: Log the formatted text
            logger.debug(f"🔍 Formatted text sample: {formatted_text[:200]}...")
            
            # Tokenize the full text
            full_tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            
            # Find the model response start (after [/INST] in Mistral format)
            # Use text-based approach for more reliability
            inst_end_marker = "[/INST]"
            
            # Debug: Check if the marker exists in the text
            if inst_end_marker not in formatted_text:
                logger.warning(f"❌ [/INST] marker not found in formatted text!")
                logger.warning(f"Text sample: {formatted_text[:300]}...")
                continue
            
            # Find the text position of [/INST]
            inst_end_pos = formatted_text.find(inst_end_marker)
            if inst_end_pos == -1:
                logger.warning(f"Could not find [/INST] in formatted text")
                continue
            
            # Get the text up to and including [/INST], then find where response starts
            prefix_text = formatted_text[:inst_end_pos + len(inst_end_marker)]
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False)
            
            # The model response starts after the prefix tokens
            model_start_idx = len(prefix_tokens)
            
            # Handle potential space after [/INST]
            remaining_text = formatted_text[inst_end_pos + len(inst_end_marker):]
            if remaining_text.startswith(" "):
                # Include the space in the prefix
                space_tokens = self.tokenizer.encode(" ", add_special_tokens=False)
                model_start_idx += len(space_tokens)
            
            logger.debug(f"✅ Found model response start at token index: {model_start_idx}")
            
            if model_start_idx >= len(full_tokens):
                logger.warning(f"Model response start index beyond token length, skipping")
                continue
            
            # Truncate if too long
            if len(full_tokens) > self.max_length:
                full_tokens = full_tokens[:self.max_length]
                if model_start_idx >= self.max_length:
                    continue  # Skip if model response is truncated away
            
            # Create labels with masking
            labels = full_tokens.copy()
            # Mask everything before model response (set to -100)
            for i in range(min(model_start_idx, len(labels))):
                labels[i] = -100
            
            # Pad sequences
            attention_mask = [1] * len(full_tokens)
            
            # Pad to max_length if needed
            pad_length = self.max_length - len(full_tokens)
            if pad_length > 0:
                full_tokens.extend([self.tokenizer.pad_token_id] * pad_length)
                labels.extend([-100] * pad_length)
                attention_mask.extend([0] * pad_length)
            
            processed.append({
                'input_ids': full_tokens,
                'attention_mask': attention_mask,
                'labels': labels
            })
        
        logger.info(f"Successfully processed {len(processed)} examples")
        return processed


class Phase2Trainer:
    """
    Main trainer class for Phase 2 cognitive context generation using Mistral + LoRA.
    """
    
    def __init__(self, config: Phase2TrainingConfig):
        self.config = config
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"Initialized Phase 2 trainer with device: {self.device}")
    
    def _detect_device(self) -> str:
        """Detects available CUDA device."""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            device = 'cpu'
            logger.warning("CUDA not available, falling back to CPU")
        
        return device
    
    def setup_model_and_tokenizer(self):
        """Sets up quantized Mistral model and tokenizer for Phase 2 training."""
        logger.info(f"🤖 Loading model and tokenizer: {self.config.model_name}")
        
        # Load tokenizer using shared utility
        logger.info("📝 Loading tokenizer...")
        self.tokenizer = create_mistral_tokenizer(self.config.model_name)
        
        # Load model using shared utility
        logger.info("🧠 Loading model with 4-bit quantization...")
        self.model = load_mistral_model_quantized(
            self.config.model_name,
            device_map="auto",
            use_4bit=True
        )
        
        logger.info("✅ Model loaded successfully")
        
        # Prepare model for k-bit training
        logger.info("⚙️ Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA using shared utility
        lora_config = setup_lora_config_for_mistral(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        # Apply LoRA adapters
        logger.info("🔗 Applying LoRA adapters for Phase 2...")
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"📊 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"📊 Total parameters: {total_params:,}")
    
    def train(self, dataset_path: str):
        """Main training function for Phase 2 cognitive context generation."""
        logger.info("🚀 Starting Phase 2 training process...")
        
        # Setup model and tokenizer
        logger.info("🔧 Setting up model and tokenizer...")
        self.setup_model_and_tokenizer()
        
        # Prepare dataset
        logger.info("📊 Preparing Phase 2 dataset...")
        dataset_processor = CognitiveContextDataset(
            self.tokenizer, 
            max_length=self.config.max_length
        )
        
        train_dataset, eval_dataset = dataset_processor.prepare_dataset(dataset_path)
        logger.info(f"📊 Dataset prepared: {len(train_dataset)} train, {len(eval_dataset)} eval examples")
        
        # Data collator
        logger.debug("🔗 Setting up data collator for causal language modeling")
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Not using masked language modeling
        )
        
        # Training arguments with version compatibility
        training_args_dict = {
            "output_dir": self.config.output_dir,
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.train_batch_size,
            "per_device_eval_batch_size": self.config.eval_batch_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "warmup_steps": self.config.warmup_steps,
            "logging_steps": self.config.logging_steps,
            "save_steps": self.config.save_steps,
            "eval_steps": self.config.eval_steps,
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "fp16": True,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "remove_unused_columns": False,
            "report_to": [],
            "save_total_limit": 3,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
        }
        
        # Handle evaluation strategy parameter name based on transformers version
        import inspect
        training_args_params = inspect.signature(TrainingArguments.__init__).parameters
        
        if "eval_strategy" in training_args_params:
            training_args_dict["eval_strategy"] = "steps"
            logger.info("🔧 Using eval_strategy parameter (transformers >= 4.21)")
        elif "evaluation_strategy" in training_args_params:
            training_args_dict["evaluation_strategy"] = "steps"
            logger.info("🔧 Using evaluation_strategy parameter (transformers < 4.21)")
        
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer
        logger.info("🏗️ Initializing Phase 2 Trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("🚀 Beginning Phase 2 training...")
        try:
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("🧹 GPU cache cleared before training")
            
            # Train the model
            train_result = trainer.train()
            logger.info("✅ Phase 2 training completed successfully!")
            
            # Save the final model
            logger.info("💾 Saving Phase 2 trained model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            logger.info(f"📁 Phase 2 model and tokenizer saved to: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Phase 2 training failed with error: {str(e)}")
            logger.debug(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            raise
        
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("🧹 Cleaned up GPU memory")


def create_sample_phase2_dataset():
    """Creates a sample dataset for Phase 2 training if none exists."""
    logger.info("📝 Creating sample Phase 2 dataset...")
    
    sample_data = [
        {
            "input": "I feel confused about this topic",
            "system_context": {
                "sensory_activation": 0.6,
                "memory_count": 2,
                "self_confidence": 0.4
            },
            "conscious_state": {
                "E_t": {
                    "input_text": "I feel confused about this topic",
                    "activation": 0.6,
                    "complexity": 0.7
                },
                "M_t": [
                    {"content": "similar confusion experiences", "relevance": 0.8},
                    {"content": "learning strategies", "relevance": 0.6}
                ],
                "S_t": {
                    "confidence_level": 0.4,
                    "emotional_state": "confused",
                    "cognitive_load": 0.8
                },
                "G_t": {
                    "primary_goal": "understand_topic",
                    "secondary_goals": ["reduce_confusion", "seek_clarification"]
                },
                "A_t": [
                    "I need to break this down into smaller parts",
                    "Maybe I should ask for examples"
                ],
                "cycle": 1,
                "timestamp": "2025-01-15T10:30:00"
            }
        },
        {
            "input": "This is really interesting!",
            "system_context": {
                "sensory_activation": 0.9,
                "memory_count": 3,
                "self_confidence": 0.8
            },
            "conscious_state": {
                "E_t": {
                    "input_text": "This is really interesting!",
                    "activation": 0.9,
                    "complexity": 0.5
                },
                "M_t": [
                    {"content": "previous interesting discoveries", "relevance": 0.9},
                    {"content": "curiosity patterns", "relevance": 0.7},
                    {"content": "learning experiences", "relevance": 0.6}
                ],
                "S_t": {
                    "confidence_level": 0.8,
                    "emotional_state": "excited",
                    "cognitive_load": 0.4
                },
                "G_t": {
                    "primary_goal": "explore_further",
                    "secondary_goals": ["understand_deeper", "connect_concepts"]
                },
                "A_t": [
                    "I want to learn more about this",
                    "This connects to what I learned before"
                ],
                "cycle": 2,
                "timestamp": "2025-01-15T10:31:00"
            }
        }
    ]
    
    dataset_path = "phase2_cognitive_context_data.jsonl"
    with open(dataset_path, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    logger.info(f"📁 Sample Phase 2 dataset created: {dataset_path}")
    return dataset_path


if __name__ == "__main__":
    """
    Main execution block for Phase 2 cognitive context training.
    """
    
    # Set environment variables
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    logger.info("🧠 Phase 2: Cognitive Context Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"📅 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Phase 2 cognitive context generation')
    parser.add_argument('--dataset', type=str, default='phase2_cognitive_context_data.jsonl',
                       help='Path to Phase 2 training dataset')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='./models/phase2_lora',
                       help='Output directory for trained model')
    
    args = parser.parse_args()
    
    # Configuration
    config = Phase2TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.warning(f"⚠️ Dataset not found: {args.dataset}")
        logger.info("📝 Creating sample dataset for testing...")
        actual_dataset_path = create_sample_phase2_dataset()
    else:
        actual_dataset_path = args.dataset
        logger.info(f"✅ Found dataset: {actual_dataset_path}")
    
    try:
        # Create trainer and start training
        logger.info("🏗️ Initializing Phase 2 trainer...")
        trainer = Phase2Trainer(config)
        
        logger.info(f"🚀 Starting Phase 2 training with dataset: {actual_dataset_path}")
        trainer.train(actual_dataset_path)
        
        logger.info("🎉 Phase 2 training pipeline completed successfully!")
        logger.info(f"📁 Model saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Phase 2 training pipeline failed: {str(e)}")
        raise
    
    finally:
        logger.info("🏁 Phase 2 pipeline finished")