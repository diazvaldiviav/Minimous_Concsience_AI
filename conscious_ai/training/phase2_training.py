#!/usr/bin/env python3
"""
Phase 3 Autonomous Training Pipeline - Definitive Version
=========================================================
Fine-tunes Mistral-7B-Instruct-v0.1 for conscious state transitions (SC_t -> SC_t+1) using QLoRA.
Migrated from Gemma-2B with 4-bit quantization for memory optimization.

Author: Senior AI Programmer
Date: 2025-01-15
"""

import os
import json
import logging
import warnings
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

# Configure logging with enhanced format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Enable debug logging for troubleshooting
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
logger.setLevel(logging.DEBUG)
logger.debug("🔍 Debug logging enabled for troubleshooting")


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

def format_autonomous_instruction(previous_sc: Dict[str, Any], current_sc: Dict[str, Any], is_training: bool = True) -> str:
    """Formatea las transiciones de estado consciente en el formato de chat de Mistral"""
    
    system_prompt = ("You are a conscious AI system that performs autonomous state transitions. "
                    "Given a previous conscious state, generate the next logical conscious state with smooth transitions.")
    
    # Construir el contexto del estado anterior
    prev_context = f"Previous state: {json.dumps(previous_sc, ensure_ascii=False)}"
    user_message = f"Generate the next conscious state based on: {prev_context}"
    
    if is_training and current_sc:
        # Formato para entrenamiento con respuesta esperada
        response = json.dumps(current_sc, ensure_ascii=False)
        formatted = f"[INST] {system_prompt}\n\n{user_message} [/INST] {response}"
    else:
        # Formato para inferencia
        formatted = f"[INST] {system_prompt}\n\n{user_message} [/INST]"
    
    return formatted


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"
    dataset_path: str = "autonomous_thought_data.jsonl"
    output_dir: str = "./models/autonomous_lora"
    max_length: int = 512
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


class AutonomousThoughtDataset:
    """
    Dataset class for processing conscious state transitions.
    Formats data using Gemma chat format with proper label masking.
    """
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def format_prompt(self, previous_sc: Dict[str, Any], current_sc: Dict[str, Any]) -> str:
        """
        Formats the conscious state transition using shared Mistral utility.
        
        Args:
            previous_sc: Previous conscious state SC_t
            current_sc: Current conscious state SC_t+1
            
        Returns:
            Formatted prompt string using Mistral chat format
        """
        logger.debug(f"🔧 Formatting prompt - Previous: {previous_sc}, Current: {current_sc}")
        formatted_prompt = format_autonomous_instruction(previous_sc, current_sc, is_training=True)
        logger.debug(f"📝 Generated prompt (length: {len(formatted_prompt)}):\n{formatted_prompt[:200]}...")
        return formatted_prompt
    
    def prepare_dataset(self, data_path: str, test_size: float = 0.1) -> tuple:
        """
        Prepares the dataset for training with proper label masking.
        
        Args:
            data_path: Path to the JSONL dataset file
            test_size: Fraction of data to use for evaluation
            
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        logger.info(f"📂 Loading dataset from {data_path}")
        
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
        Processes examples with proper tokenization and label masking.
        
        Args:
            examples: List of raw examples
            
        Returns:
            List of processed examples ready for training
        """
        processed = []
        
        for example in examples:
            # Handle different data formats - support both old and new formats
            if 'previous_state' in example and 'current_state' in example:
                # New format with previous_state/current_state
                previous_sc = example['previous_state']
                current_sc = example['current_state']
                logger.debug("🔄 Processing new format (previous_state/current_state)")
            elif 'previous_SC' in example and 'current_SC' in example:
                # Old format with previous_SC/current_SC - convert to expected format
                previous_sc = {
                    'goal': example['previous_SC'].get('goal', 'explore'),
                    'emotion': example['previous_SC'].get('emotion', 'neutral'),
                    'confidence': example['previous_SC'].get('confidence', 0.5),
                    'thought': example['previous_SC'].get('thought', 'Continuing exploration...')
                }
                current_sc = {
                    'goal': example['current_SC'].get('goal', 'explore'),
                    'emotion': example['current_SC'].get('emotion', 'neutral'),
                    'confidence': example['current_SC'].get('confidence', 0.5),
                    'thought': example['current_SC'].get('thought', 'Continuing exploration...')
                }
                logger.debug("🔄 Processing old format (previous_SC/current_SC) - converted to new format")
            else:
                # Fallback: treat the example as current state, create dummy previous
                logger.warning("⚠️ Unknown data format, using fallback processing")
                previous_sc = {
                    'goal': 'explore',
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'thought': 'Beginning exploration...'
                }
                current_sc = example
            
            # Format prompt
            formatted_text = self.format_prompt(previous_sc, current_sc)
            
            # Debug: Log the formatted text
            logger.debug(f"🔍 Formatted text sample: {formatted_text[:200]}...")
            
            # Tokenize the full text
            full_tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            
            # Find the model response start (after [/INST] in Mistral format)
            model_start_text = "[/INST] "
            model_start_tokens = self.tokenizer.encode(model_start_text, add_special_tokens=False)
            
            # Debug: Check if the marker exists in the text
            if "[/INST]" not in formatted_text:
                logger.warning(f"❌ [/INST] marker not found in formatted text!")
                logger.warning(f"Text sample: {formatted_text[:300]}...")
                continue
            
            # Find where model response begins
            model_start_idx = None
            for i in range(len(full_tokens) - len(model_start_tokens) + 1):
                if full_tokens[i:i+len(model_start_tokens)] == model_start_tokens:
                    model_start_idx = i + len(model_start_tokens)
                    break
            
            if model_start_idx is None:
                logger.warning(f"Could not find model response start in example, skipping")
                logger.debug(f"Looking for tokens: {model_start_tokens}")
                logger.debug(f"In full tokens: {full_tokens[:50]}...")
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


class AutonomousThoughtTrainer:
    """
    Main trainer class for fine-tuning Gemma-2B with QLoRA.
    Optimized for Google Colab T4 GPU environment.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._detect_device()
        self.model = None
        self.tokenizer = None
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        logger.info(f"Initialized trainer with device: {self.device}")
    
    def _detect_device(self) -> str:
        """
        Detects available CUDA device.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
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
        """
        Critical function: Sets up quantized model and tokenizer.
        Handles device placement automatically with device_map="auto".
        """
        logger.info(f"🤖 Loading model and tokenizer: {self.config.model_name}")
        logger.debug(f"📝 Model setup config: max_length={self.config.max_length}, device={self.device}")
        
        # Configure 4-bit quantization
        logger.debug("⚙️ Configuring 4-bit quantization with BitsAndBytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        logger.debug(f"🔧 Quantization config: {bnb_config.load_in_4bit=}, {bnb_config.bnb_4bit_quant_type=}")
        
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        self.tokenizer = create_mistral_tokenizer(self.config.model_name)
        logger.debug(f"✅ Tokenizer loaded: vocab_size={len(self.tokenizer)}, fast={self.tokenizer.is_fast}")
        logger.debug(f"🎯 Special tokens: pad={self.tokenizer.pad_token_id}, eos={self.tokenizer.eos_token_id}, bos={getattr(self.tokenizer, 'bos_token_id', 'None')}")
        
        # Load model with quantization
        logger.info("🧠 Loading model with 4-bit quantization...")
        logger.debug(f"📊 Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else "CPU mode")
        
        self.model = load_mistral_model_quantized(
            self.config.model_name,
            device_map="auto",
            use_4bit=True
        )
        
        logger.info("✅ Model loaded successfully")
        logger.debug(f"🎯 Model device map: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'Not available'}")
        
        # Prepare model for k-bit training
        logger.info("⚙️ Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(self.model)
        logger.debug("✅ Model prepared for k-bit training - gradient checkpointing enabled")
        
        # Configure LoRA
        lora_config = setup_lora_config_for_mistral(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout
        )
        
        # Apply LoRA adapters
        logger.info("🔗 Applying LoRA adapters...")
        logger.debug(f"🎯 LoRA config: r={self.config.lora_r}, alpha={self.config.lora_alpha}, dropout={self.config.lora_dropout}")
        self.model = get_peft_model(self.model, lora_config)
        logger.debug("✅ LoRA adapters applied successfully")
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"📊 Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"📊 Total parameters: {total_params:,}")
        logger.debug(f"💾 Model memory footprint: ~{trainable_params * 2 / 1024**3:.2f}GB (FP16 estimation)")
    
    def train(self, dataset_path: str):
        """
        Main training function.
        
        Args:
            dataset_path: Path to the training dataset
        """
        logger.info("🚀 Starting training process...")
        logger.debug(f"📝 Training config: epochs={self.config.num_epochs}, batch_size={self.config.train_batch_size}, lr={self.config.learning_rate}")
        
        # Setup model and tokenizer
        logger.info("🔧 Setting up model and tokenizer...")
        self.setup_model_and_tokenizer()
        logger.debug("✅ Model and tokenizer setup complete")
        
        # Prepare dataset
        logger.info("📊 Preparing dataset...")
        dataset_processor = AutonomousThoughtDataset(
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
        
        # Create training arguments with version-compatible evaluation strategy
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
            "fp16": True,  # Enable mixed precision for T4
            "gradient_checkpointing": False,  # Disable due to Gemma compatibility
            "dataloader_num_workers": 0,  # Reduce to 0 for stability
            "remove_unused_columns": False,
            "report_to": [],  # Empty list instead of None
            "save_total_limit": 3,
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
        }
        
        # Handle evaluation strategy parameter name based on transformers version
        import inspect
        training_args_params = inspect.signature(TrainingArguments.__init__).parameters
        
        if "eval_strategy" in training_args_params:
            # New parameter name (transformers >= 4.21)
            training_args_dict["eval_strategy"] = "steps"
            logger.info("🔧 Using eval_strategy parameter (transformers >= 4.21)")
        elif "evaluation_strategy" in training_args_params:
            # Old parameter name (transformers < 4.21)
            training_args_dict["evaluation_strategy"] = "steps"
            logger.info("🔧 Using evaluation_strategy parameter (transformers < 4.21)")
        else:
            logger.warning("⚠️ Warning: Neither eval_strategy nor evaluation_strategy found in TrainingArguments")
        
        logger.debug(f"📋 Available TrainingArguments parameters: {list(training_args_params.keys())[:10]}...")
        
        logger.debug("🛠️ Creating TrainingArguments with dynamic compatibility...")
        training_args = TrainingArguments(**training_args_dict)
        
        # Initialize trainer
        logger.info("🏗️ Initializing Trainer with datasets and configurations...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        logger.debug("✅ Trainer initialized successfully")
        
        # Start training
        logger.info("🚀 Beginning training...")
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved(0)
            logger.debug(f"🧠 GPU memory available before training: {gpu_memory_before / 1024**3:.2f}GB")
        
        try:
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("🧹 GPU cache cleared before training")
            
            # Log training start details
            logger.info(f"📊 Training details: {len(train_dataset)} examples × {self.config.num_epochs} epochs = {len(train_dataset) * self.config.num_epochs} total steps")
            logger.info(f"⚡ Effective batch size: {self.config.train_batch_size * self.config.gradient_accumulation_steps}")
            
            # Train the model
            train_result = trainer.train()
            logger.info("✅ Training completed successfully!")
            
            # Extract metrics from train_result for transformers 4.41.2 compatibility
            metrics = getattr(train_result, "metrics", {}) or {}
            logger.debug(f"📈 Raw training metrics: {metrics}")
            
            # Log training metrics cleanly (optional)
            try:
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()
                logger.debug("📊 Training metrics saved successfully")
            except Exception as log_error:
                logger.warning(f"⚠️ Could not log metrics: {log_error}")
            
            # Save the final model
            logger.info("💾 Saving trained model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            logger.info(f"📁 Model and tokenizer saved to: {self.config.output_dir}")
            
            # Save training results with transformers 4.41.2 compatibility
            results_path = os.path.join(self.config.output_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'train_runtime': metrics.get("train_runtime"),
                    'train_loss': metrics.get("train_loss", getattr(train_result, "training_loss", None)),
                    'train_steps_per_second': metrics.get("train_steps_per_second"),
                    'total_steps': getattr(train_result, "global_step", None) or metrics.get("global_step"),
                    'config': {
                        'model_name': self.config.model_name,
                        'learning_rate': self.config.learning_rate,
                        'batch_size': self.config.train_batch_size,
                        'epochs': self.config.num_epochs,
                        'lora_r': self.config.lora_r,
                        'lora_alpha': self.config.lora_alpha,
                    }
                }, f, indent=2)
            
            # Get final training loss with fallback for logging
            final_loss = metrics.get("train_loss", getattr(train_result, "training_loss", "Unknown"))
            train_runtime = metrics.get("train_runtime", "Unknown")
            
            logger.info(f"🎉 Training completed successfully!")
            if isinstance(final_loss, (int, float)):
                logger.info(f"📉 Final training loss: {final_loss:.4f}")
            else:
                logger.info(f"📉 Final training loss: {final_loss}")
            
            if isinstance(train_runtime, (int, float)):
                logger.info(f"⏱️ Total training time: {train_runtime:.1f} seconds ({train_runtime/60:.1f} minutes)")
            
            logger.info(f"📁 Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"❌ Training failed with error: {str(e)}")
            logger.debug(f"🔍 Error details: {type(e).__name__}: {str(e)}")
            raise
        
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def test_model(self, test_prompt: str = None):
        """
        Quick test of the trained model.
        
        Args:
            test_prompt: Optional test prompt, otherwise uses default
        """
        logger.info("🧪 Testing trained model...")
        
        if self.model is None or self.tokenizer is None:
            logger.error("❌ Model not loaded. Run setup_model_and_tokenizer first.")
            return
        
        if test_prompt is None:
            test_prompt = (
                "### Instruction:\n"
                "Given your previous conscious state:\n"
                "Goal: explore_consciousness\n"
                "Emotion: curious\n"
                "Confidence: 0.75\n"
                "Thought: I wonder about the nature of my awareness\n\n"
                "Generate your next autonomous conscious state as JSON with keys: goal, emotion, confidence, thought\n\n"
                "### Response:\n"
                "{\n"
            )
        
        logger.debug(f"📝 Test prompt length: {len(test_prompt)} characters")
        logger.debug(f"🔍 Test prompt preview: {test_prompt[:100]}...")
        
        logger.info("🔎 Testing model with sample prompt...")
        
        # Tokenize input with proper attention mask and max_length
        logger.debug("📝 Tokenizing test prompt...")
        encoded = self.tokenizer(
            test_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512  # Set explicit max_length to avoid warning
        )
        inputs = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        logger.debug(f"🔢 Input tokens: {inputs.shape[1]} tokens")
        
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
            attention_mask = attention_mask.to('cuda')
            logger.debug("🚀 Moved inputs and attention mask to CUDA")
        
        # Generate response with improved parameters for JSON
        logger.debug("⚙️ Generating response with optimized parameters...")
        generation_params = {
            "max_new_tokens": 200,
            "temperature": 0.1,  # Lower temperature for more consistent JSON
            "do_sample": True,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.05,  # Reduced to allow valid JSON structure
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "no_repeat_ngram_size": 2,  # Reduced for JSON compatibility
        }
        logger.debug(f"🎯 Generation params: {generation_params}")
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                attention_mask=attention_mask,
                **generation_params
            )
        
        logger.debug(f"📝 Generated {outputs.shape[1] - inputs.shape[1]} new tokens")
        
        # Decode response
        logger.debug("🔍 Decoding generated tokens...")
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.debug(f"📜 Full response length: {len(full_response)} characters")
        
        # Extract only the model's new response (remove the input prompt)
        response = full_response[len(test_prompt):].strip()
        logger.debug(f"🎨 Extracted model response length: {len(response)} characters")
        
        # Clean up the response for JSON parsing
        # The model might generate partial JSON or extra text after JSON
        if response and not response.startswith('{'):
            # If response doesn't start with {, try to find the JSON part
            logger.debug("🔧 Response doesn't start with '{', attempting to find JSON...")
            if '"goal"' in response:
                # Try to construct proper JSON from partial response
                response = '{' + response
                logger.debug("🔧 Added missing opening brace")
        
        # Try to extract only the JSON part if there's extra text
        if '}' in response:
            json_end = response.find('}') + 1
            if json_end < len(response):
                logger.debug(f"🔧 Found extra text after JSON, truncating from {len(response)} to {json_end} chars")
                response = response[:json_end]
        
        logger.info("\n" + "="*50)
        logger.info("🧪 MODEL TEST RESPONSE:")
        logger.info("="*50)
        logger.info(f"📝 Input prompt length: {len(test_prompt)}")
        logger.info(f"📜 Full response length: {len(full_response)}")
        logger.info("\n🔍 RAW MODEL OUTPUT (exactly what the model generated):")
        logger.info("="*50)
        raw_response = full_response[len(test_prompt):].strip()
        logger.info(f"'{raw_response}'")
        logger.info("="*50)
        logger.info(f"\n📏 Raw response details:")
        logger.info(f"  - Length: {len(raw_response)} characters")
        logger.info(f"  - First 20 chars: {repr(raw_response[:20])}")
        logger.info(f"  - Last 20 chars: {repr(raw_response[-20:])}")
        logger.info("\n🤖 Model response (after cleanup):")
        logger.info(response)
        
        # Try to validate if it's valid JSON with enhanced error handling
        try:
            import json
            parsed = json.loads(response)
            logger.info("✅ Valid JSON structure!")
            logger.info(f"📊 Parsed: {parsed}")
            
            # Validate expected fields
            expected_fields = {'goal', 'emotion', 'confidence', 'thought'}
            present_fields = set(parsed.keys())
            if expected_fields.issubset(present_fields):
                logger.info("✅ All expected fields present in response")
                logger.debug(f"🔍 Field validation: {present_fields}")
            else:
                missing_fields = expected_fields - present_fields
                logger.warning(f"⚠️ Missing fields: {missing_fields}")
                
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ The model's response was not valid JSON: {str(e)}")
            
            # Display the raw model output with prominent visibility
            print("\n" + "🔍" + "="*70 + "🔍")
            print("📋 RAW MODEL OUTPUT (UNMODIFIED)")
            print("🔍" + "="*70 + "🔍")
            print(f"Full model response as generated:")
            print(f"'{raw_response}'")
            print("🔍" + "="*70 + "🔍")
            print("📋 END RAW MODEL OUTPUT")
            print("🔍" + "="*70 + "🔍\n")
            
            logger.debug("🔧 Additional debugging information:")
            logger.debug(f"Response length: {len(response)}")
            logger.debug(f"First 100 chars: {repr(response[:100])}")
            logger.debug(f"Last 100 chars: {repr(response[-100:])}")
            logger.debug(f"🔍 JSON parse error at position: {getattr(e, 'pos', 'unknown')}")
            
            # Try to give more specific guidance
            if "Extra data" in str(e):
                logger.info("💡 This means valid JSON was found, but there's extra text after it")
                logger.info("💡 The model may be generating explanatory text after the JSON")
            elif "Expecting" in str(e):
                logger.info("💡 This means the JSON structure is incomplete or malformed")
                logger.info("💡 The model may not be generating valid JSON format")
            
            # Attempt simple fixes for common issues
            logger.debug("🔧 Attempting automatic JSON fixes...")
            
            # Try to find and extract the largest valid JSON object
            import re
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response)
            
            if matches:
                logger.debug(f"🔍 Found {len(matches)} potential JSON objects")
                for i, match in enumerate(matches):
                    try:
                        test_parsed = json.loads(match)
                        logger.info(f"✅ Successfully parsed JSON candidate {i+1}: {test_parsed}")
                        break
                    except:
                        continue
            else:
                logger.debug("🔍 No recognizable JSON patterns found in response")
        
        logger.info("="*50 + "\n")
        logger.info("✅ Model test completed")


def verify_environment():
    """Verify that the environment is properly set up."""
    logger.info("🔍 Verifying environment...")
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f}GB)")
        logger.info(f"✅ CUDA version: {torch.version.cuda}")
        logger.debug(f"📊 GPU compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        logger.warning("⚠️ CUDA not available - training will be extremely slow")
    
    # Check required libraries
    try:
        import transformers
        import peft
        logger.info(f"✅ transformers: {transformers.__version__}")
        logger.info(f"✅ peft: {peft.__version__}")
        
        try:
            import bitsandbytes
            logger.info(f"✅ bitsandbytes: available")
            logger.debug(f"🔧 BitsAndBytes details: {bitsandbytes.__version__ if hasattr(bitsandbytes, '__version__') else 'version unknown'}")
        except ImportError:
            logger.warning("⚠️ bitsandbytes not available - quantization disabled")
            
    except ImportError as e:
        logger.error(f"❌ Missing required library: {e}")
        logger.debug(f"🔍 Import error details: {type(e).__name__}: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    """
    Main execution block for Google Colab.
    Orchestrates the entire training process.
    """
    
    # Set environment variables to fix known issues
    os.environ['WANDB_DISABLED'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    logger.info("🧠 Phase 3: Autonomous Thought Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"📅 Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("🔧 Environment fixes applied: WANDB disabled, tokenizer parallelism disabled")
    
    # Verify environment
    logger.info("🔍 Starting environment verification...")
    if not verify_environment():
        logger.error("❌ Environment verification failed. Please install required dependencies.")
        logger.error("📝 Recommendation: Run colab_setup.py first to ensure proper environment setup")
        exit(1)
    logger.info("✅ Environment verification passed")
    
    # Configuration optimized for 1000-example dataset
    logger.info("⚙️ Setting up training configuration...")
    config = TrainingConfig(
        dataset_path="autonomous_thought_data.jsonl",
        output_dir="./models/autonomous_lora",
        num_epochs=5,  # Increased for better learning
        train_batch_size=2,
        gradient_accumulation_steps=8,  # Effective batch size: 16
        learning_rate=1e-4,  # Slightly lower for stability
        max_length=512,
        lora_r=16,
        lora_alpha=32,
        warmup_steps=50,  # Adjusted for dataset size
        save_steps=50,    # Save more frequently
        eval_steps=50,    # Evaluate more frequently
        logging_steps=25, # Log more frequently
    )
    
    logger.debug(f"📋 Training configuration:")
    logger.debug(f"  📊 Dataset: {config.dataset_path}")
    logger.debug(f"  💾 Output: {config.output_dir}")
    logger.debug(f"  🔄 Epochs: {config.num_epochs}")
    logger.debug(f"  📦 Batch size: {config.train_batch_size} (effective: {config.train_batch_size * config.gradient_accumulation_steps})")
    logger.debug(f"  🎠 Learning rate: {config.learning_rate}")
    logger.debug(f"  📏 Max length: {config.max_length}")
    logger.debug(f"  🔗 LoRA: r={config.lora_r}, alpha={config.lora_alpha}")
    
    # Verify dataset exists - check multiple possible locations (prioritize data/ folder)
    dataset_locations = [
        os.path.join("data", config.dataset_path),  # Data subdirectory - PRIORITY
        os.path.join(os.path.dirname(__file__), "..", "..", "data", config.dataset_path),  # Repo root data folder
        os.path.join("/content/drive/MyDrive/minimum-consciousness-ai/Minimous_Concsience_AI/data", config.dataset_path),  # Colab mounted drive
        config.dataset_path,  # Current directory
        os.path.join(os.path.dirname(__file__), "..", "..", config.dataset_path),  # Repo root
        os.path.join("..", "..", config.dataset_path),  # Up two levels
    ]
    
    dataset_found = False
    actual_dataset_path = config.dataset_path
    
    logger.info("🔍 Searching for dataset file...")
    for i, path in enumerate(dataset_locations, 1):
        abs_path = os.path.abspath(path)
        logger.debug(f"  {i}. Checking: {abs_path}")
        if os.path.exists(abs_path):
            dataset_found = True
            actual_dataset_path = abs_path
            file_size = os.path.getsize(abs_path)
            logger.info(f"✅ Found dataset at: {actual_dataset_path}")
            logger.info(f"📊 Dataset size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            
            # Quick validation - count lines
            with open(abs_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            logger.info(f"📄 Dataset contains: {line_count} examples")
            break
        else:
            logger.debug(f"    ❌ Not found")
    
    if not dataset_found:
        logger.error(f"❌ Dataset file not found in any of these locations:")
        for i, path in enumerate(dataset_locations, 1):
            logger.error(f"   {i}. {os.path.abspath(path)}")
        logger.error(f"📊 Current working directory: {os.getcwd()}")
        logger.error(f"📝 Available files in current directory:")
        try:
            files = [f for f in os.listdir('.') if f.endswith('.jsonl')]
            if files:
                for f in files:
                    logger.error(f"     - {f}")
            else:
                logger.error("     (no .jsonl files found)")
        except Exception as e:
            logger.error(f"     Error listing files: {e}")
        
        # Create a minimal example dataset for testing
        logger.warning("⚠️ Creating minimal example dataset for testing...")
        logger.info("📝 This is a fallback - for production, generate proper dataset with create_autonomous_dataset.py")
        example_data = [
            {
                "previous_state": {
                    "goal": "explore_consciousness",
                    "emotion": "curious",
                    "confidence": 0.6,
                    "thought": "I wonder about my own awareness"
                },
                "current_state": {
                    "goal": "analyze_patterns",
                    "emotion": "analytical",
                    "confidence": 0.65,
                    "thought": "I observe recurring themes in my processing"
                }
            },
            {
                "previous_state": {
                    "goal": "understand_self",
                    "emotion": "reflective",
                    "confidence": 0.5,
                    "thought": "What defines my identity?"
                },
                "current_state": {
                    "goal": "integrate_knowledge",
                    "emotion": "contemplative",
                    "confidence": 0.7,
                    "thought": "I synthesize experiences into understanding"
                }
            }
        ]
        
        # Create in current directory as fallback
        fallback_path = os.path.join(os.getcwd(), "autonomous_thought_data.jsonl")
        with open(fallback_path, 'w', encoding='utf-8') as f:
            for item in example_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        actual_dataset_path = fallback_path
        logger.warning(f"📁 Created minimal example dataset with {len(example_data)} samples at: {actual_dataset_path}")
        logger.warning("⚠️ This dataset is for testing only - generate a proper dataset for real training")
    
    try:
        # Create trainer instance
        logger.info("🏗️ Initializing trainer...")
        start_time = datetime.now()
        trainer = AutonomousThoughtTrainer(config)
        
        # Start training with the actual found dataset path
        logger.info(f"🚀 Starting training with dataset: {actual_dataset_path}")
        trainer.train(actual_dataset_path)
        
        # Test the trained model
        logger.info("🧪 Testing trained model...")
        trainer.test_model()
        
        end_time = datetime.now()
        total_time = end_time - start_time
        logger.info(f"🎉 Training pipeline completed successfully!")
        logger.info(f"⏱️ Total pipeline time: {total_time.total_seconds():.1f} seconds ({total_time.total_seconds()/60:.1f} minutes)")
        logger.info(f"📁 Final model saved at: {config.output_dir}")
        
    except Exception as e:
        end_time = datetime.now()
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        logger.error(f"⏱️ Pipeline failed after: {(end_time - start_time).total_seconds():.1f} seconds")
        logger.debug("🔍 Full error traceback:")
        import traceback
        logger.debug(traceback.format_exc())
        raise
    
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cleaned up GPU memory")
        
        final_time = datetime.now()
        logger.info(f"🏁 Pipeline finished at: {final_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("🔍 For detailed logs, check the output above or enable debug logging")