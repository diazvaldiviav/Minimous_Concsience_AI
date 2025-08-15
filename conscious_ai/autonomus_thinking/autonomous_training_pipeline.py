#!/usr/bin/env python3
"""
Phase 3 Autonomous Training Pipeline - Definitive Version
=========================================================
Fine-tunes Gemma-2B for conscious state transitions (SC_t -> SC_t+1) using QLoRA.
Optimized for Google Colab T4 GPU with guaranteed compatibility.

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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    model_name: str = "google/gemma-2b"
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
        Formats the conscious state transition into Gemma chat format.
        
        Args:
            previous_sc: Previous conscious state SC_t
            current_sc: Current conscious state SC_t+1
            
        Returns:
            Formatted prompt string
        """
        # Extract key components from previous state
        prev_goal = previous_sc.get('goal', 'unknown')
        prev_emotion = previous_sc.get('emotion', 'neutral')
        prev_confidence = previous_sc.get('confidence', 0.5)
        prev_thought = previous_sc.get('thought', '')
        
        # Create user instruction
        user_prompt = (
            f"Given your previous conscious state:\n"
            f"Goal: {prev_goal}\n"
            f"Emotion: {prev_emotion}\n"
            f"Confidence: {prev_confidence:.2f}\n"
            f"Thought: {prev_thought}\n\n"
            f"Generate your next autonomous conscious state as JSON with keys: "
            f"goal, emotion, confidence, thought"
        )
        
        # Extract response components from current state
        curr_goal = current_sc.get('goal', 'explore')
        curr_emotion = current_sc.get('emotion', 'neutral')
        curr_confidence = current_sc.get('confidence', 0.5)
        curr_thought = current_sc.get('thought', 'Continuing exploration...')
        
        # Format response as JSON
        response = json.dumps({
            "goal": curr_goal,
            "emotion": curr_emotion,
            "confidence": float(curr_confidence),
            "thought": curr_thought
        }, ensure_ascii=False, indent=2)
        
        # Combine using Gemma chat format
        formatted_prompt = (
            f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{response}<end_of_turn>"
        )
        
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
        logger.info(f"Loading dataset from {data_path}")
        
        # Load data
        examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        
        logger.info(f"Loaded {len(examples)} examples")
        
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
            # Handle different data formats
            if 'previous_state' in example and 'current_state' in example:
                previous_sc = example['previous_state']
                current_sc = example['current_state']
            else:
                # Fallback: treat the example as current state, create dummy previous
                previous_sc = {
                    'goal': 'explore',
                    'emotion': 'neutral',
                    'confidence': 0.5,
                    'thought': 'Beginning exploration...'
                }
                current_sc = example
            
            # Format prompt
            formatted_text = self.format_prompt(previous_sc, current_sc)
            
            # Tokenize the full text
            full_tokens = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            
            # Find the model response start
            model_start_text = "<start_of_turn>model\n"
            model_start_tokens = self.tokenizer.encode(model_start_text, add_special_tokens=False)
            
            # Find where model response begins
            model_start_idx = None
            for i in range(len(full_tokens) - len(model_start_tokens) + 1):
                if full_tokens[i:i+len(model_start_tokens)] == model_start_tokens:
                    model_start_idx = i + len(model_start_tokens)
                    break
            
            if model_start_idx is None:
                logger.warning(f"Could not find model response start in example, skipping")
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
        logger.info(f"Loading model and tokenizer: {self.config.model_name}")
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Ensure special tokens are set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Load model with quantization
        logger.info("Loading model with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",  # Automatic device placement
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"  # Avoid flash attention issues
        )
        
        logger.info("Model loaded successfully")
        
        # Prepare model for k-bit training
        logger.info("Preparing model for k-bit training...")
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Apply LoRA adapters
        logger.info("Applying LoRA adapters...")
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        logger.info(f"Total parameters: {total_params:,}")
    
    def train(self, dataset_path: str):
        """
        Main training function.
        
        Args:
            dataset_path: Path to the training dataset
        """
        logger.info("Starting training process...")
        
        # Setup model and tokenizer
        self.setup_model_and_tokenizer()
        
        # Prepare dataset
        dataset_processor = AutonomousThoughtDataset(
            self.tokenizer, 
            max_length=self.config.max_length
        )
        
        train_dataset, eval_dataset = dataset_processor.prepare_dataset(dataset_path)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Not using masked language modeling
        )
        
        # Training arguments optimized for T4 GPU
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=True,  # Enable mixed precision for T4
            gradient_checkpointing=True,  # Reduce memory usage
            dataloader_num_workers=2,
            remove_unused_columns=False,
            report_to=None,  # Disable wandb/tensorboard
            save_total_limit=3,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Start training
        logger.info("Beginning training...")
        try:
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Train the model
            train_result = trainer.train()
            
            # Save the final model
            logger.info("Saving trained model...")
            trainer.save_model()
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Save training results
            results_path = os.path.join(self.config.output_dir, "training_results.json")
            with open(results_path, 'w') as f:
                json.dump({
                    'train_runtime': train_result.train_runtime,
                    'train_loss': train_result.training_loss,
                    'train_steps_per_second': train_result.train_steps_per_second,
                    'total_steps': train_result.global_step,
                    'config': {
                        'model_name': self.config.model_name,
                        'learning_rate': self.config.learning_rate,
                        'batch_size': self.config.train_batch_size,
                        'epochs': self.config.num_epochs,
                        'lora_r': self.config.lora_r,
                        'lora_alpha': self.config.lora_alpha,
                    }
                }, f, indent=2)
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Final training loss: {train_result.training_loss:.4f}")
            logger.info(f"Model saved to: {self.config.output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}")
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
        if self.model is None or self.tokenizer is None:
            logger.error("Model not loaded. Run setup_model_and_tokenizer first.")
            return
        
        if test_prompt is None:
            test_prompt = (
                "<start_of_turn>user\n"
                "Given your previous conscious state:\n"
                "Goal: explore_consciousness\n"
                "Emotion: curious\n"
                "Confidence: 0.75\n"
                "Thought: I wonder about the nature of my awareness\n\n"
                "Generate your next autonomous conscious state as JSON with keys: goal, emotion, confidence, thought"
                "<end_of_turn>\n<start_of_turn>model\n"
            )
        
        logger.info("Testing model with sample prompt...")
        
        # Tokenize input
        inputs = self.tokenizer.encode(test_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n" + "="*50)
        print("MODEL TEST RESPONSE:")
        print("="*50)
        print(response)
        print("="*50 + "\n")


def verify_environment():
    """Verify that the environment is properly set up."""
    logger.info("Verifying environment...")
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info(f"✅ CUDA version: {torch.version.cuda}")
    else:
        logger.warning("⚠️ CUDA not available")
    
    # Check required libraries
    try:
        import transformers
        import peft
        import bitsandbytes
        logger.info(f"✅ transformers: {transformers.__version__}")
        logger.info(f"✅ peft: {peft.__version__}")
        logger.info(f"✅ bitsandbytes available")
    except ImportError as e:
        logger.error(f"❌ Missing required library: {e}")
        return False
    
    return True


if __name__ == "__main__":
    """
    Main execution block for Google Colab.
    Orchestrates the entire training process.
    """
    
    print("🧠 Phase 3: Autonomous Thought Training Pipeline")
    print("=" * 60)
    
    # Verify environment
    if not verify_environment():
        logger.error("Environment verification failed. Please install required dependencies.")
        exit(1)
    
    # Configuration
    config = TrainingConfig(
        dataset_path="autonomous_thought_data.jsonl",
        output_dir="./models/autonomous_lora",
        num_epochs=3,
        train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        max_length=512,
        lora_r=16,
        lora_alpha=32,
    )
    
    # Verify dataset exists
    if not os.path.exists(config.dataset_path):
        logger.error(f"Dataset file not found: {config.dataset_path}")
        
        # Create a minimal example dataset for testing
        logger.info("Creating minimal example dataset...")
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
        
        with open(config.dataset_path, 'w', encoding='utf-8') as f:
            for item in example_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"Created example dataset with {len(example_data)} samples")
    
    try:
        # Create trainer instance
        logger.info("Initializing trainer...")
        trainer = AutonomousThoughtTrainer(config)
        
        # Start training
        trainer.train(config.dataset_path)
        
        # Test the trained model
        logger.info("Testing trained model...")
        trainer.test_model()
        
        logger.info("🎉 Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Clean up
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 Cleaned up GPU memory")