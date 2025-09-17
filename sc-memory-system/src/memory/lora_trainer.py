"""
LoRA Trainer for SC Memory System.

This module trains LoRA adapters on conversation-specific training data,
enabling the base model to "remember" specific conversations through
learned adapter weights.
"""

import asyncio
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
import numpy as np

# PEFT and training dependencies
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import (
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
    )
    from datasets import Dataset
    TRAINING_AVAILABLE = True
except ImportError:
    TRAINING_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "Training dependencies not available. Install with: "
        "pip install peft transformers datasets accelerate"
    )

from ..core.config import Settings, get_settings
from ..core.exceptions import LoRATrainingError, ModelLoadError
from ..core.models import TrainingExample, ConversationAdapter
from .base_model import BaseModelManager

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA adapter trainer for conversation-specific memory.
    
    Trains lightweight LoRA adapters on conversation data to enable
    the base model to "remember" specific conversations without
    requiring context injection.
    
    Features:
    - Conversation-specific adapter training
    - Memory-efficient LoRA configuration
    - Training progress monitoring
    - Adapter persistence and metadata
    - Performance evaluation
    """
    
    def __init__(
        self,
        base_model: BaseModelManager,
        settings: Optional[Settings] = None
    ) -> None:
        """
        Initialize the LoRA trainer.
        
        Args:
            base_model: Base model manager instance
            settings: Application settings
        """
        if not TRAINING_AVAILABLE:
            raise ModelLoadError(
                message="Training dependencies not available",
                model_name="lora_trainer",
                model_type="training"
            )
        
        self._settings = settings or get_settings()
        self._base_model = base_model
        
        # Training configuration
        self._lora_config = self._setup_lora_config()
        self._adapters_dir = self._settings.lora_training.adapters_dir
        
        # Ensure adapters directory exists
        self._adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self._trained_adapters: List[ConversationAdapter] = []
        self._training_history: List[Dict[str, Any]] = []
        
        logger.info(
            f"Initialized LoRATrainer (r={self._settings.lora_training.r}, "
            f"alpha={self._settings.lora_training.alpha})"
        )
    
    def _setup_lora_config(self) -> "LoraConfig":
        """
        Setup LoRA configuration for conversation memory.
        
        Returns:
            LoRA configuration optimized for memory consolidation
        """
        return LoraConfig(
            r=self._settings.lora_training.r,
            lora_alpha=self._settings.lora_training.alpha,
            target_modules=self._settings.lora_training.target_modules,
            lora_dropout=0.1,  # Reasonable dropout for memory tasks
            bias="none",
            task_type="CAUSAL_LM",
        )
    
    async def train_conversation_adapter(
        self,
        training_data_path: Path,
        conversation_id: str,
        provider: str = None,
        external_user_id: str = None,
        external_chat_id: str = None,
        proposal_id: str = None
    ) -> ConversationAdapter:
        """
        Train a LoRA adapter on conversation-specific data.

        Args:
            training_data_path: Path to training data JSONL file
            conversation_id: Unique conversation identifier
            provider: Provider identifier (e.g., 'openai', 'anthropic')
            external_user_id: External user identifier
            external_chat_id: External chat identifier
            proposal_id: Original MEP proposal ID

        Returns:
            Trained conversation adapter metadata
        """
        adapter_id = f"conv_{conversation_id}_{int(time.time())}"
        
        try:
            # Ensure base model is loaded
            if not self._base_model.is_loaded():
                await self._base_model.load_model()
            
            # Load and prepare training data
            dataset = await self._load_training_dataset(training_data_path)
            
            # Create PEFT model
            base_model = self._base_model.get_model()
            peft_model = get_peft_model(base_model, self._lora_config)
            
            # Setup training
            training_results = await self._run_training_loop(
                peft_model, dataset, adapter_id
            )
            
            # Save adapter with complete metadata
            adapter_path = await self.save_conversation_adapter(
                peft_model, adapter_id, provider, external_user_id, external_chat_id, proposal_id
            )
            
            # Evaluate adapter performance
            performance_metrics = self._evaluate_adapter_performance(
                peft_model, dataset
            )
            
            # Create adapter metadata
            adapter = ConversationAdapter(
                adapter_id=adapter_id,
                conversation_id=conversation_id,
                source_proposal_id=conversation_id,  # For now, same as conversation
                adapter_path=adapter_path,
                training_examples_count=len(dataset),
                performance_metrics=performance_metrics
            )
            
            self._trained_adapters.append(adapter)
            
            # Clean up GPU memory
            del peft_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(
                f"Successfully trained adapter {adapter_id} for conversation {conversation_id} "
                f"({len(dataset)} examples, {training_results['training_time']:.2f}s)"
            )
            
            return adapter
            
        except Exception as e:
            error_msg = f"Failed to train conversation adapter: {str(e)}"
            logger.error(error_msg)
            raise LoRATrainingError(
                message=error_msg,
                adapter_id=adapter_id,
                details={"conversation_id": conversation_id}
            )
    
    async def _load_training_dataset(self, data_path: Path) -> Dataset:
        """
        Load and prepare training dataset.
        
        Args:
            data_path: Path to training data file
            
        Returns:
            Prepared HuggingFace dataset
        """
        try:
            # Load training examples
            training_examples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line.strip())
                    training_examples.append(example)
            
            if len(training_examples) == 0:
                raise ValueError("No training examples found")
            
            # Create dataset
            dataset_dict = {
                "instruction": [ex["instruction"] for ex in training_examples],
                "response": [ex["response"] for ex in training_examples],
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Tokenize dataset
            tokenized_dataset = dataset.map(
                self._tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            logger.info(f"Loaded dataset with {len(tokenized_dataset)} examples")
            
            return tokenized_dataset
            
        except Exception as e:
            error_msg = f"Failed to load training dataset: {str(e)}"
            logger.error(error_msg)
            raise LoRATrainingError(
                message=error_msg,
                details={"data_path": str(data_path)}
            )
    
    def _tokenize_function(self, examples):
        """
        Tokenize training examples.
        
        Args:
            examples: Batch of examples to tokenize
            
        Returns:
            Tokenized examples
        """
        tokenizer = self._base_model.get_tokenizer()
        
        # Format as instruction-response pairs
        formatted_texts = []
        for instruction, response in zip(examples["instruction"], examples["response"]):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}\n"
            formatted_texts.append(text)
        
        # Tokenize with padding for consistent batch processing
        tokenized = tokenizer(
            formatted_texts,
            truncation=True,
            padding=True,  # Enable padding for consistent tensor shapes
            max_length=512,  # Reasonable length for conversation memory
            return_tensors=None
        )

        # Set labels for causal language modeling
        # Convert to proper format for DataCollatorForLanguageModeling
        labels = []
        for input_ids in tokenized["input_ids"]:
            # Copy input_ids as labels, DataCollator will handle padding tokens
            labels.append(input_ids.copy())

        tokenized["labels"] = labels

        return tokenized
    
    async def _run_training_loop(
        self,
        peft_model,
        dataset: Dataset,
        adapter_id: str
    ) -> Dict[str, Any]:
        """
        Run the actual training loop.
        
        Args:
            peft_model: PEFT model to train
            dataset: Training dataset
            adapter_id: Adapter identifier
            
        Returns:
            Training results and metrics
        """
        start_time = time.time()
        
        try:
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=str(self._adapters_dir / f"training_{adapter_id}"),
                num_train_epochs=3,  # Few epochs for conversation memory
                per_device_train_batch_size=self._settings.lora_training.batch_size,
                gradient_accumulation_steps=self._settings.lora_training.gradient_accumulation_steps,
                warmup_steps=50,  # Fewer warmup steps for small datasets
                learning_rate=self._settings.lora_training.learning_rate,
                fp16=torch.cuda.is_available(),
                logging_steps=10,
                save_steps=self._settings.lora_training.training_steps // 2,
                save_strategy="steps",
                evaluation_strategy="no",  # Skip evaluation for MVP
                dataloader_drop_last=False,
                remove_unused_columns=False,
                report_to=None,  # Disable logging integrations
            )
            
            # Data collator
            tokenizer = self._base_model.get_tokenizer()
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=dataset,
                data_collator=data_collator,
                tokenizer=tokenizer,
            )
            
            # Train the model
            logger.info(f"Starting training for adapter {adapter_id}")
            train_result = trainer.train()
            
            training_time = time.time() - start_time
            
            # Record training history
            training_record = {
                "adapter_id": adapter_id,
                "training_time": training_time,
                "final_loss": train_result.training_loss,
                "training_examples": len(dataset),
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            self._training_history.append(training_record)
            
            return {
                "training_time": training_time,
                "final_loss": train_result.training_loss,
                "train_result": train_result,
            }
            
        except Exception as e:
            error_msg = f"Training loop failed: {str(e)}"
            logger.error(error_msg)
            raise LoRATrainingError(
                message=error_msg,
                adapter_id=adapter_id,
                training_step=0
            )
    
    async def save_conversation_adapter(
        self,
        peft_model,
        adapter_id: str,
        provider: str = None,
        external_user_id: str = None,
        external_chat_id: str = None,
        proposal_id: str = None
    ) -> Path:
        """
        Save trained conversation adapter.
        
        Args:
            peft_model: Trained PEFT model
            adapter_id: Adapter identifier
            
        Returns:
            Path to saved adapter
        """
        adapter_path = self._adapters_dir / adapter_id
        adapter_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save the adapter
            peft_model.save_pretrained(str(adapter_path))
            
            # Save adapter metadata WITH CRITICAL FIELDS for retrieval
            metadata = {
                "adapter_id": adapter_id,
                "provider": provider,  # CRITICAL: Needed for MAP query matching
                "external_user_id": external_user_id,  # CRITICAL: Needed for MAP query matching
                "external_chat_id": external_chat_id,  # CRITICAL: Needed for MAP query matching
                "proposal_id": proposal_id,
                "base_model": self._base_model._model_name,
                "lora_config": {
                    "r": self._lora_config.r,
                    "alpha": self._lora_config.lora_alpha,
                    "target_modules": list(self._lora_config.target_modules) if isinstance(self._lora_config.target_modules, set) else self._lora_config.target_modules,
                },
                "training_timestamp": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow().isoformat(),  # For sorting
                "training_settings": {
                    "learning_rate": self._settings.lora_training.learning_rate,
                    "batch_size": self._settings.lora_training.batch_size,
                    "training_steps": self._settings.lora_training.training_steps,
                }
            }
            
            with open(adapter_path / "adapter_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Also save tokenizer for convenience
            tokenizer = self._base_model.get_tokenizer()
            tokenizer.save_pretrained(str(adapter_path))
            
            logger.info(f"Saved adapter {adapter_id} to {adapter_path}")
            
            return adapter_path
            
        except Exception as e:
            error_msg = f"Failed to save adapter: {str(e)}"
            logger.error(error_msg)
            raise LoRATrainingError(
                message=error_msg,
                adapter_id=adapter_id,
                details={"adapter_path": str(adapter_path)}
            )
    
    def _evaluate_adapter_performance(
        self,
        peft_model,
        dataset: Dataset
    ) -> Dict[str, float]:
        """
        Evaluate adapter performance on training data.
        
        Args:
            peft_model: Trained PEFT model
            dataset: Training dataset
            
        Returns:
            Performance metrics
        """
        try:
            # Simple evaluation: calculate perplexity on training data
            peft_model.eval()
            
            total_loss = 0.0
            num_examples = 0
            
            with torch.no_grad():
                for i in range(min(10, len(dataset))):  # Evaluate on subset
                    example = dataset[i]
                    
                    input_ids = torch.tensor([example["input_ids"]])
                    labels = torch.tensor([example["labels"]])
                    
                    if torch.cuda.is_available():
                        input_ids = input_ids.cuda()
                        labels = labels.cuda()
                    
                    outputs = peft_model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    num_examples += 1
            
            avg_loss = total_loss / max(1, num_examples)
            perplexity = np.exp(avg_loss)
            
            return {
                "average_loss": avg_loss,
                "perplexity": perplexity,
                "evaluated_examples": num_examples,
            }
            
        except Exception as e:
            logger.warning(f"Adapter evaluation failed: {e}")
            return {"evaluation_error": str(e)}
    
    def get_trained_adapters(self) -> List[ConversationAdapter]:
        """
        Get list of trained adapters.
        
        Returns:
            List of conversation adapters
        """
        return self._trained_adapters.copy()
    
    def get_training_history(self) -> List[Dict[str, Any]]:
        """
        Get training history.
        
        Returns:
            List of training records
        """
        return self._training_history.copy()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """
        Get training statistics.
        
        Returns:
            Dictionary with training stats
        """
        return {
            "total_adapters_trained": len(self._trained_adapters),
            "total_training_sessions": len(self._training_history),
            "adapters_directory": str(self._adapters_dir),
            "lora_config": {
                "r": self._lora_config.r,
                "alpha": self._lora_config.lora_alpha,
                "target_modules": list(self._lora_config.target_modules) if isinstance(self._lora_config.target_modules, set) else self._lora_config.target_modules,
            },
            "training_settings": {
                "learning_rate": self._settings.lora_training.learning_rate,
                "batch_size": self._settings.lora_training.batch_size,
                "training_steps": self._settings.lora_training.training_steps,
            },
            "base_model": self._base_model._model_name if self._base_model else None,
            "training_available": TRAINING_AVAILABLE,
        }


# Convenience function for creating LoRA trainer
def create_lora_trainer(
    base_model: BaseModelManager,
    settings: Optional[Settings] = None
) -> LoRATrainer:
    """
    Create and return a LoRATrainer instance.
    
    Args:
        base_model: Base model manager
        settings: Application settings
        
    Returns:
        LoRATrainer instance
    """
    return LoRATrainer(
        base_model=base_model,
        settings=settings
    )


# Export main classes and functions
__all__ = [
    "LoRATrainer",
    "create_lora_trainer",
]