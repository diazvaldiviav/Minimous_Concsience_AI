"""
Pipeline de entrenamiento para el modelo de pensamiento autónomo
Entrena transiciones SCₜ₋₁ → SCₜ sin input externo
Auto-toggle: GPU (bnb 4-bit) / CPU (FP32)
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
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# -------------------------------------
# Logging
# -------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------------
# Dataset
# -------------------------------------
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
        prev_sc = example["previous_SC"]
        curr_sc = example["current_SC"]

        prompt = f"""<start_of_turn>user
Given the previous conscious state:
{json.dumps(prev_sc, ensure_ascii=False, indent=2)}

Generate the next autonomous thought in JSON format.<end_of_turn>
<start_of_turn>model
{json.dumps(curr_sc, ensure_ascii=False, indent=2)}<end_of_turn>"""

        enc = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = enc["input_ids"].clone()

        # Enmascarar el prompt del usuario: solo calcular pérdida sobre la respuesta del modelo
        model_prompt_text = "<start_of_turn>model"
        model_prompt_tokens = self.tokenizer(model_prompt_text, add_special_tokens=False)["input_ids"]
        labels_list = labels[0].tolist()
        sep_idx = -1
        for i in range(len(labels_list) - len(model_prompt_tokens) + 1):
            if labels_list[i : i + len(model_prompt_tokens)] == model_prompt_tokens:
                sep_idx = i
                break
        if sep_idx != -1:
            labels[0, :sep_idx] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }

# -------------------------------------
# Trainer wrapper
# -------------------------------------
class AutonomousThoughtTrainer:
    """Entrenador para el modelo de pensamiento autónomo con auto-toggle CPU/GPU."""

    def __init__(
        self,
        model_name: str = "google/gemma-2b",
        output_dir: str = "./models/autonomous_gemma_lora",
        logs_dir: str = "./logs/autonomous_gemma_training",
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        self.use_gpu = torch.cuda.is_available()

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        logger.info(f"Inicializando entrenador autónomo con {model_name} en: {'GPU' if self.use_gpu else 'CPU'}")

    def setup_model_and_tokenizer(self):
        """Configura el modelo Gemma + tokenizer (bnb 4-bit en GPU, FP32 en CPU)."""
        logger.info(f"Cargando núcleo cognitivo: {self.model_name}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True, trust_remote_code=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}

        if self.use_gpu:
            # --- GPU: bitsandbytes 4-bit ---
            from transformers import BitsAndBytesConfig

            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # bfloat16 en GPU modernas
            )
            model_kwargs.update(dict(quantization_config=bnb_cfg, device_map="auto", torch_dtype=torch.bfloat16))
        else:
            # --- CPU: sin quantización ---
            model_kwargs.update(dict(device_map={"": "cpu"}, torch_dtype=torch.float32))

        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=False, **model_kwargs)
        except Exception as e:
            import traceback

            print("🚨 Error al cargar modelo:", e)
            traceback.print_exc()
            raise

        # Gradient checkpointing y preparación k-bit solo si hay GPU/quant
        if self.use_gpu:
            self.base_model.gradient_checkpointing_enable()
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # LoRA
        lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.base_model, lora_cfg)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ Modelo listo: trainable={trainable:,} / total={total:,} ({100*trainable/total:.2f}%)")

    def train(
        self,
        dataset_path: str,
        num_epochs: int = 3,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 16,
        max_length: int = 512,
    ):
        """Entrena el modelo de pensamiento autónomo."""
        self.setup_model_and_tokenizer()

        train_dataset, eval_dataset = self.load_dataset(dataset_path, max_length=max_length)
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

        # --- TrainingArguments con compat shim por si Colab se pone rebelde ---
        from transformers.training_args import TrainingArguments as HFTrainingArguments
        import inspect

        TA = HFTrainingArguments
        if "evaluation_strategy" not in inspect.signature(TA.__init__).parameters:
            class _CompatTA(TA):
                def __init__(self, *args, evaluation_strategy=None, **kwargs):
                    if evaluation_strategy and "evaluate_during_training" in inspect.signature(super().__init__).parameters:
                        kwargs["evaluate_during_training"] = (evaluation_strategy != "no")
                    super().__init__(*args, **kwargs)
            TA = _CompatTA

        fp16_flag = self.use_gpu  # solo en GPU
        bf16_flag = self.use_gpu  # bfloat16 en GPU modernas; en CPU = False
        no_cuda_flag = not self.use_gpu

        training_args = TA(
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
            fp16=fp16_flag,
            bf16=bf16_flag,
            no_cuda=no_cuda_flag,
            gradient_checkpointing=self.use_gpu,  # en CPU suele empeorar
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

        logger.info(f"Iniciando entrenamiento ({'GPU 4-bit' if self.use_gpu else 'CPU FP32'})...")
        trainer.train()

        logger.info("Guardando modelo final...")
        trainer.save_model()

    def load_dataset(self, dataset_path: str, max_length: int) -> Tuple[Dataset, Dataset]:
        """Carga y divide el dataset de pensamientos autónomos."""
        logger.info(f"Cargando historial de pensamientos desde {dataset_path}")
        examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
        logger.info(f"Total de transiciones cargadas: {len(examples)}")

        split_idx = int(len(examples) * 0.90)
        train_examples = examples[:split_idx]
        eval_examples = examples[split_idx:]

        train_dataset = AutonomousThoughtDataset(train_examples, self.tokenizer, max_length=max_length)
        eval_dataset = AutonomousThoughtDataset(eval_examples, self.tokenizer, max_length=max_length)

        logger.info(f"División de datos - Entrenamiento: {len(train_dataset)}, Evaluación: {len(eval_dataset)}")
        return train_dataset, eval_dataset


if __name__ == "__main__":
    dataset_file = "./data/autonomous_thought_data.jsonl"

    if not os.path.exists(dataset_file):
        logger.error(f"Error: No se encontró el dataset en '{dataset_file}'")
        logger.error("Por favor, primero ejecuta 'create_autonomous_dataset.py' para generar los datos.")
        raise SystemExit(1)

    trainer = AutonomousThoughtTrainer()
    trainer.train(dataset_path=dataset_file)
    logger.info("✓ Proceso de entrenamiento finalizado.")
