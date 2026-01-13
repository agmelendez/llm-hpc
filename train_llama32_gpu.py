"""
LLaMA 3.2 QLoRA Fine-tuning Script with Unsloth
================================================

This script fine-tunes the LLaMA 3.2 1B Instruct model using QLoRA (4-bit quantization)
through the Unsloth library for efficient training on GPU.

Features:
    - 4-bit quantization for reduced memory footprint
    - LoRA adapters for parameter-efficient fine-tuning
    - Automatic train/validation split
    - Early stopping support
    - Comprehensive metrics logging
    - Configurable through environment variables or config.py

Usage:
    # Using default configuration
    python train_llama32_gpu.py

    # Using environment variables to override defaults
    MODEL="meta-llama/Llama-3.2-1B" DATA="./data/my_data.jsonl" python train_llama32_gpu.py

    # On HPC cluster with SLURM
    sbatch train_block_full_gpu.sbatch

Requirements:
    - Python 3.8+
    - PyTorch 2.7.1+
    - Transformers, Datasets, Unsloth
    - CUDA-capable GPU (recommended: A100 80GB)

Author: Alison Lobo Salas
Organization: Universidad de Costa Rica (UCR)
"""

import os
import sys
import math
import time
import json
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.trainer import SFTTrainer
from transformers import EarlyStoppingCallback

# Add project root to path to import config
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import config
    USE_CONFIG_FILE = True
except ImportError:
    print("⚠️  Warning: config.py not found. Using environment variables only.")
    USE_CONFIG_FILE = False


# ==============================================================================
# Configuration
# ==============================================================================
def load_configuration():
    """
    Load configuration from config.py file or environment variables.
    Environment variables take precedence over config.py.

    Returns:
        dict: Configuration dictionary with all training parameters
    """
    if USE_CONFIG_FILE:
        env_config = config.get_env_config()
        cfg = {
            "MODEL_NAME": env_config["MODEL_NAME"],
            "DATASET_PATH": env_config["DATASET_PATH"],
            "OUTPUT_DIR": env_config["OUTPUT_DIR"],
            "MAX_STEPS": env_config["MAX_STEPS"],
            "EPOCHS": env_config["EPOCHS"],
            "EVAL_STEPS": env_config["EVAL_STEPS"],
            "LOG_STEPS": env_config["LOG_STEPS"],
            "MAX_SEQ_LENGTH": config.MAX_SEQ_LENGTH,
            "LOAD_IN_4BIT": config.LOAD_IN_4BIT,
            "LORA_R": config.LORA_R,
            "LORA_ALPHA": config.LORA_ALPHA,
            "LORA_DROPOUT": config.LORA_DROPOUT,
            "LORA_TARGET_MODULES": config.LORA_TARGET_MODULES,
            "PER_DEVICE_TRAIN_BATCH_SIZE": config.PER_DEVICE_TRAIN_BATCH_SIZE,
            "GRADIENT_ACCUMULATION_STEPS": config.GRADIENT_ACCUMULATION_STEPS,
            "LEARNING_RATE": config.LEARNING_RATE,
            "WARMUP_RATIO": config.WARMUP_RATIO,
            "LR_SCHEDULER_TYPE": config.LR_SCHEDULER_TYPE,
            "SAVE_STRATEGY": config.SAVE_STRATEGY,
            "SAVE_TOTAL_LIMIT": config.SAVE_TOTAL_LIMIT,
            "EARLY_STOPPING_PATIENCE": config.EARLY_STOPPING_PATIENCE,
            "TRAIN_TEST_SPLIT_RATIO": config.TRAIN_TEST_SPLIT_RATIO,
            "RANDOM_SEED": config.RANDOM_SEED,
        }
    else:
        # Fallback to environment variables only
        cfg = {
            "MODEL_NAME": os.getenv("MODEL", "meta-llama/Llama-3.2-1B-Instruct"),
            "DATASET_PATH": os.getenv("DATA", str(project_root / "data" / "base.jsonl")),
            "OUTPUT_DIR": os.getenv("OUT", str(project_root / "outputs" / "llama32_qlora")),
            "MAX_STEPS": int(os.getenv("MAX_STEPS", "0")),
            "EPOCHS": int(os.getenv("EPOCHS", "60")),
            "EVAL_STEPS": int(os.getenv("EVAL_STEPS", "200")),
            "LOG_STEPS": int(os.getenv("LOG_STEPS", "20")),
            "MAX_SEQ_LENGTH": 4096,
            "LOAD_IN_4BIT": True,
            "LORA_R": 16,
            "LORA_ALPHA": 32,
            "LORA_DROPOUT": 0.05,
            "LORA_TARGET_MODULES": ["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"],
            "PER_DEVICE_TRAIN_BATCH_SIZE": 2,
            "GRADIENT_ACCUMULATION_STEPS": 4,
            "LEARNING_RATE": 2e-4,
            "WARMUP_RATIO": 0.03,
            "LR_SCHEDULER_TYPE": "cosine",
            "SAVE_STRATEGY": "epoch",
            "SAVE_TOTAL_LIMIT": 3,
            "EARLY_STOPPING_PATIENCE": 3,
            "TRAIN_TEST_SPLIT_RATIO": 0.1,
            "RANDOM_SEED": 42,
        }

    return cfg


def print_system_info():
    """Print system and GPU information for debugging."""
    print("\n" + "="*70)
    print("🖥️  SYSTEM INFORMATION")
    print("="*70)
    print(f"Python version:   {sys.version.split()[0]}")
    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version:     {torch.version.cuda}")
        print(f"GPU device:       {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  WARNING: No GPU detected! Training will be very slow.")
    print("="*70 + "\n")


def format_prompt(example):
    """
    Format a single example into instruction-following format.

    Args:
        example (dict): Dictionary with 'instruction', 'input' (optional), and 'output' keys

    Returns:
        dict: Dictionary with formatted 'text' field
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        formatted_text = (
            f"### Instrucción:\n{instruction}\n\n"
            f"### Entrada:\n{input_text}\n\n"
            f"### Respuesta:\n{output}"
        )
    else:
        formatted_text = (
            f"### Instrucción:\n{instruction}\n\n"
            f"### Respuesta:\n{output}"
        )

    return {"text": formatted_text}


def load_and_prepare_dataset(dataset_path, test_size=0.1, seed=42):
    """
    Load and prepare the dataset for training.

    Args:
        dataset_path (str): Path to JSONL dataset file
        test_size (float): Proportion of data to use for validation
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_dataset, eval_dataset)

    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If dataset format is invalid
    """
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at: {dataset_path}\n"
            f"Please ensure your data is placed in the correct location."
        )

    print(f"📂 Loading dataset from: {dataset_path}")

    # Load dataset
    try:
        if dataset_path.endswith(".jsonl") or dataset_path.endswith(".json"):
            ds = load_dataset("json", data_files=dataset_path)
        else:
            ds = load_dataset(dataset_path)
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")

    # Format dataset
    ds = ds.map(format_prompt, remove_columns=ds["train"].column_names)

    # Split into train/validation
    ds = ds["train"].train_test_split(test_size=test_size, seed=seed)
    train_data = ds["train"]
    eval_data = ds["test"]

    print(f"✅ Dataset loaded successfully")
    print(f"   Training samples:   {len(train_data):,}")
    print(f"   Validation samples: {len(eval_data):,}")

    return train_data, eval_data


def setup_model_and_tokenizer(model_name, max_seq_length=4096, load_in_4bit=True):
    """
    Load the base model and tokenizer with specified configuration.

    Args:
        model_name (str): HuggingFace model identifier
        max_seq_length (int): Maximum sequence length for training
        load_in_4bit (bool): Whether to load model in 4-bit precision

    Returns:
        tuple: (model, tokenizer)
    """
    print(f"🧠 Loading model: {model_name}")
    print(f"   Max sequence length: {max_seq_length}")
    print(f"   4-bit quantization: {load_in_4bit}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        max_seq_length=max_seq_length,
        device_map="auto",
    )

    print("✅ Model loaded successfully")
    return model, tokenizer


def setup_lora(model, r=16, lora_alpha=32, lora_dropout=0.05, target_modules=None):
    """
    Configure LoRA adapters for the model.

    Args:
        model: The base model
        r (int): LoRA rank
        lora_alpha (int): LoRA alpha parameter
        lora_dropout (float): Dropout rate for LoRA layers
        target_modules (list): List of module names to apply LoRA to

    Returns:
        model: Model with LoRA adapters attached
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"]

    print(f"🔧 Configuring LoRA adapters")
    print(f"   Rank (r):         {r}")
    print(f"   Alpha:            {lora_alpha}")
    print(f"   Dropout:          {lora_dropout}")
    print(f"   Target modules:   {len(target_modules)}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )

    print("✅ LoRA configuration complete")
    return model


def calculate_perplexity_from_logs(log_history):
    """
    Calculate and print perplexity for each evaluation step.

    Args:
        log_history (list): List of training log dictionaries
    """
    if not log_history:
        print("⚠️  No training logs available")
        return

    print("\n" + "="*70)
    print("📊 PERPLEXITY BY EVALUATION STEP")
    print("="*70)
    print(f"{'Step':>8} | {'Eval Loss':>12} | {'Perplexity':>12}")
    print("-" * 70)

    for log in log_history:
        if "eval_loss" in log:
            step = log.get("step", "?")
            eval_loss = log["eval_loss"]
            try:
                perplexity = math.exp(eval_loss)
            except OverflowError:
                perplexity = float("inf")
            print(f"{step:>8} | {eval_loss:>12.4f} | {perplexity:>12.2f}")

    print("="*70 + "\n")


def save_training_summary(output_dir, metrics, start_time, config):
    """
    Save a comprehensive training summary to JSON file.

    Args:
        output_dir (str): Directory to save the summary
        metrics (dict): Final evaluation metrics
        start_time (float): Training start timestamp
        config (dict): Training configuration
    """
    perplexity = None
    if "eval_loss" in metrics:
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")

    summary = {
        "model_name": config["MODEL_NAME"],
        "dataset_path": config["DATASET_PATH"],
        "training_config": {
            "epochs": config["EPOCHS"],
            "max_steps": config["MAX_STEPS"],
            "learning_rate": config["LEARNING_RATE"],
            "batch_size": config["PER_DEVICE_TRAIN_BATCH_SIZE"],
            "gradient_accumulation_steps": config["GRADIENT_ACCUMULATION_STEPS"],
            "lora_r": config["LORA_R"],
            "lora_alpha": config["LORA_ALPHA"],
            "lora_dropout": config["LORA_DROPOUT"],
        },
        "final_metrics": {
            "train_loss": metrics.get("train_loss"),
            "eval_loss": metrics.get("eval_loss"),
            "eval_perplexity": perplexity,
        },
        "runtime": {
            "total_seconds": round(time.time() - start_time, 2),
            "total_hours": round((time.time() - start_time) / 3600, 2),
        },
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📝 Training summary saved to: {summary_path}")


# ==============================================================================
# Main Training Function
# ==============================================================================
def main():
    """Main training pipeline."""
    print("\n" + "="*70)
    print("🚀 LLAMA 3.2 QLORA FINE-TUNING WITH UNSLOTH")
    print("="*70)

    start_time = time.time()

    try:
        # Load configuration
        cfg = load_configuration()

        # Print system info
        print_system_info()

        # Setup directories
        if USE_CONFIG_FILE:
            config.setup_directories()
        else:
            os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
            os.makedirs(os.path.dirname(cfg["DATASET_PATH"]), exist_ok=True)

        # Print configuration
        print("\n" + "="*70)
        print("⚙️  TRAINING CONFIGURATION")
        print("="*70)
        print(f"Model:            {cfg['MODEL_NAME']}")
        print(f"Dataset:          {cfg['DATASET_PATH']}")
        print(f"Output directory: {cfg['OUTPUT_DIR']}")
        print(f"Epochs:           {cfg['EPOCHS']}")
        print(f"Max steps:        {cfg['MAX_STEPS']} (0 = use full epochs)")
        print(f"Learning rate:    {cfg['LEARNING_RATE']}")
        print(f"Batch size:       {cfg['PER_DEVICE_TRAIN_BATCH_SIZE']}")
        print(f"Gradient accum:   {cfg['GRADIENT_ACCUMULATION_STEPS']}")
        print("="*70 + "\n")

        # Load dataset
        train_data, eval_data = load_and_prepare_dataset(
            cfg["DATASET_PATH"],
            test_size=cfg["TRAIN_TEST_SPLIT_RATIO"],
            seed=cfg["RANDOM_SEED"]
        )

        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(
            cfg["MODEL_NAME"],
            max_seq_length=cfg["MAX_SEQ_LENGTH"],
            load_in_4bit=cfg["LOAD_IN_4BIT"]
        )

        # Configure LoRA
        model = setup_lora(
            model,
            r=cfg["LORA_R"],
            lora_alpha=cfg["LORA_ALPHA"],
            lora_dropout=cfg["LORA_DROPOUT"],
            target_modules=cfg["LORA_TARGET_MODULES"]
        )

        # Prepare training arguments
        args_kwargs = {
            "output_dir": cfg["OUTPUT_DIR"],
            "per_device_train_batch_size": cfg["PER_DEVICE_TRAIN_BATCH_SIZE"],
            "gradient_accumulation_steps": cfg["GRADIENT_ACCUMULATION_STEPS"],
            "logging_steps": cfg["LOG_STEPS"],
            "evaluation_strategy": "steps",
            "eval_steps": cfg["EVAL_STEPS"],
            "save_strategy": cfg["SAVE_STRATEGY"],
            "save_total_limit": cfg["SAVE_TOTAL_LIMIT"],
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "num_train_epochs": cfg["EPOCHS"],
            "max_steps": cfg["MAX_STEPS"],
            "bf16": torch.cuda.is_bf16_supported(),
            "learning_rate": cfg["LEARNING_RATE"],
            "warmup_ratio": cfg["WARMUP_RATIO"],
            "lr_scheduler_type": cfg["LR_SCHEDULER_TYPE"],
            "logging_dir": os.path.join(cfg["OUTPUT_DIR"], "logs"),
            "report_to": "none",
        }

        # Initialize trainer
        print("\n🔧 Initializing trainer...")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args_kwargs=args_kwargs,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["EARLY_STOPPING_PATIENCE"])],
        )

        # Start training
        print("\n" + "="*70)
        print("🚦 STARTING TRAINING")
        print("="*70 + "\n")

        trainer.train()

        # Evaluation
        print("\n" + "="*70)
        print("📈 FINAL EVALUATION")
        print("="*70)
        metrics = trainer.evaluate()

        print("\n✅ Final Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

        if "eval_loss" in metrics:
            try:
                perplexity = math.exp(metrics["eval_loss"])
                print(f"\n🔢 Final Perplexity: {perplexity:.3f}")
            except OverflowError:
                print(f"\n⚠️  Perplexity overflow (loss too high)")

        # Print perplexity history
        calculate_perplexity_from_logs(trainer.state.log_history)

        # Save model
        print("\n💾 Saving model and adapters...")
        os.makedirs(cfg["OUTPUT_DIR"], exist_ok=True)
        trainer.save_model(cfg["OUTPUT_DIR"])
        print(f"✅ Model saved to: {cfg['OUTPUT_DIR']}")

        # Save training summary
        save_training_summary(cfg["OUTPUT_DIR"], metrics, start_time, cfg)

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"Total time: {total_time/3600:.2f} hours ({total_time:.0f} seconds)")
        print(f"Output directory: {cfg['OUTPUT_DIR']}")
        print("="*70 + "\n")

        return 0

    except FileNotFoundError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease ensure:")
        print("1. Your dataset exists at the specified path")
        print("2. The data directory structure is correct")
        return 1

    except Exception as e:
        print("\n" + "="*70)
        print("❌ ERROR DURING TRAINING")
        print("="*70)
        print(f"\nError message: {str(e)}\n")
        print("Full traceback:")
        traceback.print_exc()
        print("\n" + "="*70)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
