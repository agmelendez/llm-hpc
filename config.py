"""
Configuration file for LLaMA 3.2 QLoRA Fine-tuning Project
===========================================================

This file centralizes all configuration parameters for the project.
Modify these values according to your environment and requirements.
"""

import os
from pathlib import Path

# ==============================================================================
# Project Paths
# ==============================================================================
# Base directory - automatically detected or can be set manually
PROJECT_ROOT = Path(__file__).parent.absolute()

# User home directory - change this to your username or use environment variable
USER_HOME = os.getenv("HOME", "/home/user")

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
DATASET_PATH = DATA_DIR / "base.jsonl"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "llama32_qlora"

# Logs directory
LOGS_DIR = PROJECT_ROOT / "logs"

# HuggingFace cache directories
HF_HOME = PROJECT_ROOT / "models" / "hf_home"
TRANSFORMERS_CACHE = PROJECT_ROOT / "models" / "hf_cache"

# ==============================================================================
# Model Configuration
# ==============================================================================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
MAX_SEQ_LENGTH = 4096
LOAD_IN_4BIT = True

# ==============================================================================
# LoRA Configuration
# ==============================================================================
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05  # Set to 0.0 for maximum performance
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ==============================================================================
# Training Configuration
# ==============================================================================
# Batch size and gradient accumulation
PER_DEVICE_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

# Training duration
NUM_TRAIN_EPOCHS = 60
MAX_STEPS = 0  # 0 = no limit, use full epochs

# Learning rate and scheduler
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.03
LR_SCHEDULER_TYPE = "cosine"

# Evaluation and logging
EVAL_STEPS = 200
LOGGING_STEPS = 20
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 3

# Early stopping
EARLY_STOPPING_PATIENCE = 3

# Data split
TRAIN_TEST_SPLIT_RATIO = 0.1
RANDOM_SEED = 42

# ==============================================================================
# HPC/SLURM Configuration (for batch jobs)
# ==============================================================================
SLURM_JOB_NAME = "llama32_qlora"
SLURM_PARTITION = "gpu"
SLURM_NODES = 1
SLURM_GPUS = 1
SLURM_CPUS_PER_TASK = 8
SLURM_MEMORY = "48G"
SLURM_TIME_LIMIT = "10:00:00"

# User email for SLURM notifications (change this!)
SLURM_EMAIL = os.getenv("USER_EMAIL", "your.email@example.com")

# ==============================================================================
# Environment Variables
# ==============================================================================
# Unsloth configuration
UNSLOTH_DISABLE_RL = "1"

# ==============================================================================
# Inference Configuration
# ==============================================================================
INFERENCE_MAX_NEW_TOKENS = 200
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_P = 0.9

# ==============================================================================
# Helper Functions
# ==============================================================================
def setup_directories():
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR,
        OUTPUT_DIR,
        MODEL_OUTPUT_DIR,
        LOGS_DIR,
        HF_HOME,
        TRANSFORMERS_CACHE,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    print(f"✅ Directories created/verified: {len(directories)} directories")


def get_env_config():
    """
    Get configuration from environment variables.
    Useful for overriding defaults without modifying this file.
    """
    return {
        "MODEL_NAME": os.getenv("MODEL", MODEL_NAME),
        "DATASET_PATH": os.getenv("DATA", str(DATASET_PATH)),
        "OUTPUT_DIR": os.getenv("OUT", str(MODEL_OUTPUT_DIR)),
        "MAX_STEPS": int(os.getenv("MAX_STEPS", str(MAX_STEPS))),
        "EPOCHS": int(os.getenv("EPOCHS", str(NUM_TRAIN_EPOCHS))),
        "EVAL_STEPS": int(os.getenv("EVAL_STEPS", str(EVAL_STEPS))),
        "LOG_STEPS": int(os.getenv("LOG_STEPS", str(LOGGING_STEPS))),
    }


def print_config():
    """Print current configuration for verification."""
    print("\n" + "="*70)
    print("📋 CONFIGURATION")
    print("="*70)
    print(f"Project Root:     {PROJECT_ROOT}")
    print(f"Model:            {MODEL_NAME}")
    print(f"Dataset:          {DATASET_PATH}")
    print(f"Output Dir:       {MODEL_OUTPUT_DIR}")
    print(f"Epochs:           {NUM_TRAIN_EPOCHS}")
    print(f"Learning Rate:    {LEARNING_RATE}")
    print(f"LoRA r/alpha:     {LORA_R}/{LORA_ALPHA}")
    print(f"Batch Size:       {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"Grad Accum:       {GRADIENT_ACCUMULATION_STEPS}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # When run directly, show configuration and create directories
    print_config()
    setup_directories()
    print("\n✅ Configuration loaded successfully!")
