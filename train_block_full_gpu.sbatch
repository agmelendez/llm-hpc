#!/bin/bash
#SBATCH --job-name=llama32_qlora_full
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00

# ==============================================================================
# LLaMA 3.2 QLoRA Fine-tuning - SLURM Batch Script
# ==============================================================================
# This script runs the training job on an HPC cluster using SLURM
#
# Usage:
#   1. Edit the USER_HOME variable below to match your home directory
#   2. Edit the SLURM email notification settings if desired
#   3. Submit with: sbatch train_block_full_gpu.sbatch
#
# Configuration:
#   - Edit config.py for training parameters
#   - Or override with environment variables (see below)
# ==============================================================================

# ==============================================================================
# IMPORTANT: User-specific Configuration
# ==============================================================================
# Change this to your home directory or username
USER_HOME="${HOME}"  # Will use your $HOME environment variable
# Or set explicitly: USER_HOME="/home/your_username"

# Project root - change if your project is in a different location
PROJECT_ROOT="${USER_HOME}/llama32_qlora"

# Virtual environment path
VENV_PATH="${PROJECT_ROOT}/venv"
# Or if using conda: CONDA_ENV="llama_finetune"

# Email notifications (optional) - uncomment and edit
# #SBATCH --mail-user=your.email@example.com
# #SBATCH --mail-type=END,FAIL

# ==============================================================================
# Job Information
# ==============================================================================
echo "========================================================================"
echo "🚀 LLaMA 3.2 QLoRA Fine-tuning with Unsloth"
echo "========================================================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "Started at:    $(date)"
echo "Project root:  $PROJECT_ROOT"
echo "========================================================================"

# ==============================================================================
# Environment Setup
# ==============================================================================
echo ""
echo "Setting up environment..."

# Clean module environment (if using modules)
if command -v module &> /dev/null; then
    module purge
    echo "✅ Modules purged"
fi

# Activate virtual environment or conda
if [ -d "$VENV_PATH" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
elif command -v conda &> /dev/null; then
    echo "Activating conda environment: $CONDA_ENV"
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    echo "✅ Conda environment activated"
else
    echo "⚠️  Warning: No virtual environment found"
    echo "   Using system Python: $(which python3)"
fi

# Set Unsloth environment variable (disable RL patch if needed)
export UNSLOTH_DISABLE_RL=1
echo "✅ UNSLOTH_DISABLE_RL=1 (RL patch disabled)"

# Set HuggingFace cache directories
export HF_HOME="${PROJECT_ROOT}/models/hf_home"
export TRANSFORMERS_CACHE="${PROJECT_ROOT}/models/hf_cache"
echo "✅ HuggingFace cache: $HF_HOME"

# Create output directories if they don't exist
mkdir -p "${PROJECT_ROOT}/logs"
mkdir -p "${PROJECT_ROOT}/outputs"
mkdir -p "$HF_HOME"
mkdir -p "$TRANSFORMERS_CACHE"

# ==============================================================================
# Training Configuration
# ==============================================================================
# These can be overridden by editing config.py or passing environment variables
# when submitting the job
echo ""
echo "Training configuration:"

# Model and data paths
export MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
export DATA="${DATA:-${PROJECT_ROOT}/data/base.jsonl}"
export OUT="${OUT:-${PROJECT_ROOT}/outputs/llama32_qlora}"

# Training parameters
export MAX_STEPS="${MAX_STEPS:-0}"      # 0 = use full epochs
export EPOCHS="${EPOCHS:-60}"            # Number of epochs
export EVAL_STEPS="${EVAL_STEPS:-200}"  # Evaluate every N steps
export LOG_STEPS="${LOG_STEPS:-20}"     # Log every N steps

echo "  Model:       $MODEL"
echo "  Dataset:     $DATA"
echo "  Output dir:  $OUT"
echo "  Epochs:      $EPOCHS"
echo "  Max steps:   $MAX_STEPS (0 = full epochs)"
echo "  Eval steps:  $EVAL_STEPS"
echo "  Log steps:   $LOG_STEPS"

# ==============================================================================
# System Information
# ==============================================================================
echo ""
echo "========================================================================"
echo "System Information"
echo "========================================================================"
echo "Hostname:      $(hostname)"
echo "Python:        $(python --version 2>&1)"
echo "Working dir:   $(pwd)"

# GPU information
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "⚠️  nvidia-smi not available"
fi

echo "========================================================================"

# ==============================================================================
# Run Training
# ==============================================================================
echo ""
echo "========================================================================"
echo "🚦 Starting Training"
echo "========================================================================"
echo ""

cd "$PROJECT_ROOT"

# Run with Python unbuffered output for real-time logging
python -u scripts/train_llama32_gpu.py

EXIT_CODE=$?

# ==============================================================================
# Job Completion
# ==============================================================================
echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi
echo "========================================================================"
echo "Finished at:   $(date)"
echo "Job ID:        $SLURM_JOB_ID"
echo "Output saved:  $OUT"
echo "========================================================================"

exit $EXIT_CODE
