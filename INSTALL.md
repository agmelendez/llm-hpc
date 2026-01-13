# Installation Guide

Complete installation guide for the LLaMA 3.2 QLoRA Fine-tuning project.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Setup (Recommended)](#quick-setup-recommended)
- [Manual Setup](#manual-setup)
- [HPC/Cluster Setup](#hpccluster-setup)
- [Verifying Installation](#verifying-installation)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended: A100 80GB)
  - Minimum: 12GB VRAM for 4-bit quantization
  - Recommended: 40GB+ VRAM for stable training
- **Storage**: At least 20GB free space for models and outputs
- **Memory**: 16GB+ RAM recommended

### Software Requirements

- Git (for cloning the repository)
- CUDA 11.8 or higher (for GPU support)
- Python development headers (python3-dev)

---

## Quick Setup (Recommended)

The easiest way to set up the project is using the automated setup script:

```bash
# 1. Clone the repository (if not already done)
git clone <repository-url>
cd llama32_qlora

# 2. Run the setup script
bash setup.sh
```

The setup script will:
- ✅ Check Python version compatibility
- ✅ Create necessary directories
- ✅ Set up a virtual environment
- ✅ Install all dependencies
- ✅ Verify the installation

### Setup Options

```bash
# Skip virtual environment creation
bash setup.sh --skip-venv

# Skip dependency installation (if already installed)
bash setup.sh --skip-install

# Quick mode (minimal checks)
bash setup.sh --quick
```

---

## Manual Setup

If you prefer to set up manually or the automated script doesn't work:

### Step 1: Create Virtual Environment

```bash
# Navigate to project directory
cd llama32_qlora

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows
```

### Step 2: Upgrade pip

```bash
python -m pip install --upgrade pip setuptools wheel
```

### Step 3: Install PyTorch

Install PyTorch with CUDA support:

```bash
# For CUDA 11.8
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# For CUDA 12.1+
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# OR install packages individually
pip install transformers datasets accelerate sentencepiece
```

### Step 5: Install Unsloth

```bash
# Try pip installation first
pip install unsloth

# If that fails, install from source
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 6: Create Project Structure

```bash
# Create necessary directories
mkdir -p data outputs logs models/hf_home models/hf_cache
```

### Step 7: Test Configuration

```bash
# Run config.py to verify setup
python config.py
```

---

## HPC/Cluster Setup

If you're using an HPC cluster with SLURM:

### Step 1: Configure User Settings

Edit `scripts/train_block_full_gpu.sbatch`:

```bash
# Open in editor
nano scripts/train_block_full_gpu.sbatch

# Change these lines:
USER_HOME="${HOME}"  # Or set to your home directory
PROJECT_ROOT="${USER_HOME}/llama32_qlora"

# Configure email notifications (optional)
# Uncomment and edit:
# #SBATCH --mail-user=your.email@example.com
# #SBATCH --mail-type=END,FAIL
```

### Step 2: Set Up Environment

Choose one of the following:

#### Option A: Virtual Environment

```bash
# On login node or compute node
cd ~/llama32_qlora
bash setup.sh
```

#### Option B: Conda Environment

```bash
# Create conda environment
conda create -n llama_finetune python=3.10
conda activate llama_finetune

# Install dependencies
bash setup.sh --skip-venv
```

### Step 3: Configure SLURM Settings

Review and adjust SLURM settings in `train_block_full_gpu.sbatch`:

```bash
#SBATCH --partition=gpu          # Your cluster's GPU partition name
#SBATCH --gres=gpu:1             # Number of GPUs (1 recommended)
#SBATCH --cpus-per-task=8        # CPU cores (adjust based on cluster)
#SBATCH --mem=48G                # Memory (adjust based on cluster)
#SBATCH --time=10:00:00          # Time limit (adjust as needed)
```

### Step 4: Verify GPU Access

```bash
# Request interactive GPU session
srun --partition=gpu --gres=gpu:1 --pty bash

# Check GPU
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Exit interactive session
exit
```

---

## Verifying Installation

### Check Python and PyTorch

```bash
python - << 'EOF'
import sys
import torch
import transformers

print("="*70)
print("INSTALLATION VERIFICATION")
print("="*70)
print(f"Python version:        {sys.version.split()[0]}")
print(f"PyTorch version:       {torch.__version__}")
print(f"Transformers version:  {transformers.__version__}")
print(f"CUDA available:        {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:          {torch.version.cuda}")
    print(f"GPU device:            {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  WARNING: No GPU detected!")

print("="*70)
EOF
```

### Verify Unsloth Installation

```bash
python -c "from unsloth import FastLanguageModel; print('✅ Unsloth installed successfully')"
```

### Test Training Script

```bash
# This will show configuration without training
python scripts/train_llama32_gpu.py --help 2>/dev/null || python config.py
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Available

**Problem**: `torch.cuda.is_available()` returns `False`

**Solutions**:
- Verify NVIDIA driver: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Check CUDA installation: `nvcc --version`

```bash
# Reinstall PyTorch with correct CUDA
pip uninstall torch torchvision torchaudio
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
```

#### 2. Unsloth Installation Fails

**Problem**: `pip install unsloth` fails

**Solutions**:

```bash
# Try installing from source
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or install individual components
pip install bitsandbytes peft
```

#### 3. Out of Memory Errors

**Problem**: CUDA out of memory during training

**Solutions**:
- Reduce batch size in `config.py`:
  ```python
  PER_DEVICE_TRAIN_BATCH_SIZE = 1  # Reduce from 2
  ```
- Increase gradient accumulation:
  ```python
  GRADIENT_ACCUMULATION_STEPS = 8  # Increase from 4
  ```
- Reduce sequence length:
  ```python
  MAX_SEQ_LENGTH = 2048  # Reduce from 4096
  ```

#### 4. Dataset Not Found

**Problem**: `FileNotFoundError: Dataset not found`

**Solutions**:
- Ensure dataset is at `data/base.jsonl`
- Check file format (JSONL with correct fields)
- Verify path in `config.py` or environment variable

#### 5. Permission Denied on HPC

**Problem**: Cannot write to output directories

**Solutions**:
```bash
# Create directories with correct permissions
mkdir -p outputs logs
chmod 755 outputs logs

# Or change output location in config.py
```

#### 6. Virtual Environment Issues

**Problem**: Cannot activate virtual environment or import errors

**Solutions**:
```bash
# Remove and recreate virtual environment
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### 7. HuggingFace Authentication

**Problem**: Cannot download model (private models)

**Solutions**:
```bash
# Login to HuggingFace
pip install huggingface-hub
huggingface-cli login

# Enter your HuggingFace token when prompted
```

---

## Advanced Configuration

### Using Different Models

Edit `config.py`:

```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Use 3B model instead
```

### Custom Dataset Location

```bash
# Set environment variable
export DATA="/path/to/your/dataset.jsonl"

# Or edit config.py
```

### Adjusting Training Parameters

Edit `config.py` to modify:
- Learning rate
- Number of epochs
- Batch size
- LoRA parameters
- Evaluation frequency

See `config.py` for detailed parameter descriptions.

---

## Next Steps

After successful installation:

1. **Prepare your dataset**: See [README.md](README.md#data-format) for format details
2. **Configure training**: Edit `config.py` with your preferences
3. **Run training**: See [README.md](README.md#training) for training instructions
4. **Run inference**: See [README.md](README.md#inference) for inference examples

---

## Getting Help

If you encounter issues not covered here:

1. Check the main [README.md](README.md) for additional information
2. Review error messages carefully
3. Check CUDA and PyTorch compatibility
4. Verify file permissions and paths
5. Ensure sufficient disk space and memory

---

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.10+ |
| GPU VRAM | 12GB | 40GB+ (A100) |
| System RAM | 16GB | 32GB+ |
| Storage | 20GB | 50GB+ |
| CUDA | 11.8 | 12.1+ |

---

**Note**: Installation may take 15-30 minutes depending on internet speed and system specifications.
