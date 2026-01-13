#!/bin/bash

# ==============================================================================
# LLaMA 3.2 QLoRA Fine-tuning - Setup Script
# ==============================================================================
# This script helps you set up the environment for training and inference
# Usage: bash setup.sh [options]
#
# Options:
#   --skip-venv        Skip virtual environment creation
#   --skip-install     Skip dependency installation
#   --quick            Quick setup (minimal checks)
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
SKIP_VENV=false
SKIP_INSTALL=false
QUICK_MODE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --skip-venv) SKIP_VENV=true ;;
        --skip-install) SKIP_INSTALL=true ;;
        --quick) QUICK_MODE=true ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# ==============================================================================
# Helper Functions
# ==============================================================================

print_header() {
    echo -e "\n${BLUE}======================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}======================================================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        print_success "$1 is installed"
        return 0
    else
        print_error "$1 is not installed"
        return 1
    fi
}

# ==============================================================================
# Main Setup
# ==============================================================================

print_header "🦙 LLaMA 3.2 QLoRA Fine-tuning - Setup"

# Check Python installation
print_header "Checking Prerequisites"

if ! check_command python3; then
    print_error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_info "Python version: $PYTHON_VERSION"

# Check if Python version is >= 3.8
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

print_success "Python version is compatible"

# Check for git
if ! $QUICK_MODE; then
    check_command git || print_warning "Git is not installed (optional)"
fi

# ==============================================================================
# Project Structure
# ==============================================================================

print_header "Creating Project Structure"

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

print_info "Project root: $PROJECT_ROOT"

# Create necessary directories
mkdir -p data
mkdir -p outputs
mkdir -p logs
mkdir -p models/hf_home
mkdir -p models/hf_cache

print_success "Directory structure created"

# ==============================================================================
# Virtual Environment
# ==============================================================================

if [ "$SKIP_VENV" = false ]; then
    print_header "Setting up Virtual Environment"

    VENV_DIR="$PROJECT_ROOT/venv"

    if [ -d "$VENV_DIR" ]; then
        print_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
            SKIP_INSTALL=true
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        print_info "Creating virtual environment at $VENV_DIR"
        python3 -m venv "$VENV_DIR"
        print_success "Virtual environment created"
    fi

    # Activate virtual environment
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    print_success "Virtual environment activated"
else
    print_info "Skipping virtual environment setup"
fi

# ==============================================================================
# Install Dependencies
# ==============================================================================

if [ "$SKIP_INSTALL" = false ]; then
    print_header "Installing Dependencies"

    # Upgrade pip
    print_info "Upgrading pip, setuptools, and wheel..."
    python -m pip install --upgrade pip setuptools wheel

    # Install PyTorch
    print_info "Installing PyTorch 2.7.1..."
    python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

    # Install other dependencies
    print_info "Installing Transformers, Datasets, Accelerate..."
    python -m pip install transformers datasets accelerate sentencepiece

    # Install Unsloth
    print_info "Installing Unsloth..."
    print_warning "If this fails, you may need to install from source"
    python -m pip install unsloth || print_warning "Unsloth installation failed (you may need to install manually)"

    # Install optional dependencies
    print_info "Installing optional dependencies..."
    python -m pip install ipython jupyter tensorboard

    print_success "All dependencies installed"
else
    print_info "Skipping dependency installation"
fi

# ==============================================================================
# Configuration
# ==============================================================================

print_header "Configuring Project"

# Check if config.py exists and test it
if [ -f "config.py" ]; then
    print_info "Testing configuration..."
    python config.py
    print_success "Configuration is valid"
else
    print_error "config.py not found!"
fi

# ==============================================================================
# Verify Installation
# ==============================================================================

print_header "Verifying Installation"

# Test PyTorch and CUDA
print_info "Testing PyTorch and CUDA availability..."
python - <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  No GPU detected. Training will be slow!")
EOF

# Check if dataset exists
print_info "Checking for dataset..."
if [ -f "data/base.jsonl" ]; then
    print_success "Dataset found at data/base.jsonl"
else
    print_warning "No dataset found at data/base.jsonl"
    print_info "Please place your training data in data/base.jsonl"
    print_info "Format: Each line should be a JSON object with 'instruction', 'input' (optional), and 'output' fields"
fi

# ==============================================================================
# Final Instructions
# ==============================================================================

print_header "✅ Setup Complete!"

echo -e "${GREEN}The project is ready to use!${NC}\n"

if [ "$SKIP_VENV" = false ]; then
    echo "To activate the virtual environment, run:"
    echo -e "${BLUE}  source venv/bin/activate${NC}\n"
fi

echo "Next steps:"
echo ""
echo "1. Place your training data in: ${BLUE}data/base.jsonl${NC}"
echo ""
echo "2. Review and modify configuration in: ${BLUE}config.py${NC}"
echo ""
echo "3. To train locally:"
echo -e "   ${BLUE}python scripts/train_llama32_gpu.py${NC}"
echo ""
echo "4. To train on HPC cluster with SLURM:"
echo -e "   ${BLUE}sbatch scripts/train_block_full_gpu.sbatch${NC}"
echo ""
echo "5. To run inference after training:"
echo -e "   ${BLUE}python scripts/infer_llama.py --model_path outputs/llama32_qlora --prompt \"Your prompt here\"${NC}"
echo ""
echo "6. For interactive inference:"
echo -e "   ${BLUE}python scripts/infer_llama.py --model_path outputs/llama32_qlora --interactive${NC}"
echo ""
echo "For more information, see:"
echo "  - ${BLUE}README.md${NC} - Project overview"
echo "  - ${BLUE}INSTALL.md${NC} - Detailed installation guide"
echo ""

print_header "🎉 Happy Training!"
