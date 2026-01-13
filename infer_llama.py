"""
LLaMA 3.2 Inference Script
===========================

This script performs inference using a fine-tuned LLaMA 3.2 model.
It loads the trained model and generates responses to user prompts.

Features:
    - Automatic GPU/CPU detection
    - Configurable generation parameters
    - Support for both base models and fine-tuned adapters
    - Interactive and batch inference modes

Usage:
    # Single prompt inference
    python infer_llama.py --model_path ./outputs/llama32_qlora --prompt "¿Qué es Python?"

    # With custom generation parameters
    python infer_llama.py \
        --model_path ./outputs/llama32_qlora \
        --prompt "Explica qué es machine learning" \
        --max_tokens 300 \
        --temperature 0.8 \
        --top_p 0.95

    # Interactive mode
    python infer_llama.py --model_path ./outputs/llama32_qlora --interactive

Requirements:
    - Python 3.8+
    - PyTorch 2.7.1+
    - Transformers
    - Trained model checkpoint

Author: Alison Lobo Salas
Organization: Universidad de Costa Rica (UCR)
"""

import argparse
import sys
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run inference with a fine-tuned LLaMA 3.2 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python infer_llama.py --model_path ./outputs/llama32_qlora --prompt "¿Qué es Python?"

  # Custom temperature and max tokens
  python infer_llama.py --model_path ./outputs/llama32_qlora \\
      --prompt "Explica machine learning" --temperature 0.9 --max_tokens 500

  # Interactive mode
  python infer_llama.py --model_path ./outputs/llama32_qlora --interactive
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model directory (containing adapter files)"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Input prompt for generating response (required unless --interactive is set)"
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode (ask for prompts continuously)"
    )

    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate (default: 200)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (0.0 = deterministic, higher = more random) (default: 0.7)"
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p value (default: 0.9)"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling value (default: 50)"
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty to avoid repeating tokens (default: 1.1)"
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Force CPU usage even if GPU is available"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.prompt:
        parser.error("Either --prompt or --interactive must be specified")

    return args


def check_model_path(model_path):
    """
    Verify that the model path exists and contains necessary files.

    Args:
        model_path (str): Path to model directory

    Raises:
        FileNotFoundError: If model path doesn't exist
        ValueError: If model directory is missing required files
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {model_path}\n"
            f"Please ensure you have trained a model or provide the correct path."
        )

    if not model_path.is_dir():
        raise ValueError(f"Model path must be a directory: {model_path}")

    # Check for essential files
    essential_files = ["adapter_config.json", "adapter_model.safetensors"]
    missing_files = [f for f in essential_files if not (model_path / f).exists()]

    if missing_files:
        print(f"⚠️  Warning: Some adapter files are missing: {missing_files}")
        print("   This might be a base model directory instead of fine-tuned adapters.")


def print_system_info(use_gpu):
    """
    Print system information.

    Args:
        use_gpu (bool): Whether GPU will be used
    """
    print("\n" + "="*70)
    print("🖥️  SYSTEM INFORMATION")
    print("="*70)
    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")

    if torch.cuda.is_available() and use_gpu:
        print(f"CUDA version:     {torch.version.cuda}")
        print(f"GPU device:       {torch.cuda.get_device_name(0)}")
        print(f"Using device:     GPU")
    else:
        print(f"Using device:     CPU")
        if not use_gpu:
            print("                  (GPU disabled via --no_cuda flag)")

    print("="*70 + "\n")


def load_model(model_path, use_gpu=True):
    """
    Load the tokenizer and model from the specified path.

    Args:
        model_path (str): Path to model directory
        use_gpu (bool): Whether to use GPU if available

    Returns:
        tuple: (tokenizer, model)
    """
    print(f"🔄 Loading model from: {model_path}")

    # Determine device and dtype
    if use_gpu and torch.cuda.is_available():
        device_map = "auto"
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"   Using dtype: {torch_dtype}")
    else:
        device_map = None
        torch_dtype = torch.float32
        print("   Using CPU with float32")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Tokenizer loaded")
    except Exception as e:
        print(f"⚠️  Warning: Could not load tokenizer from model path: {e}")
        print("   Attempting to load from base model...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        print("✅ Model loaded successfully\n")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")

    return tokenizer, model


def create_generator(model, tokenizer, args):
    """
    Create a text generation pipeline.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        args: Command line arguments

    Returns:
        pipeline: Configured text generation pipeline
    """
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        do_sample=True if args.temperature > 0 else False,
    )

    return generator


def generate_response(generator, prompt, show_prompt=True):
    """
    Generate a response for the given prompt.

    Args:
        generator: Text generation pipeline
        prompt (str): Input prompt
        show_prompt (bool): Whether to display the prompt before generation

    Returns:
        str: Generated text
    """
    if show_prompt:
        print("\n" + "="*70)
        print("🧠 PROMPT")
        print("="*70)
        print(prompt)
        print("="*70)

    print("\n💬 Generating response...\n")

    try:
        outputs = generator(prompt)
        generated_text = outputs[0]["generated_text"]
        return generated_text
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        return None


def interactive_mode(generator):
    """
    Run inference in interactive mode, continuously asking for prompts.

    Args:
        generator: Text generation pipeline
    """
    print("\n" + "="*70)
    print("🎮 INTERACTIVE MODE")
    print("="*70)
    print("Enter prompts to get responses. Type 'quit', 'exit', or 'q' to stop.")
    print("="*70 + "\n")

    while True:
        try:
            prompt = input("👤 Prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("\n👋 Goodbye!")
                break

            if not prompt:
                print("⚠️  Empty prompt. Please enter a valid prompt.\n")
                continue

            response = generate_response(generator, prompt, show_prompt=False)

            if response:
                print("\n" + "="*70)
                print("🤖 RESPONSE")
                print("="*70)
                print(response)
                print("="*70 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except EOFError:
            print("\n\n👋 End of input. Goodbye!")
            break


def main():
    """Main inference pipeline."""
    print("\n" + "="*70)
    print("🦙 LLAMA 3.2 INFERENCE")
    print("="*70)

    # Parse arguments
    args = parse_arguments()

    # Check model path
    try:
        check_model_path(args.model_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {str(e)}")
        return 1

    # Print system info
    use_gpu = not args.no_cuda
    print_system_info(use_gpu)

    # Print generation parameters
    print("⚙️  GENERATION PARAMETERS")
    print("="*70)
    print(f"Max new tokens:      {args.max_tokens}")
    print(f"Temperature:         {args.temperature}")
    print(f"Top-p:               {args.top_p}")
    print(f"Top-k:               {args.top_k}")
    print(f"Repetition penalty:  {args.repetition_penalty}")
    print("="*70 + "\n")

    # Load model and tokenizer
    try:
        tokenizer, model = load_model(args.model_path, use_gpu=use_gpu)
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return 1

    # Create generator
    generator = create_generator(model, tokenizer, args)

    # Run inference
    try:
        if args.interactive:
            interactive_mode(generator)
        else:
            response = generate_response(generator, args.prompt)
            if response:
                print("\n" + "="*70)
                print("🤖 GENERATED RESPONSE")
                print("="*70)
                print(response)
                print("="*70 + "\n")
    except Exception as e:
        print(f"\n❌ Error during inference: {str(e)}")
        return 1

    print("✅ Inference completed successfully\n")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
