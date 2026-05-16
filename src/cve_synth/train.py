"""
Fine-tune Gemma-2-9b-it on synthetic CVE analysis dataset using Unsloth and LoRA.

Optimized for 16GB VRAM (e.g., Kaggle P100, RTX 2070, etc.) with 4-bit quantization,
gradient checkpointing, and LoRA adapters.

Usage:
    python -m cve_synth.train \
        --dataset-path data/dataset.jsonl \
        --output-dir ./output \
        --num-train-epochs 3 \
        --batch-size 2 \
        --gradient-accumulation-steps 4
"""

from __future__ import annotations

# CRITICAL: Import Unsloth FIRST before torch, transformers, or trl
# to allow Unsloth to apply monkey-patches to attention layers.
from unsloth import FastLanguageModel, is_bfloat16_supported

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    TrainingArguments,
    TextIteratorStreamer,
    get_linear_schedule_with_warmup,
)
from trl import SFTTrainer


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "unsloth/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 4096
GRADIENT_CHECKPOINTING_TECHNIQUE = "unsloth"

# LoRA Configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0  # Must be 0.0 for Unsloth fast-path CUDA kernels; non-zero forces fallback to slower layers
LORA_BIAS = "none"
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ============================================================================
# Dataset Loading and Formatting
# ============================================================================


def load_jsonl_dataset(jsonl_path: str | Path) -> Dataset:
    """Load JSONL dataset from disk."""
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                data.append(record)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed JSON line: {e}")
                continue
    
    print(f"[INFO] Loaded {len(data)} records from {jsonl_path}")
    return Dataset.from_dict({"records": data})


def formatting_prompts_func(examples: dict[str, Any]) -> dict[str, list[str]]:
    """
    Format dataset records into Qwen chat templates with Chain-of-Thought structure.
    
    Expected input structure:
    {
      "analysis": {
        "vulnerability_summary": "...",
        "root_cause": "...",
        "tags": ["#LogicError", ...],
        "reasoning_chain": ["step 1", "step 2", ...],
        "assembly_fix": "..."
      },
      "source": {
        "title": "...",
        "cve_id": "CVE-XXXX-XXXXX",
        ...
      }
    }
    """
    texts = []
    
    # The dataset produced by cve_synth is flat (top-level fields), not nested under `analysis`.
    # Support both flat and nested schemas for backward compatibility.
    for record in examples.get("records", []):
        # Prefer flat fields if present, otherwise fall back to nested `analysis` object
        if "vulnerability_summary" in record:
            vuln_summary = record.get("vulnerability_summary", "")
            root_cause = record.get("root_cause", "")
            tags = record.get("tags", []) or []
            reasoning_chain = record.get("reasoning_chain", []) or []
            assembly_fix = record.get("assembly_fix", "")
            cve_id = record.get("cve_id", record.get("source_id", "UNKNOWN"))
        else:
            analysis = record.get("analysis", {})
            vuln_summary = analysis.get("vulnerability_summary", "")
            root_cause = analysis.get("root_cause", "")
            tags = analysis.get("tags", []) or []
            reasoning_chain = analysis.get("reasoning_chain", []) or []
            assembly_fix = analysis.get("assembly_fix", "")
            source = record.get("source", {})
            cve_id = source.get("cve_id", record.get("source_id", "UNKNOWN"))

        # Build prompt using Qwen's chat template: <|im_start|>...<|im_end|>
        prompt = (
            f"<|im_start|>user\n"
            f"Analyze the following security vulnerability and provide a detailed reasoning process inside <thought> tags, then the final analysis.\n\n"
            f"CVE ID: {cve_id}\n"
            f"Summary: {vuln_summary}\n"
            f"Root Cause Indication: {root_cause}\n"
            f"<|im_end|>\n"
        )

        # Convert reasoning_chain list to bullet points inside <thought>
        reasoning_text = "\n".join([f"- {step}" for step in reasoning_chain]) if reasoning_chain else "-"
        tags_text = ", ".join(tags) if tags else "#Unknown"

        response = (
            f"<|im_start|>assistant\n"
            f"<thought>\n{reasoning_text}\n</thought>\n\n"
            f"**Vulnerability Analysis:**\n"
            f"- Classification Tags: {tags_text}\n"
            f"- Assembly-level Fix:\n{assembly_fix}\n"
            f"<|im_end|>"
        )

        texts.append(prompt + response)
    
    return {"text": texts}


# ============================================================================
# Model Initialization
# ============================================================================


def initialize_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    max_seq_length: int = MAX_SEQ_LENGTH,
    load_in_4bit: bool = True,
) -> tuple:
    """
    Initialize Gemma-2-9b-it model with Unsloth optimizations and LoRA.
    
    Returns:
        (model, tokenizer) tuple
    """
    print(f"[INFO] Loading model: {model_name}")
    print(f"[INFO] Max sequence length: {max_seq_length}")
    print(f"[INFO] 4-bit quantization: {load_in_4bit}")
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=torch.float16 if not is_bfloat16_supported() else torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )
    
    # Add LoRA adapters
    print(f"[INFO] Applying LoRA with rank={LORA_R}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING_TECHNIQUE,
        target_modules=LORA_TARGET_MODULES,
    )
    
    # Ensure tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# ============================================================================
# Training
# ============================================================================


def train(
    dataset_path: str | Path,
    output_dir: str | Path,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    warmup_ratio: float = 0.03,
    save_steps: int = 500,
    logging_steps: int = 10,
    save_total_limit: int = 3,
) -> None:
    """
    Fine-tune the model on the synthetic CVE dataset.
    
    Args:
        dataset_path: Path to dataset.jsonl
        output_dir: Output directory for checkpoints and final model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per GPU
        gradient_accumulation_steps: Number of accumulation steps
        learning_rate: Learning rate
        warmup_ratio: Warmup ratio (0.0 to 1.0)
        save_steps: Save checkpoint every N steps
        logging_steps: Log metrics every N steps
        save_total_limit: Keep only the last N checkpoints
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("FINE-TUNING GEMMA-2-9B-IT ON SYNTHETIC CVE DATASET")
    print("=" * 80)
    
    # Load dataset
    print(f"\n[STEP 1] Loading dataset from {dataset_path}")
    dataset = load_jsonl_dataset(dataset_path)
    print(f"[INFO] Dataset has {len(dataset)} samples")
    
    # Format dataset
    print(f"\n[STEP 2] Formatting dataset with Chain-of-Thought prompts")
    formatted_dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
        batch_size=64,
        remove_columns=dataset.column_names,
        desc="Formatting prompts",
    )
    print(f"[INFO] Dataset formatted: {len(formatted_dataset)} training examples")
    
    # Initialize model and tokenizer
    print(f"\n[STEP 3] Initializing model and applying LoRA")
    model, tokenizer = initialize_model_and_tokenizer()
    print(f"[INFO] Model initialized with {LORA_R}×{LORA_ALPHA} LoRA adapters")
    print(f"[INFO] Trainable parameters (LoRA): ~15-20M / Total: ~9B")
    
    # Set up training arguments
    print(f"\n[STEP 4] Configuring training parameters")
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    print(f"[INFO] Effective batch size: {effective_batch_size}")
    print(f"       (per_device={per_device_train_batch_size}, accumulation={gradient_accumulation_steps})")
    
    # Calculate warmup steps (TRL v0.8+: warmup_ratio deprecated, use warmup_steps)
    total_steps = (len(formatted_dataset) * num_train_epochs) // effective_batch_size
    warmup_steps = int(total_steps * 0.03)  # 3% warmup
    print(f"[INFO] Total training steps: {total_steps}, Warmup steps: {warmup_steps}")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,  # TRL v0.8+ uses warmup_steps instead of warmup_ratio
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=logging_steps,
        optim="paged_adamw_8bit",
        seed=42,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        # logging_dir removed to avoid deprecation warning; TensorBoard logs handled via env var or default
        logging_first_step=True,
        report_to=["tensorboard"],
        push_to_hub=False,
    )
    
    # Initialize trainer
    print(f"\n[STEP 5] Initializing SFT trainer")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,  # TRL v0.8+: use processing_class instead of deprecated tokenizer argument
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=training_args,
    )
    
    # Train
    print(f"\n[STEP 6] Starting training...")
    print(f"         Model: {MODEL_NAME}")
    print(f"         Epochs: {num_train_epochs}")
    print(f"         Effective batch size: {effective_batch_size}")
    print(f"         Learning rate: {learning_rate}")
    print(f"         Output directory: {output_dir}")
    print(f"\n" + "=" * 80)
    print(f"[READY] Training {MODEL_NAME} with {len(formatted_dataset)} samples.")
    
    # === UNSLOTH LOSS COMPATIBILITY PATCH ===
    # Save the original method
    _original_unsloth_step = trainer.training_step

    def _patched_training_step(model, inputs, *args, **kwargs):
        # Call the original step
        loss = _original_unsloth_step(model, inputs, *args, **kwargs)
        # If loss is a scalar (int/float), wrap it as a differentiable tensor
        if isinstance(loss, (int, float)):
            loss = torch.tensor(loss, requires_grad=True, device=model.device)
        return loss

    # Apply the patch to the trainer instance
    import types
    trainer.training_step = types.MethodType(lambda self, model, inputs, *a, **kw: _patched_training_step(model, inputs, *a, **kw), trainer)
    print("[PATCH] Applied loss scalar compatibility fix.")
    # === END PATCH ===
    
    trainer.train()
    
    # Save the final model
    print(f"\n[STEP 7] Saving final model")
    model.save_pretrained(str(output_dir / "final_model"))
    tokenizer.save_pretrained(str(output_dir / "final_model"))
    print(f"[INFO] Final model saved to {output_dir / 'final_model'}")
    
    # Save training metadata
    metadata = {
        "model_name": MODEL_NAME,
        "base_model": "gemma-2-9b-it",
        "max_seq_length": MAX_SEQ_LENGTH,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "dataset_path": str(dataset_path),
        "num_training_samples": len(formatted_dataset),
    }
    
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[INFO] Training metadata saved to {output_dir / 'training_metadata.json'}")
    print(f"\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80 + "\n")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Gemma-2-9b-it on synthetic CVE dataset using Unsloth and LoRA"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/dataset.jsonl",
        help="Path to the fine-tuning dataset (JSONL format)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/train_results",
        help="Output directory for checkpoints and final model",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device training batch size (GPU memory constrained)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (increases effective batch size without GPU memory overhead)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Warmup ratio (fraction of total steps)",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Log metrics every N steps",
    )
    
    args = parser.parse_args()
    
    train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
    )


if __name__ == "__main__":
    main()
