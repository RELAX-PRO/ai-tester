# Training Implementation Summary

## What Was Created

This document summarizes the complete fine-tuning pipeline for the Gemma-2-9b-it student model using your synthetic CVE dataset.

### Files Created/Modified

1. **`src/cve_synth/train.py`** (NEW - 400+ lines)
   - Complete Unsloth + LoRA fine-tuning script
   - Loads `data/dataset.jsonl` with HuggingFace datasets
   - Formats records into Gemma-2 chat templates with Chain-of-Thought
   - 4-bit quantization (bitsandbytes) + LoRA (PEFT)
   - SFTTrainer from TRL for supervised fine-tuning
   - Optimized for 16GB VRAM GPUs
   - Comprehensive logging and checkpoint management

2. **`pyproject.toml`** (UPDATED)
   - Added `[project.optional-dependencies]` with `train` extra
   - Includes: unsloth, transformers, datasets, trl, torch, peft, bitsandbytes, accelerate, tensorboard

3. **`TRAINING_GUIDE.md`** (NEW - Comprehensive)
   - Step-by-step installation instructions
   - Quick start commands (minimal and advanced)
   - Parameter explanations for 16GB VRAM
   - Dataset formatting details (Chain-of-Thought structure)
   - Output directory structure
   - Model usage examples (inference, merging)
   - Troubleshooting guide
   - Performance expectations

4. **`verify_training_setup.py`** (NEW - Smoke Test)
   - Pre-training validation script
   - Checks: dataset exists, structure valid, all dependencies installed
   - Verifies CUDA/GPU availability
   - Quick diagnostic tool

5. **`README.md`** (UPDATED)
   - Added training section
   - Updated overview to include fine-tuning pipeline
   - Added quick start for both synthesis and training

## Architecture Overview

```
Dataset Generation (Teacher Model)
├── Groq API (openai/gpt-oss-120b)
├── DeepSeek API (v4-pro)
└── Output: data/dataset.jsonl (1304+ records)

Fine-Tuning Pipeline (Student Model)
├── Load data/dataset.jsonl
├── Format with Chain-of-Thought prompts
├── Initialize Gemma-2-9b-it (4-bit, Unsloth)
├── Apply LoRA (rank=16, ~15-20M params)
├── Train with SFTTrainer
└── Save: ./output/train_results/final_model/
```

## Key Features

### 1. Dataset Formatting (Chain-of-Thought)

Each CVE record is formatted as a Gemma-2 chat template:

```
<start_of_turn>user
Analyze the following security vulnerability...
<end_of_turn>
<start_of_turn>model
**Root Cause Analysis:**
[teacher model's root_cause]

**Chain of Thought Reasoning:**
1. [reasoning_chain[0]]
2. [reasoning_chain[1]]
...

**Vulnerability Classification:**
Tags: [tags]

**Assembly-Level Fix:**
[assembly_fix]
<end_of_turn>
```

This structure:
- **Enforces explicit reasoning** (chain-of-thought outputs)
- **Reuses teacher annotations** from your Groq/DeepSeek models
- **Matches Gemma-2's chat format** for optimal alignment
- **Preserves all analysis fields** (root_cause, tags, assembly_fix)

### 2. Model Initialization (Unsloth + LoRA)

```python
# Load with 4-bit quantization (2.25 GB weights)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b-it-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)

# Apply LoRA (rank=16, ~15-20M trainable params)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
```

**Memory breakdown (16GB GPU):**
- Base model (4-bit): ~2.25 GB
- LoRA adapters: ~0.08 GB
- Optimizer state: ~1-2 GB
- Activations: ~1-2 GB
- **Total headroom**: ~10+ GB free (safe margin)

### 3. Training Configuration (16GB VRAM Optimized)

- **Batch size**: 2 per GPU (aggressive constraint)
- **Gradient accumulation**: 4 steps → effective batch size 8
- **Precision**: mixed-precision (fp16/bfloat16)
- **Optimizer**: AdamW 8-bit (memory-efficient)
- **Learning rate**: 2e-4 (LoRA standard)
- **Gradient checkpointing**: Unsloth-optimized kernels
- **Training time**: ~60-120 min for 3 epochs on 1304 records

## Usage

### Installation

```bash
# Install training dependencies
pip install -e ".[train]"

# Or manually:
pip install unsloth transformers datasets trl torch peft bitsandbytes accelerate tensorboard
```

### Run Training (Quick Start)

```bash
# Defaults (3 epochs, 1304 records, ~90 min on 16GB GPU)
python -m cve_synth.train \
    --dataset-path data/dataset.jsonl \
    --output-dir ./output/train_results

# Or verify setup first:
python verify_training_setup.py
```

### Advanced Configuration

```bash
python -m cve_synth.train \
    --dataset-path data/dataset.jsonl \
    --output-dir ./output/train_results \
    --num-train-epochs 5 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --learning-rate 5e-4 \
    --warmup-ratio 0.05 \
    --logging-steps 5 \
    --save-steps 250
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for full parameter reference.

## Output

After training completes:

```
./output/train_results/
├── checkpoint-500/
├── checkpoint-1000/
├── final_model/                    # <-- Use this
│   ├── adapter_config.json         # LoRA config
│   ├── adapter_model.bin           # LoRA weights (~80 MB)
│   ├── config.json
│   └── tokenizer_config.json
├── logs/                           # TensorBoard logs
├── training_metadata.json
└── README.md
```

## Next Steps

1. **Install training dependencies:**
   ```bash
   pip install -e ".[train]"
   ```

2. **Verify setup:**
   ```bash
   python verify_training_setup.py
   ```

3. **Run training (first time, default config):**
   ```bash
   python -m cve_synth.train \
       --dataset-path data/dataset.jsonl \
       --output-dir ./output/train_results
   ```

4. **Monitor with TensorBoard** (optional):
   ```bash
   tensorboard --logdir ./output/train_results/logs
   ```

5. **Use fine-tuned model** (see TRAINING_GUIDE.md for inference examples)

## Key Hyperparameters Explained

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `r` (LoRA rank) | 16 | Balance between expressiveness (higher r) and memory (lower r). 16 is proven effective. |
| `lora_alpha` | 16 | Scaling factor. Set to `r` for balanced learning rate. |
| `max_seq_length` | 4096 | Accommodate long CVE descriptions + multi-step reasoning. |
| `batch_size` | 2 | Aggressive constraint for 16GB VRAM. Pair with gradient accumulation. |
| `gradient_accumulation_steps` | 4 | Simulate batch size 8 without exceeding VRAM. |
| `learning_rate` | 2e-4 | Standard for LoRA. Lower than full fine-tuning (2e-5 typically). |
| `warmup_ratio` | 0.03 | 3% of training steps for learning rate warm-up. |

## Troubleshooting

**Q: "No module named 'transformers'"**
- A: Run `pip install -e ".[train]"` to install all training dependencies.

**Q: CUDA out-of-memory error**
- A: Reduce `--batch-size 1` and increase `--gradient-accumulation-steps 16`.

**Q: Training is slow**
- A: Normal on 16GB GPU (~30K tokens/min). Use `--logging-steps 50` to reduce I/O.

**Q: Model quality is poor**
- A: Generate more synthetic data (higher `--limit` in pipeline) or train longer (`--num-train-epochs 10`).

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed troubleshooting.

## References

- **Unsloth**: https://github.com/unslothai/unsloth
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **Gemma-2**: https://blog.google/technology/ai/google-gemma-2/
- **SFTTrainer (TRL)**: https://huggingface.co/docs/trl/

---

**Status**: ✅ Complete and ready to use.

For questions, see [TRAINING_GUIDE.md](TRAINING_GUIDE.md) or review inline comments in `src/cve_synth/train.py`.
