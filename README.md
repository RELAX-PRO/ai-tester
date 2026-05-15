# cve-synth

Pipeline for turning CVE reports and GitHub security advisories into a structured JSONL fine-tuning dataset, with built-in support for fine-tuning student models (Gemma-2-9b-it) using Unsloth and LoRA.

## Status

This repository contains a complete pipeline for:

1. **Data Synthesis**: Ingest CVE reports → teacher model (Groq/DeepSeek) → structured JSONL dataset
2. **Model Fine-Tuning**: Student model (Gemma-2-9b-it) on synthetic data using Unsloth + LoRA

Core features:
- Canonical data models for CVE analysis
- Incremental JSONL writer with checkpointing
- Multi-key rate limiting for API providers
- Teacher model client (Groq API)
- Chain-of-Thought dataset formatting
- Unsloth-optimized LoRA training for 16GB VRAM

## Quick Start

### 1. Synthesize Dataset (Teacher Model)

#### Fetch raw CVE data:
```bash
python -m cve_synth.fetch_data --output-dir data/raw
```

#### Generate synthetic annotations (teacher model):
```bash
python -m cve_synth.cli \
    --input-dir data/raw \
    --output data/dataset.jsonl \
    --checkpoint data/checkpoint.json \
    --api-keys-file groq_keys.txt \
    --limit 100
```

This creates `data/dataset.jsonl` with structured CVE analysis (1304+ records in your current setup).

### 2. Fine-Tune Student Model (Gemma-2-9b-it)

#### Install training dependencies:
```bash
pip install -e ".[train]"
```

#### Run training (defaults optimized for 16GB VRAM):
```bash
python -m cve_synth.train \
    --dataset-path data/dataset.jsonl \
    --output-dir ./output/train_results \
    --num-train-epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 4
```

#### Verify setup before training:
```bash
python verify_training_setup.py
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed configuration and troubleshooting.

## Run (Legacy - Teacher Model Only)

```bash
python -m cve_synth.cli --input-dir data/raw --output data/dataset.jsonl --checkpoint data/checkpoint.json
```

## Small-Scale Synthesis

```bash
python -m cve_synth.cli --input-dir data/raw --output data/dataset.jsonl --checkpoint data/checkpoint.json --limit 10
```

