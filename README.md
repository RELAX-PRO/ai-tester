# cve-synth

Pipeline for turning recent CVE and GitHub Security Advisory reports into a structured JSONL dataset for fine-tuning Qwen2.5-Coder-7B.

The pipeline now does three things end to end:

1. Fetch recent NVD and GitHub advisory data into `data/raw/`.
2. Filter the input down to memory-corruption-relevant reports.
3. Send the surviving reports through an LLM teacher model and write the annotated dataset to `data/dataset.jsonl`.

The training script then converts that JSONL into Qwen chat-format examples for student fine-tuning.

## What It Uses

- Default teacher model: `openai/gpt-oss-120b` through Groq.
- Optional teacher fallback: Gemini, then DeepSeek if configured.
- Student model: `Qwen2.5-Coder-7B-Instruct`.
- Dataset format: JSONL with `source`, `evidence_spans`, `analysis`, `quality_score`, and `tags`.

## Environment

Create a `.env` file in the project root with the keys you actually use:

```env
GROQ_API_KEYS=key1,key2,key3
GEMINI_API_KEY=your_gemini_key
GITHUB_TOKEN=your_github_token
NVD_API_KEY=your_nvd_key
```

Optional legacy support:

```env
GROQ_API_KEY=single_key
GEMINI_API_KEYS=key1,key2
DEEPSEEK_API_KEY=legacy_key
DEEPSEEK_API_KEYS=key1,key2
```

The loader reads `.env` from the current working directory first, then falls back to the project root.

## Repository Layout

- `src/cve_synth/fetch_data.py` downloads recent NVD CVEs and GitHub advisories.
- `src/cve_synth/ingest.py` normalizes raw files into `SourceRecord` objects.
- `src/cve_synth/extract.py` pulls vulnerable snippets and surrounding context.
- `src/cve_synth/filtering.py` keeps only memory-corruption candidates.
- `src/cve_synth/groq_client.py`, `src/cve_synth/gemini_client.py`, and `src/cve_synth/deepseek_client.py` handle teacher-model calls.
- `src/cve_synth/pipeline.py` orchestrates filtering, analysis, checkpointing, quality gating, and JSONL writing.
- `src/cve_synth/train.py` converts the dataset into Qwen2.5-Coder-7B training text.

## Quick Start

### 1. Fetch Raw Data

```bash
python -m cve_synth.fetch_data --output-dir data/raw
```

This downloads normalized recent records into:

- `data/raw/nvd_cve_recent.json`
- `data/raw/github_security_advisories.json`

### 2. Build the Dataset

Run the main pipeline:

```bash
python -m cve_synth.cli \
    --input-dir data/raw \
    --output data/dataset.jsonl \
    --checkpoint data/checkpoint.json
```

Behavior:

- Memory-corruption-relevant records are kept.
- Records already processed in the checkpoint are skipped.
- Groq is the default teacher backend with model `openai/gpt-oss-120b`.
- Gemini is tried when configured, and DeepSeek remains a legacy fallback.
- Every accepted record is appended to `data/dataset.jsonl`.

### 3. Fine-Tune Qwen2.5-Coder-7B

Install the training extras:

```bash
pip install -e ".[train]"
```

Run training:

```bash
python -m cve_synth.train \
    --dataset-path data/dataset.jsonl \
    --output-dir ./output/train_results \
    --num-train-epochs 3 \
    --batch-size 2 \
    --gradient-accumulation-steps 4
```

Before training, you can sanity-check the dataset and environment:

```bash
python verify_training_setup.py
```

## Provider Selection

The CLI accepts a provider order via `--provider-priority`.

Examples:

```bash
python -m cve_synth.cli --provider-priority groq,gemini,deepseek
python -m cve_synth.cli --provider-priority gemini,groq
```

If the first provider fails or is rate limited, the pipeline falls back to the next configured provider automatically.

## Output Schema

Each dataset record contains:

- source metadata
- extracted evidence spans
- `analysis.vulnerability_summary`
- `analysis.root_cause`
- `analysis.reasoning_chain`
- `analysis.fix_strategy`
- `analysis.assembly_fix`
- `analysis.tags`
- `analysis.confidence`

The trainer converts that into this student-facing structure:

```text
System: expert security researcher and code reviewer
User: CVE ID, summary, root cause, vulnerable snippet
Assistant: reasoning chain, fix strategy, assembly fix, tags
```

## Helpful Commands

```bash
python -m cve_synth.fetch_data --help
python -m cve_synth.cli --help
python -m cve_synth.train --help
```

## Notes

- The pipeline is intentionally conservative. It only keeps strong memory-corruption indicators.
- Groq remains the default model path because it matches the existing `openai/gpt-oss-120b` setup.
- Gemini support is optional and uses the Google Generative Language API.
- The training script expects the dataset produced by this pipeline, not raw CVE JSON.
