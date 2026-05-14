# cve-synth

Pipeline starter for turning CVE reports and GitHub security advisories into a structured JSONL fine-tuning dataset.

## Status

This repository currently contains the initial implementation scaffold:

- canonical data models
- incremental JSONL writer
- resumable checkpoint state
- multi-key rate limiting
- DeepSeek-compatible analysis client
- CLI pipeline entry point

## Run

```bash
python -m cve_synth.cli --input-dir data/raw --output data/dataset.jsonl --checkpoint data/checkpoint.json
```
