from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .pipeline import PipelineConfig, run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic CVE fine-tuning dataset.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing raw CVE reports and advisories")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Checkpoint JSON file")
    parser.add_argument("--api-key", action="append", dest="api_keys", default=[], help="DeepSeek API key; may be provided multiple times")
    parser.add_argument("--api-keys-file", type=Path, help="Optional file with one API key per line")
    parser.add_argument("--target-assembly", default="x86-64", help="Target assembly family for fix synthesis")
    parser.add_argument("--min-quality-score", type=float, default=0.7)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--model-name", default="deepseek-v4-pro-max")
    return parser.parse_args(argv)


def _load_api_keys(args: argparse.Namespace) -> list[str]:
    keys = [key.strip() for key in args.api_keys if key and key.strip()]
    if args.api_keys_file:
        keys.extend(line.strip() for line in args.api_keys_file.read_text(encoding="utf-8").splitlines() if line.strip())
    if not keys:
        raise SystemExit("At least one API key is required via --api-key or --api-keys-file")
    return keys


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        api_keys=_load_api_keys(args),
        target_assembly=args.target_assembly,
        min_quality_score=args.min_quality_score,
        min_confidence=args.min_confidence,
        prompt_version=args.prompt_version,
        model_name=args.model_name,
    )
    result = run_pipeline(config)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
