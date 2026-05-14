from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from cve_synth.pipeline import PipelineConfig, run_pipeline
else:
    from .pipeline import PipelineConfig, run_pipeline


DEFAULT_INPUT_DIR = Path("data/raw")
DEFAULT_OUTPUT_PATH = Path("data/dataset.jsonl")
DEFAULT_CHECKPOINT_PATH = Path("data/checkpoint.json")


def _load_env_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_dotenv_files() -> None:
    project_root = Path(__file__).resolve().parents[2]
    cwd_env_files = [Path.cwd() / ".env", Path.cwd() / ".env.local"]
    loaded_any_cwd_env = False

    for env_path in cwd_env_files:
        if env_path.exists():
            _load_env_file(env_path)
            loaded_any_cwd_env = True

    if loaded_any_cwd_env:
        return

    for env_path in (project_root / ".env", project_root / ".env.local"):
        if env_path.exists():
            _load_env_file(env_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a synthetic CVE fine-tuning dataset.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing raw CVE reports and advisories")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL file")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH, help="Checkpoint JSON file")
    parser.add_argument("--limit", type=int, help="Optional maximum number of accepted records to write")
    parser.add_argument("--api-key", action="append", dest="api_keys", default=[], help="Groq API key; may be provided multiple times")
    parser.add_argument("--api-keys-file", type=Path, help="Optional file with one API key per line")
    parser.add_argument("--target-assembly", default="x86-64", help="Target assembly family for fix synthesis")
    parser.add_argument("--min-quality-score", type=float, default=0.7)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--model-name", default="deepseek-r1-distill-llama-70b")
    return parser.parse_args(argv)


def _load_api_keys(args: argparse.Namespace) -> list[str]:
    keys = [key.strip() for key in args.api_keys if key and key.strip()]
    if args.api_keys_file:
        keys.extend(line.strip() for line in args.api_keys_file.read_text(encoding="utf-8").splitlines() if line.strip())
    # Prioritize Groq API keys from environment
    env_keys = [key.strip() for key in os.environ.get("GROQ_API_KEYS", "").split(",") if key.strip()]
    keys.extend(env_keys)
    env_key = os.environ.get("GROQ_API_KEY", "").strip()
    if env_key and env_key not in keys:
        keys.append(env_key)
    # Fallback to DeepSeek for backward compatibility
    if not keys:
        env_keys = [key.strip() for key in os.environ.get("DEEPSEEK_API_KEYS", "").split(",") if key.strip()]
        keys.extend(env_keys)
        env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if env_key and env_key not in keys:
            keys.append(env_key)
    return keys


def main(argv: list[str] | None = None) -> int:
    _load_dotenv_files()
    args = parse_args(argv)
    api_keys = _load_api_keys(args)
    if not api_keys:
        raise SystemExit("No Groq API key found. Create a .env file with GROQ_API_KEYS or pass --api-key/--api-keys-file.")
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        api_keys=api_keys,
        limit=args.limit,
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
