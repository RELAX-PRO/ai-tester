from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .checkpoint import CheckpointStore
from .groq_client import GroqClient, GroqConfig, GroqRateLimitError
from .extract import extract_evidence
from .ingest import load_sources
from .models import DatasetRecord
from .quality import QualityThresholds, is_acceptable, score_record
from .rate_limit import MultiKeyRateLimiter
from .writer import JsonlWriter


@dataclass(slots=True)
class PipelineConfig:
    input_dir: Path
    output_path: Path
    checkpoint_path: Path
    api_keys: list[str]
    limit: int | None = None
    target_assembly: str = "x86-64"
    min_quality_score: float = 0.7
    min_confidence: float = 0.65
    prompt_version: str = "v1"
    model_name: str = "deepseek-r1-distill-llama-70b"


def run_pipeline(config: PipelineConfig) -> dict[str, int]:
    sources = load_sources(config.input_dir)
    checkpoint_store = CheckpointStore(config.checkpoint_path)
    checkpoint = checkpoint_store.load()
    writer = JsonlWriter(config.output_path)
    limiter = MultiKeyRateLimiter(config.api_keys)
    thresholds = QualityThresholds(minimum_score=config.min_quality_score, minimum_confidence=config.min_confidence)
    groq_config = GroqConfig(model=config.model_name, prompt_version=config.prompt_version)

    processed = 0
    skipped = 0
    written = 0
    failed = 0

    for ingested in sources:
        if config.limit is not None and processed >= config.limit:
            break

        source = ingested.record
        if source.source_id in checkpoint.processed_source_ids:
            skipped += 1
            continue

        extraction = extract_evidence(source)

        try:
            analysis = _analyze_with_rotation(
                limiter=limiter,
                source=source,
                extraction=extraction,
                target_assembly=config.target_assembly,
                groq_config=groq_config,
            )

            candidate = DatasetRecord(
                source=source,
                evidence_spans=extraction.evidence_spans,
                analysis=analysis,
                record_id=f"{source.source_id}:{len(checkpoint.processed_source_ids) + 1}",
                tags=analysis.tags,
                quality_score=0.0,
            )
            candidate.quality_score = score_record(candidate)

            if not is_acceptable(candidate, thresholds):
                failed += 1
                checkpoint.failed_source_ids.add(source.source_id)
                checkpoint_store.save(checkpoint)
                continue

            writer.append(candidate)
            checkpoint.processed_source_ids.add(source.source_id)
            checkpoint.last_output_index += 1
            checkpoint_store.save(checkpoint)
            written += 1
            processed += 1
        except Exception as e:
            # تمت إضافة سطر الطباعة هذا لكشف العلة الأساسية
            print(f"[ERROR] Failed to process {source.source_id}: {type(e).__name__} - {str(e)}")
            
            failed += 1
            checkpoint.failed_source_ids.add(source.source_id)
            checkpoint_store.save(checkpoint)

    return {
        "sources": len(sources),
        "processed": processed,
        "skipped": skipped,
        "written": written,
        "failed": failed,
    }


def _analyze_with_rotation(
    *,
    limiter: MultiKeyRateLimiter,
    source,
    extraction,
    target_assembly: str,
    groq_config: GroqConfig,
):
    """
    Analyze with key rotation and exponential backoff for rate limiting.
    
    Strategy:
    1. Try to acquire a key and analyze
    2. On 429, mark key as rate-limited and rotate to next
    3. If all keys hit 429, sleep with exponential backoff (60s → 120s → 300s max)
    4. Retry with exponential backoff until success or non-recoverable error
    """
    last_error: Exception | None = None
    all_keys_rate_limited_count = 0  # Track consecutive "all keys hit 429" events
    backoff_sleep_seconds = 60  # Start with 60 seconds

    # Allow multiple full rotations through all keys with exponential backoff
    max_total_attempts = len(limiter.snapshot()) * 3  # 3 full rotations
    total_attempts = 0

    while total_attempts < max_total_attempts:
        total_attempts += 1
        
        try:
            api_key = limiter.acquire()
        except RuntimeError:
            # No key currently available; wait for next available
            sleep_for = limiter.seconds_until_next_available()
            if sleep_for > 0:
                print(f"[WAIT] All keys rate-limited, waiting {sleep_for:.1f}s before retry...")
                time.sleep(sleep_for)
            try:
                api_key = limiter.acquire()
            except RuntimeError:
                # Still no keys available after waiting
                all_keys_rate_limited_count += 1
                if all_keys_rate_limited_count >= 1:
                    print(f"[BACKOFF] All keys exhausted, sleeping {backoff_sleep_seconds}s before retry...")
                    try:
                        time.sleep(backoff_sleep_seconds)
                    except KeyboardInterrupt:
                        raise
                    backoff_sleep_seconds = min(backoff_sleep_seconds * 2, 300)  # Exponential backoff, cap at 5 min
                continue

        client = GroqClient(api_key=api_key, config=groq_config)
        try:
            analysis = client.analyze(source, extraction, target_assembly=target_assembly)
            limiter.mark_success(api_key)
            return analysis
        except GroqRateLimitError as exc:
            limiter.mark_rate_limited(api_key, exc.retry_after_seconds)
            last_error = exc
            # Check if all keys are now rate-limited
            snapshot = limiter.snapshot()
            all_limited = all(state["available_at"] > time.monotonic() for state in snapshot)
            if all_limited:
                all_keys_rate_limited_count += 1
                print(f"[RATE_LIMIT] All {len(snapshot)} keys hit 429 (attempt {total_attempts})")
        except Exception as exc:
            limiter.mark_failure(api_key)
            last_error = exc
            print(f"[ERROR] Analysis failed for {source.source_id}: {type(exc).__name__} - {str(exc)}")

    if last_error is None:
        raise RuntimeError("analysis failed without a captured error")
    raise last_error
