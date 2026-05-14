from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .checkpoint import CheckpointStore
from .deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekRateLimitError
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
    model_name: str = "deepseek-v4-pro"


def run_pipeline(config: PipelineConfig) -> dict[str, int]:
    sources = load_sources(config.input_dir)
    checkpoint_store = CheckpointStore(config.checkpoint_path)
    checkpoint = checkpoint_store.load()
    writer = JsonlWriter(config.output_path)
    limiter = MultiKeyRateLimiter(config.api_keys)
    thresholds = QualityThresholds(minimum_score=config.min_quality_score, minimum_confidence=config.min_confidence)
    deepseek_config = DeepSeekConfig(model=config.model_name, prompt_version=config.prompt_version)

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
                deepseek_config=deepseek_config,
            )

            candidate = DatasetRecord(
                source=source,
                evidence_spans=extraction.evidence_spans,
                analysis=analysis,
                record_id=f"{source.source_id}:{len(checkpoint.processed_source_ids) + 1}",
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
        except Exception:
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
    deepseek_config: DeepSeekConfig,
):
    key_attempts = max(1, len(limiter.snapshot()) * 2)
    last_error: Exception | None = None

    for _ in range(key_attempts):
        api_key = limiter.acquire()
        client = DeepSeekClient(api_key=api_key, config=deepseek_config)
        try:
            analysis = client.analyze(source, extraction, target_assembly=target_assembly)
            limiter.mark_success(api_key)
            return analysis
        except DeepSeekRateLimitError as exc:
            limiter.mark_rate_limited(api_key, exc.retry_after_seconds)
            last_error = exc
        except Exception as exc:
            limiter.mark_failure(api_key)
            last_error = exc

    if last_error is None:
        raise RuntimeError("analysis failed without a captured error")
    raise last_error
