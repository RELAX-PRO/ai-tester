from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import time

from .checkpoint import CheckpointStore
from .deepseek_client import DeepSeekClient, DeepSeekConfig, DeepSeekRateLimitError
from .filtering import is_memory_corruption_candidate
from .gemini_client import GeminiClient, GeminiConfig, GeminiRateLimitError
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
    model_name: str = "openai/gpt-oss-120b"
    provider_priority: list[str] | None = None
    gemini_api_keys: list[str] | None = None
    deepseek_api_keys: list[str] | None = None
    gemini_model_name: str = "gemini-2.5-pro"
    deepseek_model_name: str = "deepseek-v4-pro"


@dataclass(slots=True)
class _ProviderRuntime:
    name: str
    limiter: MultiKeyRateLimiter
    client_factory: Callable[[str], Any]
    rate_limit_errors: tuple[type[BaseException], ...]


def run_pipeline(config: PipelineConfig) -> dict[str, int]:
    sources = load_sources(config.input_dir)
    checkpoint_store = CheckpointStore(config.checkpoint_path)
    checkpoint = checkpoint_store.load()
    writer = JsonlWriter(config.output_path)
    thresholds = QualityThresholds(minimum_score=config.min_quality_score, minimum_confidence=config.min_confidence)
    provider_runtimes = _build_provider_runtimes(config)

    processed = 0
    skipped = 0
    filtered_out = 0
    written = 0
    failed = 0

    for ingested in sources:
        if config.limit is not None and processed >= config.limit:
            break

        source = ingested.record
        if source.source_id in checkpoint.processed_source_ids:
            skipped += 1
            continue

        if not is_memory_corruption_candidate(source):
            filtered_out += 1
            continue

        extraction = extract_evidence(source)

        try:
            analysis = _analyze_with_provider_priority(
                providers=provider_runtimes,
                source=source,
                extraction=extraction,
                target_assembly=config.target_assembly,
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
            print(f"[SUCCESS] Wrote {source.source_id} to {config.output_path}")
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
        "filtered_out": filtered_out,
        "written": written,
        "failed": failed,
    }


def _build_provider_runtimes(config: PipelineConfig) -> list[_ProviderRuntime]:
    provider_priority = config.provider_priority or ["groq"]
    groq_keys = config.api_keys
    gemini_keys = config.gemini_api_keys or []
    deepseek_keys = config.deepseek_api_keys or []

    runtimes: list[_ProviderRuntime] = []
    for provider_name in provider_priority:
        normalized = provider_name.strip().lower()
        if normalized == "groq":
            if not groq_keys:
                continue

            def _make_groq_client(api_key: str, *, model: str = config.model_name, prompt_version: str = config.prompt_version) -> GroqClient:
                return GroqClient(api_key=api_key, config=GroqConfig(model=model, prompt_version=prompt_version))

            runtimes.append(
                _ProviderRuntime(
                    name="groq",
                    limiter=MultiKeyRateLimiter(groq_keys),
                    client_factory=_make_groq_client,
                    rate_limit_errors=(GroqRateLimitError,),
                )
            )
        elif normalized == "gemini":
            if not gemini_keys:
                continue

            def _make_gemini_client(api_key: str, *, model: str = config.gemini_model_name, prompt_version: str = config.prompt_version) -> GeminiClient:
                return GeminiClient(api_key=api_key, config=GeminiConfig(model=model, prompt_version=prompt_version))

            runtimes.append(
                _ProviderRuntime(
                    name="gemini",
                    limiter=MultiKeyRateLimiter(gemini_keys),
                    client_factory=_make_gemini_client,
                    rate_limit_errors=(GeminiRateLimitError,),
                )
            )
        elif normalized == "deepseek":
            if not deepseek_keys:
                continue

            def _make_deepseek_client(api_key: str, *, model: str = config.deepseek_model_name, prompt_version: str = config.prompt_version) -> DeepSeekClient:
                return DeepSeekClient(api_key=api_key, config=DeepSeekConfig(model=model, prompt_version=prompt_version))

            runtimes.append(
                _ProviderRuntime(
                    name="deepseek",
                    limiter=MultiKeyRateLimiter(deepseek_keys),
                    client_factory=_make_deepseek_client,
                    rate_limit_errors=(DeepSeekRateLimitError,),
                )
            )
    return runtimes


def _analyze_with_provider_priority(
    *,
    providers: list[_ProviderRuntime],
    source,
    extraction,
    target_assembly: str,
):
    last_error: Exception | None = None
    for provider in providers:
        try:
            return _analyze_with_rotation(
                limiter=provider.limiter,
                source=source,
                extraction=extraction,
                target_assembly=target_assembly,
                client_factory=provider.client_factory,
                rate_limit_errors=provider.rate_limit_errors,
                provider_name=provider.name,
            )
        except Exception as exc:
            last_error = exc
            print(f"[FALLBACK] {provider.name} failed for {source.source_id}: {type(exc).__name__} - {str(exc)}")
    if last_error is None:
        raise RuntimeError("no providers were configured")
    raise last_error


def _analyze_with_rotation(
    *,
    limiter: MultiKeyRateLimiter,
    source,
    extraction,
    target_assembly: str,
    client_factory: Callable[[str], Any],
    rate_limit_errors: tuple[type[BaseException], ...],
    provider_name: str,
):
    last_error: Exception | None = None
    all_keys_rate_limited_count = 0
    backoff_sleep_seconds = 60

    max_total_attempts = max(1, len(limiter.snapshot()) * 3)
    total_attempts = 0

    while total_attempts < max_total_attempts:
        total_attempts += 1

        try:
            api_key = limiter.acquire()
        except RuntimeError:
            sleep_for = limiter.seconds_until_next_available()
            if sleep_for > 0:
                print(f"[WAIT] All keys rate-limited, waiting {sleep_for:.1f}s before retry...")
                time.sleep(sleep_for)
            try:
                api_key = limiter.acquire()
            except RuntimeError:
                all_keys_rate_limited_count += 1
                print(f"[BACKOFF] {provider_name} keys exhausted, sleeping {backoff_sleep_seconds}s before retry...")
                try:
                    time.sleep(backoff_sleep_seconds)
                except KeyboardInterrupt:
                    raise
                backoff_sleep_seconds = min(backoff_sleep_seconds * 2, 300)
                continue

        client = client_factory(api_key)
        try:
            analysis = client.analyze(source, extraction, target_assembly=target_assembly)
            limiter.mark_success(api_key)
            return analysis
        except rate_limit_errors as exc:
            limiter.mark_rate_limited(api_key, exc.retry_after_seconds)
            last_error = exc
            snapshot = limiter.snapshot()
            all_limited = all(state["available_at"] > time.monotonic() for state in snapshot)
            if all_limited:
                all_keys_rate_limited_count += 1
                print(f"[RATE_LIMIT] {provider_name}: all {len(snapshot)} keys hit 429 (attempt {total_attempts})")
        except Exception as exc:
            limiter.mark_failure(api_key)
            last_error = exc
            print(f"[ERROR] {provider_name} analysis failed for {source.source_id}: {type(exc).__name__} - {str(exc)}")

    if last_error is None:
        raise RuntimeError(f"{provider_name} analysis failed without a captured error")
    raise last_error
