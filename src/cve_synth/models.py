from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _ensure_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


@dataclass(slots=True)
class EvidenceSpan:
    start: int
    end: int
    text: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class SourceRecord:
    source_id: str
    source_type: str
    title: str
    raw_text: str
    url: str | None = None
    cve_id: str | None = None
    ghsa_id: str | None = None
    published_at: str | None = None
    severity: str | None = None
    affected_components: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.source_id = _ensure_str(self.source_id, "source_id")
        self.source_type = _ensure_str(self.source_type, "source_type")
        self.title = _ensure_str(self.title, "title")
        self.raw_text = _ensure_str(self.raw_text, "raw_text")
        if self.url is not None:
            self.url = _ensure_str(self.url, "url")


@dataclass(slots=True)
class AnalysisRecord:
    vulnerability_summary: str
    root_cause: str
    reasoning_chain: list[str]
    vulnerable_snippet: str
    assembly_fix: str
    fix_strategy: str
    confidence: float
    model_name: str
    prompt_version: str
    created_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.vulnerability_summary = _ensure_str(self.vulnerability_summary, "vulnerability_summary")
        self.root_cause = _ensure_str(self.root_cause, "root_cause")
        self.vulnerable_snippet = _ensure_str(self.vulnerable_snippet, "vulnerable_snippet")
        self.assembly_fix = _ensure_str(self.assembly_fix, "assembly_fix")
        self.fix_strategy = _ensure_str(self.fix_strategy, "fix_strategy")
        self.model_name = _ensure_str(self.model_name, "model_name")
        self.prompt_version = _ensure_str(self.prompt_version, "prompt_version")
        if not self.reasoning_chain or not all(isinstance(step, str) and step.strip() for step in self.reasoning_chain):
            raise ValueError("reasoning_chain must contain non-empty strings")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vulnerability_summary": self.vulnerability_summary,
            "root_cause": self.root_cause,
            "reasoning_chain": self.reasoning_chain,
            "vulnerable_snippet": self.vulnerable_snippet,
            "assembly_fix": self.assembly_fix,
            "fix_strategy": self.fix_strategy,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "created_at": self.created_at,
        }


@dataclass(slots=True)
class DatasetRecord:
    source: SourceRecord
    evidence_spans: list[EvidenceSpan]
    analysis: AnalysisRecord
    record_id: str
    quality_score: float
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.record_id = _ensure_str(self.record_id, "record_id")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError("quality_score must be between 0 and 1")

    def to_jsonl(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": {
                "source_id": self.source.source_id,
                "source_type": self.source.source_type,
                "title": self.source.title,
                "url": self.source.url,
                "cve_id": self.source.cve_id,
                "ghsa_id": self.source.ghsa_id,
                "published_at": self.source.published_at,
                "severity": self.source.severity,
                "affected_components": self.source.affected_components,
                "metadata": self.source.metadata,
                "raw_text": self.source.raw_text,
            },
            "evidence_spans": [span.to_dict() for span in self.evidence_spans],
            "analysis": self.analysis.to_dict(),
            "quality_score": self.quality_score,
            "tags": self.tags,
        }
