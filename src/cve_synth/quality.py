from __future__ import annotations

from dataclasses import dataclass

from .models import DatasetRecord


@dataclass(slots=True)
class QualityThresholds:
    minimum_score: float = 0.7
    minimum_confidence: float = 0.65
    minimum_evidence_spans: int = 1


def score_record(record: DatasetRecord) -> float:
    score = 0.0
    score += 0.25 if record.source.cve_id or record.source.ghsa_id else 0.0
    score += 0.25 if record.evidence_spans else 0.0
    score += 0.25 if len(record.analysis.reasoning_chain) >= 3 else 0.1
    score += 0.25 if record.analysis.confidence >= 0.8 else 0.1 if record.analysis.confidence >= 0.6 else 0.0
    return min(score, 1.0)


def is_acceptable(record: DatasetRecord, thresholds: QualityThresholds | None = None) -> bool:
    thresholds = thresholds or QualityThresholds()
    if len(record.evidence_spans) < thresholds.minimum_evidence_spans:
        return False
    if record.analysis.confidence < thresholds.minimum_confidence:
        return False
    return score_record(record) >= thresholds.minimum_score
