from __future__ import annotations

from dataclasses import dataclass
import re

from .models import EvidenceSpan, SourceRecord


CODE_FENCE_RE = re.compile(r"```(?:[\w#+.-]+)?\n(.*?)```", re.DOTALL)


@dataclass(slots=True)
class ExtractionResult:
    vulnerable_snippet: str
    evidence_spans: list[EvidenceSpan]
    surrounding_context: str


def extract_evidence(source: SourceRecord) -> ExtractionResult:
    matches = list(CODE_FENCE_RE.finditer(source.raw_text))
    if matches:
        match = matches[0]
        snippet = match.group(1).strip()
        span = EvidenceSpan(
            start=match.start(1),
            end=match.end(1),
            text=snippet,
            rationale="Primary code block cited in the source report.",
        )
        context_start = max(0, match.start() - 200)
        context_end = min(len(source.raw_text), match.end() + 200)
        return ExtractionResult(snippet, [span], source.raw_text[context_start:context_end])

    lines = [line.strip() for line in source.raw_text.splitlines() if line.strip()]
    snippet = "\n".join(lines[: min(12, len(lines))])
    if not snippet:
        snippet = source.raw_text[:500].strip()
    span = EvidenceSpan(start=0, end=min(len(snippet), len(source.raw_text)), text=snippet, rationale="Fallback evidence extraction from the report body.")
    return ExtractionResult(snippet, [span], source.raw_text[: min(len(source.raw_text), 800)])
