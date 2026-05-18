from __future__ import annotations

import re

from .extract import ExtractionResult
from .models import SourceRecord


_MEMORY_CORRUPTION_PATTERNS = (
    re.compile(r"\bmemory corruption\b", re.IGNORECASE),
    re.compile(r"\bbuffer overflow\b", re.IGNORECASE),
    re.compile(r"\bstack overflow\b", re.IGNORECASE),
    re.compile(r"\bheap overflow\b", re.IGNORECASE),
    re.compile(r"\bout[- ]of[- ]bounds\b", re.IGNORECASE),
    re.compile(r"\buse[- ]after[- ]free\b", re.IGNORECASE),
    re.compile(r"\buaf\b", re.IGNORECASE),
    re.compile(r"\bdouble free\b", re.IGNORECASE),
    re.compile(r"\binteger overflow\b", re.IGNORECASE),
    re.compile(r"\barbitrary write\b", re.IGNORECASE),
    re.compile(r"\bwrite[- ]what[- ]where\b", re.IGNORECASE),
)


def is_memory_corruption_candidate(source: SourceRecord, extraction: ExtractionResult | None = None) -> bool:
    haystacks = [source.title, source.raw_text, source.source_id, " ".join(source.affected_components)]
    if extraction is not None:
        haystacks.append(extraction.vulnerable_snippet)
        haystacks.append(extraction.surrounding_context)

    combined = "\n".join(part for part in haystacks if part)
    return any(pattern.search(combined) for pattern in _MEMORY_CORRUPTION_PATTERNS)
