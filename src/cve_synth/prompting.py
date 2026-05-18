from __future__ import annotations

import json
from dataclasses import dataclass

from .extract import ExtractionResult
from .models import SourceRecord


DEFAULT_OUTPUT_SCHEMA = {
    "vulnerability_summary": "string",
    "root_cause": "string",
    "reasoning_chain": ["string"],
    "fix_strategy": "string",
    "assembly_fix": "string",
    "tags": ["#MemoryCorruption", "#LogicError", "#Injection"],
    "confidence": "number 0..1",
}


@dataclass(frozen=True, slots=True)
class PromptBudget:
    raw_text_chars: int = 3000
    surrounding_context_chars: int = 2000


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 12:
        return text[:max_chars]
    return text[: max_chars - 12] + "...[truncated]"


def build_teacher_messages(
    source: SourceRecord,
    extraction: ExtractionResult,
    target_assembly: str,
    *,
    budget: PromptBudget | None = None,
    output_schema: dict[str, object] | None = None,
    truncate: bool = True,
) -> list[dict[str, str]]:
    budget = budget or PromptBudget()
    output_schema = output_schema or DEFAULT_OUTPUT_SCHEMA

    raw_text = _truncate(source.raw_text, budget.raw_text_chars) if truncate else source.raw_text
    surrounding_context = _truncate(extraction.surrounding_context, budget.surrounding_context_chars) if truncate else extraction.surrounding_context

    system = (
        "You are an elite security researcher and reverse engineer. "
        "When given a vulnerability report, you must output a structured analysis containing vulnerability_summary, "
        "root_cause, reasoning_chain (as a list of logical steps), fix_strategy, assembly_fix, and tags (as a list of hashtags). "
        "Respond only with a JSON object following the exact schema. "
        "Focus on defensive reasoning, memory-corruption analysis, and evidence-backed remediation."
    )

    user = {
        "CVE ID": source.cve_id or source.source_id,
        "Source Type": source.source_type,
        "Title": source.title,
        "Vulnerable Code Snippet": extraction.vulnerable_snippet,
        "Description": raw_text,
        "Surrounding Context": surrounding_context,
        "Target Assembly": target_assembly,
        "Output Schema": output_schema,
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=True, indent=2)},
    ]
