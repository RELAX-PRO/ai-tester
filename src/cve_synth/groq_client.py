from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request

from .extract import ExtractionResult
from .models import AnalysisRecord, SourceRecord


DEFAULT_MODEL = "openai/gpt-oss-120b"
TOTAL_TOKEN_BUDGET = 4000  # ~16,000 characters at 4 chars/token
CHARS_PER_TOKEN = 4


@dataclass(slots=True)
class GroqConfig:
    api_base: str = "https://api.groq.com/openai/v1"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120.0
    prompt_version: str = "v1"
    endpoint_path: str = "/chat/completions"
    user_agent: str = "cve-synth/0.1.0"


class GroqRateLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class GroqClient:
    def __init__(self, api_key: str, config: GroqConfig | None = None) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self.api_key = api_key.strip()
        self.config = config or GroqConfig()

    @staticmethod
    def _truncate_to_token_budget(text: str, max_chars: int) -> str:
        """
        Truncate text to approximately max_chars characters.
        Appends [truncated] marker if truncation occurs.
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars - 12] + "...[truncated]"

    def build_messages(self, source: SourceRecord, extraction: ExtractionResult, target_assembly: str) -> list[dict[str, str]]:
        """
        Build message payload with aggressive token-aware truncation.
        
        Token budget breakdown (~4000 tokens = ~16,000 chars):
        - System message + schema overhead: ~8,000 chars
        - Metadata fields (IDs, titles, severity): ~2,000 chars
        - vulnerable_snippet: KEEP INTACT (typically 500-1000 chars)
        - raw_text + surrounding_context: ~5,000 chars combined
          - raw_text: 60% = ~3,000 chars
          - surrounding_context: 40% = ~2,000 chars
        """
        system = (
            "You are a cybersecurity dataset annotator. Produce structured, evidence-backed analysis only. "
            "Do not provide exploit instructions. Focus on root cause, defensive reasoning, assembly-level remediation notes, "
            "and analyst-facing vulnerability categorization tags."
        )
        # Enforce strict JSON-only responses from the model to avoid parsing issues
        system += (
            " Always respond with a single valid JSON object matching the requested schema. "
            "Do NOT include markdown, code fences, explanatory text, or any extra characters."
        )

        # Aggressive truncation for Groq's 8000 TPM constraint
        snippet_len = len(extraction.vulnerable_snippet)
        remaining_budget = 5000  # Conservative allocation for raw_text + surrounding_context
        raw_text_budget = int(remaining_budget * 0.6)  # 60% to raw_text
        context_budget = remaining_budget - raw_text_budget  # 40% to surrounding_context

        truncated_raw_text = self._truncate_to_token_budget(source.raw_text, raw_text_budget)
        truncated_context = self._truncate_to_token_budget(extraction.surrounding_context, context_budget)

        # Log truncation for debugging
        if len(source.raw_text) > raw_text_budget or len(extraction.surrounding_context) > context_budget:
            print(
                f"[TRUNCATE] {source.source_id}: raw_text {len(source.raw_text)}/{raw_text_budget} chars, "
                f"context {len(extraction.surrounding_context)}/{context_budget} chars"
            )

        user = {
            "source_id": source.source_id,
            "source_type": source.source_type,
            "title": source.title,
            "cve_id": source.cve_id,
            "ghsa_id": source.ghsa_id,
            "severity": source.severity,
            "raw_text": truncated_raw_text,
            "vulnerable_snippet": extraction.vulnerable_snippet,
            "surrounding_context": truncated_context,
            "target_assembly": target_assembly,
            "output_schema": {
                "vulnerability_summary": "string",
                "root_cause": "string",
                "reasoning_chain": ["string"],
                "tags": ["#MemoryCorruption", "#LogicError", "#Injection"],
                "assembly_fix": "string",
                "fix_strategy": "string",
                "confidence": "number 0..1",
            },
            "tagging_rules": [
                "Return one or more short hash-prefixed tags.",
                "Choose tags that categorize the vulnerability class, such as #MemoryCorruption, #LogicError, or #Injection.",
                "Prefer the smallest set of tags that accurately describes the issue.",
            ],
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=True, indent=2)},
        ]

    def analyze(self, source: SourceRecord, extraction: ExtractionResult, *, target_assembly: str = "x86-64") -> AnalysisRecord:
        payload = {
            "model": self.config.model,
            "messages": self.build_messages(source, extraction, target_assembly),
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        print(f"[DEBUG] Sending payload for {source.source_id} (Length: {len(str(payload))} chars)")
        response = self._request_json(self.config.endpoint_path, payload)
        content = self._extract_content(response)
        parsed = self._parse_json_content(content)
        return AnalysisRecord(
            vulnerability_summary=str(parsed["vulnerability_summary"]),
            root_cause=str(parsed["root_cause"]),
            reasoning_chain=[str(item) for item in parsed["reasoning_chain"]],
            tags=self._normalize_tags(parsed.get("tags", [])),
            vulnerable_snippet=extraction.vulnerable_snippet,
            assembly_fix=str(parsed["assembly_fix"]),
            fix_strategy=str(parsed["fix_strategy"]),
            confidence=float(parsed["confidence"]),
            model_name=self.config.model,
            prompt_version=self.config.prompt_version,
        )

    def _request_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        url = self.config.api_base.rstrip("/") + path
        req = request.Request(url, data=data, method="POST")
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("User-Agent", self.config.user_agent)
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                retry_after_value = exc.headers.get("Retry-After") if exc.headers else None
                retry_after_seconds = None
                if retry_after_value:
                    try:
                        retry_after_seconds = float(retry_after_value)
                    except ValueError:
                        retry_after_seconds = None
                raise GroqRateLimitError(f"Groq rate limit exceeded: {body}", retry_after_seconds=retry_after_seconds) from exc
            if exc.code == 403 and self._looks_like_waf_block(body):
                raise RuntimeError(
                    "Groq request was blocked by an upstream protection layer (possible WAF/bot detection). "
                    "Try a different egress IP, confirm the API key and account are allowed, or contact Groq support. "
                    f"HTTP 403 body: {body}"
                ) from exc
            raise RuntimeError(f"Groq request failed with HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("Groq response did not contain choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Groq response content is empty")
        return content

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:].strip()
        parsed = json.loads(text)
        required = {"vulnerability_summary", "root_cause", "reasoning_chain", "assembly_fix", "fix_strategy", "confidence"}
        missing = required - set(parsed)
        if missing:
            raise ValueError(f"Groq response missing required fields: {sorted(missing)}")
        if not isinstance(parsed["reasoning_chain"], list):
            raise ValueError("reasoning_chain must be a list")
        if "tags" in parsed and not isinstance(parsed["tags"], list):
            raise ValueError("tags must be a list when present")
        return parsed

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if not isinstance(tags, list):
            return []
        normalized: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            if not isinstance(tag, str):
                continue
            cleaned = tag.strip()
            if not cleaned:
                continue
            if not cleaned.startswith("#"):
                cleaned = f"#{cleaned.lstrip('#')}"
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    @staticmethod
    def _looks_like_waf_block(body: str) -> bool:
        lowered = body.lower()
        return any(
            marker in lowered
            for marker in (
                "waf",
                "bot",
                "automated",
                "access denied",
                "request blocked",
                "cloudflare",
                "security check",
            )
        )


def client_from_env(
    *,
    model: str = DEFAULT_MODEL,
    api_base: str | None = None,
    timeout_seconds: float = 120.0,
    prompt_version: str = "v1",
    endpoint_path: str = "/chat/completions",
) -> GroqClient:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    config = GroqConfig(
        api_base=api_base or os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1"),
        model=model,
        timeout_seconds=timeout_seconds,
        prompt_version=prompt_version,
        endpoint_path=endpoint_path,
    )
    return GroqClient(api_key=api_key, config=config)
