from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, request

from .extract import ExtractionResult
from .models import AnalysisRecord, SourceRecord
from .prompting import build_teacher_messages


DEFAULT_MODEL = "deepseek-v4-pro"


@dataclass(slots=True)
class DeepSeekConfig:
    api_base: str = "https://api.deepseek.com"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120.0
    prompt_version: str = "v1"
    reasoning_mode: str = "think_max"
    endpoint_path: str = "/chat/completions"


class DeepSeekRateLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class DeepSeekClient:
    def __init__(self, api_key: str, config: DeepSeekConfig | None = None) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self.api_key = api_key.strip()
        self.config = config or DeepSeekConfig()

    def build_messages(self, source: SourceRecord, extraction: ExtractionResult, target_assembly: str) -> list[dict[str, str]]:
        return build_teacher_messages(source, extraction, target_assembly, truncate=False)

    def analyze(self, source: SourceRecord, extraction: ExtractionResult, *, target_assembly: str = "x86-64") -> AnalysisRecord:
        payload = {
            "model": self.config.model,
            "messages": self.build_messages(source, extraction, target_assembly),
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
            "reasoning": {
                "mode": self.config.reasoning_mode,
            },
            "reasoning_mode": self.config.reasoning_mode,
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
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
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
                raise DeepSeekRateLimitError(f"DeepSeek rate limit exceeded: {body}", retry_after_seconds=retry_after_seconds) from exc
            raise RuntimeError(f"DeepSeek request failed with HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("DeepSeek response did not contain choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("DeepSeek response content is empty")
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
            raise ValueError(f"DeepSeek response missing required fields: {sorted(missing)}")
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


def client_from_env(
    *,
    model: str = DEFAULT_MODEL,
    api_base: str | None = None,
    timeout_seconds: float = 120.0,
    prompt_version: str = "v1",
    reasoning_mode: str = "think_max",
    endpoint_path: str = "/chat/completions",
) -> DeepSeekClient:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    config = DeepSeekConfig(
        api_base=api_base or os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
        model=model,
        timeout_seconds=timeout_seconds,
        prompt_version=prompt_version,
        reasoning_mode=reasoning_mode,
        endpoint_path=endpoint_path,
    )
    return DeepSeekClient(api_key=api_key, config=config)
