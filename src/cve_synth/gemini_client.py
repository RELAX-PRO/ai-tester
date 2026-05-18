from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib import error, parse, request

from .extract import ExtractionResult
from .models import AnalysisRecord, SourceRecord
from .prompting import build_teacher_messages


DEFAULT_MODEL = "gemini-2.5-pro"


@dataclass(slots=True)
class GeminiConfig:
    api_base: str = "https://generativelanguage.googleapis.com/v1beta"
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 120.0
    prompt_version: str = "v1"
    temperature: float = 0.2


class GeminiRateLimitError(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class GeminiClient:
    def __init__(self, api_key: str, config: GeminiConfig | None = None) -> None:
        if not api_key.strip():
            raise ValueError("api_key must not be empty")
        self.api_key = api_key.strip()
        self.config = config or GeminiConfig()

    def build_messages(self, source: SourceRecord, extraction: ExtractionResult, target_assembly: str) -> list[dict[str, str]]:
        return build_teacher_messages(source, extraction, target_assembly, truncate=False)

    def analyze(self, source: SourceRecord, extraction: ExtractionResult, *, target_assembly: str = "x86-64") -> AnalysisRecord:
        messages = self.build_messages(source, extraction, target_assembly)
        system_message = next((item["content"] for item in messages if item["role"] == "system"), "")
        user_message = next((item["content"] for item in messages if item["role"] == "user"), "")

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_message}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_message}],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "responseMimeType": "application/json",
            },
        }
        print(f"[DEBUG] Sending Gemini payload for {source.source_id} (Length: {len(str(payload))} chars)")
        response = self._request_json(f"/models/{parse.quote(self.config.model, safe='')}:generateContent", payload)
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
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}key={parse.quote(self.api_key)}"
        req = request.Request(url, data=data, method="POST")
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", "application/json")
        req.add_header("User-Agent", "cve-synth/0.1.0")
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
                raise GeminiRateLimitError(f"Gemini rate limit exceeded: {body}", retry_after_seconds=retry_after_seconds) from exc
            raise RuntimeError(f"Gemini request failed with HTTP {exc.code}: {body}") from exc

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        candidates = response.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError("Gemini response did not contain candidates")
        candidate = candidates[0]
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        parts = content.get("parts") if isinstance(content, dict) else []
        if not isinstance(parts, list) or not parts:
            raise ValueError("Gemini response content did not contain parts")
        text = parts[0].get("text") if isinstance(parts[0], dict) else None
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Gemini response content is empty")
        return text

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
            raise ValueError(f"Gemini response missing required fields: {sorted(missing)}")
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
) -> GeminiClient:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    config = GeminiConfig(
        api_base=api_base or os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta"),
        model=model,
        timeout_seconds=timeout_seconds,
        prompt_version=prompt_version,
    )
    return GeminiClient(api_key=api_key, config=config)
