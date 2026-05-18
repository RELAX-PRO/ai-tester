from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace
from unittest import mock
from urllib import error

import pytest

from cve_synth.cli import _load_api_keys, _load_dotenv_files, parse_args
from cve_synth.checkpoint import CheckpointState, CheckpointStore
from cve_synth.extract import extract_evidence
from cve_synth.filtering import is_memory_corruption_candidate
from cve_synth.gemini_client import GeminiClient, GeminiConfig
from cve_synth.groq_client import GroqClient, GroqConfig
from cve_synth.models import AnalysisRecord, DatasetRecord, SourceRecord
from cve_synth.quality import is_acceptable, score_record
from cve_synth.rate_limit import MultiKeyRateLimiter
from cve_synth.writer import JsonlWriter


def make_record() -> DatasetRecord:
    source = SourceRecord(
        source_id="CVE-2026-0001",
        source_type="report",
        title="Example vuln",
        raw_text="""The following snippet is vulnerable:\n```c\nvoid f(char *p){ strcpy(buf,p); }\n```""",
        cve_id="CVE-2026-0001",
    )
    extraction = extract_evidence(source)
    analysis = AnalysisRecord(
        vulnerability_summary="Unsafely copies attacker-controlled data into a fixed buffer.",
        root_cause="Missing bounds checking before write.",
        reasoning_chain=["Identify attacker-controlled input", "Observe unchecked copy", "Confirm fixed-size destination buffer"],
        tags=["#MemoryCorruption"],
        vulnerable_snippet=extraction.vulnerable_snippet,
        assembly_fix="Replace the copy path with length-checked writes and preserve callee-saved state in the prologue/epilogue.",
        fix_strategy="Use bounded operations and validate lengths before memory writes.",
        confidence=0.9,
        model_name="deepseek-v4-pro-max",
        prompt_version="v1",
    )
    candidate = DatasetRecord(
        source=source,
        evidence_spans=extraction.evidence_spans,
        analysis=analysis,
        record_id="CVE-2026-0001:1",
        tags=analysis.tags,
        quality_score=0.0,
    )
    candidate.quality_score = score_record(candidate)
    return candidate


def test_rate_limiter_rotates_keys() -> None:
    limiter = MultiKeyRateLimiter(["a", "b"], min_interval_seconds=0.0, cooldown_seconds=0.0)
    assert limiter.acquire() == "a"
    assert limiter.acquire() == "b"


def test_checkpoint_round_trip(tmp_path) -> None:
    store = CheckpointStore(tmp_path / "checkpoint.json")
    state = CheckpointState(processed_source_ids={"one", "two"}, failed_source_ids={"three"}, last_output_index=7)
    store.save(state)
    loaded = store.load()
    assert loaded.processed_source_ids == {"one", "two"}
    assert loaded.failed_source_ids == {"three"}
    assert loaded.last_output_index == 7


def test_writer_appends_jsonl(tmp_path) -> None:
    record = make_record()
    writer = JsonlWriter(tmp_path / "dataset.jsonl")
    writer.append(record)
    payload = (tmp_path / "dataset.jsonl").read_text(encoding="utf-8").strip()
    decoded = json.loads(payload)
    assert decoded["record_id"] == "CVE-2026-0001:1"
    assert decoded["analysis"]["confidence"] == 0.9


def test_quality_gate_accepts_good_record() -> None:
    record = make_record()
    assert is_acceptable(record)


def test_load_api_keys_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEYS", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fallback-key")
    args = SimpleNamespace(api_keys=[], api_keys_file=None)
    assert _load_api_keys(args) == ["fallback-key"]


def test_load_dotenv_files_populates_env(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DEEPSEEK_API_KEY=dotenv-key\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _load_dotenv_files()
    args = SimpleNamespace(api_keys=[], api_keys_file=None)
    assert _load_api_keys(args) == ["dotenv-key"]


def test_parse_args_has_run_button_defaults() -> None:
    args = parse_args([])
    assert str(args.input_dir).replace("\\", "/") == "data/raw"
    assert str(args.output).replace("\\", "/") == "data/dataset.jsonl"
    assert str(args.checkpoint).replace("\\", "/") == "data/checkpoint.json"


def test_tags_are_serialized(tmp_path) -> None:
    record = make_record()
    writer = JsonlWriter(tmp_path / "dataset.jsonl")
    writer.append(record)
    decoded = json.loads((tmp_path / "dataset.jsonl").read_text(encoding="utf-8").strip())
    assert decoded["tags"] == ["#MemoryCorruption"]
    assert decoded["analysis"]["tags"] == ["#MemoryCorruption"]


def test_groq_request_sets_identity_headers() -> None:
    client = GroqClient(api_key="test-key", config=GroqConfig(user_agent="cve-synth/0.1.0"))

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices": [{"message": {"content": "{}"}}]}'

    captured_request: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured_request["req"] = req
        captured_request["timeout"] = timeout
        return FakeResponse()

    with mock.patch("cve_synth.groq_client.request.urlopen", side_effect=fake_urlopen):
        response = client._request_json("/chat/completions", {"model": "m", "messages": []})

    assert response["choices"][0]["message"]["content"] == "{}"
    assert captured_request["timeout"] == pytest.approx(120.0)
    headers = dict(captured_request["req"].header_items())
    assert headers["Accept"] == "application/json"
    assert headers["Content-type"] == "application/json"
    assert headers["User-agent"] == "cve-synth/0.1.0"
    assert headers["Authorization"] == "Bearer test-key"


def test_groq_request_surfaces_waf_block_hint() -> None:
    client = GroqClient(api_key="test-key")

    def fake_urlopen(req, timeout):
        raise error.HTTPError(
            url=req.full_url,
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=BytesIO(b"Request blocked by WAF"),
        )

    with mock.patch("cve_synth.groq_client.request.urlopen", side_effect=fake_urlopen):
        with pytest.raises(RuntimeError, match="possible WAF/bot detection"):
            client._request_json("/chat/completions", {"model": "m", "messages": []})


def test_memory_corruption_filter_matches_strong_signals() -> None:
    vulnerable = SourceRecord(
        source_id="CVE-2026-1111",
        source_type="report",
        title="Stack buffer overflow in parser",
        raw_text="The bug leads to memory corruption via a stack buffer overflow.",
        cve_id="CVE-2026-1111",
    )
    assert is_memory_corruption_candidate(vulnerable)

    benign = SourceRecord(
        source_id="CVE-2026-1112",
        source_type="report",
        title="Logic bug in parser",
        raw_text="The issue causes an incorrect error code but does not corrupt memory.",
        cve_id="CVE-2026-1112",
    )
    assert not is_memory_corruption_candidate(benign)


def test_gemini_request_uses_generate_content_shape() -> None:
    client = GeminiClient(api_key="test-key", config=GeminiConfig(model="gemini-2.5-pro"))

    inner_payload = json.dumps(
        {
            "vulnerability_summary": "x",
            "root_cause": "y",
            "reasoning_chain": ["a"],
            "assembly_fix": "b",
            "fix_strategy": "c",
            "confidence": 0.9,
            "tags": ["#MemoryCorruption"],
        }
    )
    outer_payload = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": inner_payload,
                            }
                        ]
                    }
                }
            ]
        }
    ).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return outer_payload

    captured_request: dict[str, object] = {}

    def fake_urlopen(req, timeout):
        captured_request["req"] = req
        captured_request["timeout"] = timeout
        return FakeResponse()

    source = SourceRecord(
        source_id="CVE-2026-1113",
        source_type="report",
        title="Stack buffer overflow in parser",
        raw_text="The bug leads to memory corruption via a stack buffer overflow.",
        cve_id="CVE-2026-1113",
    )
    extraction = extract_evidence(source)

    with mock.patch("cve_synth.gemini_client.request.urlopen", side_effect=fake_urlopen):
        response = client._request_json("/models/gemini-2.5-pro:generateContent", {"contents": []})

    assert response["candidates"][0]["content"]["parts"][0]["text"]
    assert captured_request["timeout"] == pytest.approx(120.0)
    headers = dict(captured_request["req"].header_items())
    assert headers["Accept"] == "application/json"
    assert headers["Content-type"] == "application/json"
