from __future__ import annotations

import json
from types import SimpleNamespace

from cve_synth.cli import _load_api_keys, _load_dotenv_files, parse_args
from cve_synth.checkpoint import CheckpointState, CheckpointStore
from cve_synth.extract import extract_evidence
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
