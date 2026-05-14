from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .models import SourceRecord


@dataclass(slots=True)
class IngestedSource:
    path: Path
    record: SourceRecord


def load_sources(input_dir: str | Path) -> list[IngestedSource]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    sources: list[IngestedSource] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        sources.extend(_load_path(path))
    return sources


def _load_path(path: Path) -> list[IngestedSource]:
    if path.suffix.lower() == ".jsonl":
        records = []
        for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(IngestedSource(path=path, record=_record_from_payload(payload, path, index)))
        return records

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [IngestedSource(path=path, record=_record_from_payload(item, path, index)) for index, item in enumerate(payload, start=1)]
        return [IngestedSource(path=path, record=_record_from_payload(payload, path, 1))]

    text = path.read_text(encoding="utf-8")
    record = SourceRecord(
        source_id=path.stem,
        source_type="report" if path.suffix.lower() == ".md" else "advisory",
        title=path.stem.replace("_", " ").replace("-", " ").strip() or path.stem,
        raw_text=text,
        url=None,
        metadata={"path": str(path)},
    )
    return [IngestedSource(path=path, record=record)]


def _record_from_payload(payload: dict[str, object], path: Path, index: int) -> SourceRecord:
    source_id = str(payload.get("source_id") or payload.get("cve_id") or payload.get("ghsa_id") or f"{path.stem}-{index}")
    source_type = str(payload.get("source_type") or payload.get("type") or ("report" if payload.get("cve_id") else "advisory"))
    title = str(payload.get("title") or payload.get("summary") or source_id)
    raw_text = str(payload.get("raw_text") or payload.get("description") or payload.get("body") or json.dumps(payload, ensure_ascii=True))
    metadata = dict(payload.get("metadata") or {})
    metadata.setdefault("path", str(path))
    return SourceRecord(
        source_id=source_id,
        source_type=source_type,
        title=title,
        raw_text=raw_text,
        url=payload.get("url") if isinstance(payload.get("url"), str) else None,
        cve_id=payload.get("cve_id") if isinstance(payload.get("cve_id"), str) else None,
        ghsa_id=payload.get("ghsa_id") if isinstance(payload.get("ghsa_id"), str) else None,
        published_at=payload.get("published_at") if isinstance(payload.get("published_at"), str) else None,
        severity=payload.get("severity") if isinstance(payload.get("severity"), str) else None,
        affected_components=[str(item) for item in payload.get("affected_components", [])] if isinstance(payload.get("affected_components"), list) else [],
        metadata=metadata,
    )
