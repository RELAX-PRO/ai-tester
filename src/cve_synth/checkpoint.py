from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
import json


@dataclass(slots=True)
class CheckpointState:
    processed_source_ids: set[str] = field(default_factory=set)
    failed_source_ids: set[str] = field(default_factory=set)
    last_output_index: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "processed_source_ids": sorted(self.processed_source_ids),
            "failed_source_ids": sorted(self.failed_source_ids),
            "last_output_index": self.last_output_index,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "CheckpointState":
        return cls(
            processed_source_ids=set(payload.get("processed_source_ids", [])),
            failed_source_ids=set(payload.get("failed_source_ids", [])),
            last_output_index=int(payload.get("last_output_index", 0)),
        )


class CheckpointStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> CheckpointState:
        if not self.path.exists():
            return CheckpointState()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        return CheckpointState.from_dict(payload)

    def save(self, state: CheckpointState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=self.path.parent, prefix=self.path.name, suffix=".tmp") as temp_file:
            json.dump(state.to_dict(), temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        temp_path.replace(self.path)
