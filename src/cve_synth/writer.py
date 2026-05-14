from __future__ import annotations

from pathlib import Path
import json

from .models import DatasetRecord


class JsonlWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DatasetRecord) -> None:
        line = json.dumps(record.to_jsonl(), ensure_ascii=True, sort_keys=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            try:
                import os

                os.fsync(handle.fileno())
            except OSError:
                pass
