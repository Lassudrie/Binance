from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any


class ResearchRegistry:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiments_path = self.root / "experiments.jsonl"
        self.candidates_path = self.root / "candidates.jsonl"

    def record_experiment(self, payload: dict[str, Any]) -> None:
        self._append(self.experiments_path, {"record_type": "experiment", **payload})

    def record_candidate(self, payload: dict[str, Any]) -> None:
        self._append(self.candidates_path, {"record_type": "candidate", **payload})

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        latest: dict[str, Any] | None = None
        for row in self._read_jsonl(self.candidates_path):
            if str(row.get("candidate_id")) != candidate_id:
                continue
            latest = row
        return latest

    def list_candidates(self, *, status: str | None = None) -> list[dict[str, Any]]:
        latest_by_id: dict[str, dict[str, Any]] = {}
        for row in self._read_jsonl(self.candidates_path):
            candidate_id = str(row.get("candidate_id") or "")
            if not candidate_id:
                continue
            latest_by_id[candidate_id] = row
        rows = list(latest_by_id.values())
        if status is not None:
            rows = [row for row in rows if row.get("status") == status]
        rows.sort(key=lambda row: str(row.get("updated_at") or ""), reverse=True)
        return rows

    def _append(self, path: Path, payload: dict[str, Any]) -> None:
        record = {
            **payload,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record, sort_keys=True) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
