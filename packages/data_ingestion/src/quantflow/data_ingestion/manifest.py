from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from quantflow.utils.io import dump_json, ensure_directory, load_json


@dataclass(slots=True)
class IngestionRecord:
    dataset_type: str
    symbol: str
    market: str
    source_date: date
    raw_path: Path
    bronze_path: Path | None
    silver_path: Path | None
    row_count: int
    timestamp_unit: str
    sha256: str
    checksum_verified: bool
    checksum_value: str | None
    created_at: datetime

    def key(self) -> str:
        return f"{self.market}:{self.dataset_type}:{self.symbol}:{self.source_date.isoformat()}"

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "dataset_type": self.dataset_type,
            "symbol": self.symbol,
            "market": self.market,
            "source_date": self.source_date.isoformat(),
            "raw_path": str(self.raw_path),
            "bronze_path": None if self.bronze_path is None else str(self.bronze_path),
            "silver_path": None if self.silver_path is None else str(self.silver_path),
            "row_count": self.row_count,
            "timestamp_unit": self.timestamp_unit,
            "sha256": self.sha256,
            "checksum_verified": self.checksum_verified,
            "checksum_value": self.checksum_value,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, str | int | bool | None]) -> IngestionRecord:
        return cls(
            dataset_type=str(payload["dataset_type"]),
            symbol=str(payload["symbol"]),
            market=str(payload["market"]),
            source_date=date.fromisoformat(str(payload["source_date"])),
            raw_path=Path(str(payload["raw_path"])),
            bronze_path=None
            if payload["bronze_path"] is None
            else Path(str(payload["bronze_path"])),
            silver_path=None
            if payload["silver_path"] is None
            else Path(str(payload["silver_path"])),
            row_count=int(payload["row_count"]),
            timestamp_unit=str(payload["timestamp_unit"]),
            sha256=str(payload["sha256"]),
            checksum_verified=bool(payload["checksum_verified"]),
            checksum_value=None
            if payload["checksum_value"] is None
            else str(payload["checksum_value"]),
            created_at=datetime.fromisoformat(str(payload["created_at"])),
        )


class ManifestRepository:
    def __init__(self, metadata_dir: Path) -> None:
        ensure_directory(metadata_dir)
        self.path = metadata_dir / "ingestion_manifest.json"

    def load(self) -> dict[str, IngestionRecord]:
        raw_records: list[dict[str, str | int | bool | None]] = load_json(self.path, default=[])
        return {
            IngestionRecord.from_dict(item).key(): IngestionRecord.from_dict(item)
            for item in raw_records
        }

    def upsert(self, record: IngestionRecord) -> None:
        records = self.load()
        records[record.key()] = record
        serialized = [
            item.to_dict() for item in sorted(records.values(), key=lambda value: value.key())
        ]
        dump_json(self.path, serialized)
