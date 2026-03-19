from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from quantflow.utils.dates import parse_date
from quantflow.utils.io import ensure_directory

BASE_URL = "https://data.binance.vision"


@dataclass(slots=True)
class DownloadedArtifact:
    symbol: str
    dataset_type: str
    market: str
    frequency: str
    source_date: date
    path: Path
    sha256: str
    checksum_verified: bool
    checksum_value: str | None


def _date_token(source_date: date, frequency: str) -> str:
    if frequency == "daily":
        return source_date.isoformat()
    if frequency == "monthly":
        return source_date.strftime("%Y-%m")
    raise ValueError(f"Unsupported Binance frequency: {frequency}")


def build_relative_path(
    *,
    market: str,
    frequency: str,
    dataset_type: str,
    symbol: str,
    source_date: date,
    interval: str | None = None,
) -> str:
    base = f"data/{market}/{frequency}/{dataset_type}/{symbol}"
    token = _date_token(source_date, frequency)
    if dataset_type == "klines":
        if interval is None:
            raise ValueError("Klines require an interval")
        return f"{base}/{interval}/{symbol}-{interval}-{token}.zip"
    return f"{base}/{symbol}-{dataset_type}-{token}.zip"


def build_data_url(**kwargs: str | date | None) -> str:
    relative_path = build_relative_path(
        market=str(kwargs["market"]),
        frequency=str(kwargs["frequency"]),
        dataset_type=str(kwargs["dataset_type"]),
        symbol=str(kwargs["symbol"]),
        source_date=parse_date(kwargs["source_date"]),
        interval=kwargs.get("interval") if isinstance(kwargs.get("interval"), str) else None,
    )
    return f"{BASE_URL}/{relative_path}"


def build_checksum_url(**kwargs: str | date | None) -> str:
    return f"{build_data_url(**kwargs)}.CHECKSUM"


def download_bytes(url: str, timeout_seconds: int) -> bytes:
    with urlopen(url, timeout=timeout_seconds) as response:
        return response.read()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_remote_checksum(url: str, timeout_seconds: int) -> str | None:
    try:
        payload = download_bytes(url, timeout_seconds=timeout_seconds).decode("utf-8").strip()
    except HTTPError as error:
        if error.code == 404:
            return None
        raise

    if not payload:
        return None
    return payload.split()[0]


def download_dataset_file(
    *,
    raw_dir: Path,
    market: str,
    frequency: str,
    dataset_type: str,
    symbol: str,
    source_date: date,
    timeout_seconds: int,
    verify_checksum: bool,
    interval: str | None = None,
) -> DownloadedArtifact:
    relative_path = build_relative_path(
        market=market,
        frequency=frequency,
        dataset_type=dataset_type,
        symbol=symbol,
        source_date=source_date,
        interval=interval,
    )
    target_path = raw_dir / relative_path.replace("data/", "", 1)
    ensure_directory(target_path.parent)

    if not target_path.exists():
        url = f"{BASE_URL}/{relative_path}"
        target_path.write_bytes(download_bytes(url, timeout_seconds=timeout_seconds))

    checksum_value = None
    checksum_verified = False
    local_sha256 = file_sha256(target_path)

    if verify_checksum:
        checksum_url = f"{BASE_URL}/{relative_path}.CHECKSUM"
        checksum_value = fetch_remote_checksum(checksum_url, timeout_seconds=timeout_seconds)
        if checksum_value is not None and checksum_value != local_sha256:
            raise ValueError(
                "Checksum mismatch for "
                f"{target_path.name}: expected {checksum_value}, got {local_sha256}"
            )
        checksum_verified = checksum_value is not None

    return DownloadedArtifact(
        symbol=symbol,
        dataset_type=dataset_type,
        market=market,
        frequency=frequency,
        source_date=source_date,
        path=target_path,
        sha256=local_sha256,
        checksum_verified=checksum_verified,
        checksum_value=checksum_value,
    )
