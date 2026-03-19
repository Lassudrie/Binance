from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta


def parse_date(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def iter_dates(start_date: date, end_date: date) -> Iterator[date]:
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def session_from_hour(hour: int) -> str:
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "europe"
    if 13 <= hour < 21:
        return "us"
    return "off_hours"
