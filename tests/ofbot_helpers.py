from __future__ import annotations

from datetime import UTC, datetime

from ofbot.config import (
    AppConfig,
    ExecutionConfig,
    FeaturesConfig,
    GatewayConfig,
    LearningConfig,
    PathsConfig,
    RiskKillSwitchConfig,
    StrategyModeConfig,
    PolicyLibraryConfig,
)


def build_test_config(tmp_path, *, symbols: list[str] | None = None, mode: str = "paper_local") -> AppConfig:
    config = AppConfig(
        mode=mode,
        symbols=symbols or ["BTCUSDT"],
        paths=PathsConfig(
            root=tmp_path,
            raw_dir=tmp_path / "raw",
            reports_dir=tmp_path / "reports",
            memory_dir=tmp_path / "memory",
        ),
        gateway=GatewayConfig(),
        features=FeaturesConfig(),
        execution=ExecutionConfig(mode=mode, allow_short=False),
        risk=RiskKillSwitchConfig(),
        strategy_library=PolicyLibraryConfig(),
        learning=LearningConfig(duckdb_path=tmp_path / "memory" / "learning.duckdb"),
        paper_local_record_raw=False,
        seed=42,
    )
    return config


def utc_time(offset_seconds: int) -> datetime:
    return datetime.fromtimestamp(offset_seconds, tz=UTC)
