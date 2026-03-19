from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PathsConfig:
    repo_root: Path
    raw_dir: Path
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    metadata_dir: Path
    reports_dir: Path


@dataclass(slots=True)
class BinanceConfig:
    market: str
    frequency: str
    verify_checksum: bool
    timeout_seconds: int


@dataclass(slots=True)
class DatasetConfig:
    dataset_type: str
    symbols: list[str]
    start_date: date
    end_date: date


@dataclass(slots=True)
class FeaturesConfig:
    bar_size: str
    feature_profile: str
    rolling_window_bars: int
    volatility_window_bars: int
    range_window_bars: int
    normalization_window_bars: int


@dataclass(slots=True)
class StrategyConfig:
    name: str
    position_size: float
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionConfig:
    default_order_type: str
    maker_fee_bps: float
    taker_fee_bps: float
    slippage_bps: float
    fallback_half_spread_bps: float
    impact_bps_per_unit_participation: float
    latency_bars: int
    market_participation_cap: float
    limit_participation_cap: float
    limit_offset_bps: float
    max_order_lifetime_bars: int
    allow_limit_orders: bool


@dataclass(slots=True)
class PortfolioConfig:
    initial_cash: float
    allow_short: bool


@dataclass(slots=True)
class RiskConfig:
    max_holding_bars: int
    stop_loss_bps: float
    take_profit_bps: float
    trailing_stop_bps: float
    cooldown_bars_after_stop: int
    risk_per_trade_fraction: float
    daily_loss_limit_r: float
    max_open_positions: int


@dataclass(slots=True)
class ValidationConfig:
    train_bars: int
    test_bars: int
    step_bars: int
    embargo_bars: int
    bootstrap_resamples: int
    bootstrap_block_bars: int
    grid_search_max_points: int


@dataclass(slots=True)
class ReportingConfig:
    run_name: str


@dataclass(slots=True)
class AppConfig:
    paths: PathsConfig
    binance: BinanceConfig
    dataset: DatasetConfig
    features: FeaturesConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    portfolio: PortfolioConfig
    risk: RiskConfig
    validation: ValidationConfig
    reporting: ReportingConfig

    @property
    def primary_symbol(self) -> str:
        return self.dataset.symbols[0]


def _resolve_repo_root(config_path: Path) -> Path:
    parent = config_path.resolve().parent
    if parent.name == "configs":
        return parent.parent
    return parent


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def load_config(config_path: str | Path) -> AppConfig:
    path = Path(config_path)
    with path.open("rb") as file_obj:
        raw = tomllib.load(file_obj)

    repo_root = _resolve_repo_root(path)
    paths = raw["paths"]

    return AppConfig(
        paths=PathsConfig(
            repo_root=repo_root,
            raw_dir=_resolve_path(repo_root, paths["raw_dir"]),
            bronze_dir=_resolve_path(repo_root, paths["bronze_dir"]),
            silver_dir=_resolve_path(repo_root, paths["silver_dir"]),
            gold_dir=_resolve_path(repo_root, paths["gold_dir"]),
            metadata_dir=_resolve_path(repo_root, paths["metadata_dir"]),
            reports_dir=_resolve_path(repo_root, paths["reports_dir"]),
        ),
        binance=BinanceConfig(
            market=raw["binance"]["market"],
            frequency=raw["binance"]["frequency"],
            verify_checksum=bool(raw["binance"]["verify_checksum"]),
            timeout_seconds=int(raw["binance"]["timeout_seconds"]),
        ),
        dataset=DatasetConfig(
            dataset_type=raw["dataset"]["dataset_type"],
            symbols=list(raw["dataset"]["symbols"]),
            start_date=_parse_date(raw["dataset"]["start_date"]),
            end_date=_parse_date(raw["dataset"]["end_date"]),
        ),
        features=FeaturesConfig(
            bar_size=raw["features"]["bar_size"],
            feature_profile=raw["features"].get("feature_profile", "order_flow_v1"),
            rolling_window_bars=int(raw["features"]["rolling_window_bars"]),
            volatility_window_bars=int(raw["features"]["volatility_window_bars"]),
            range_window_bars=int(raw["features"]["range_window_bars"]),
            normalization_window_bars=int(raw["features"]["normalization_window_bars"]),
        ),
        strategy=StrategyConfig(
            name=raw["strategy"]["name"],
            position_size=float(raw["strategy"]["position_size"]),
            params=dict(raw["strategy"].get("params", {})),
        ),
        execution=ExecutionConfig(
            default_order_type=raw["execution"].get("default_order_type", "market"),
            maker_fee_bps=float(raw["execution"]["maker_fee_bps"]),
            taker_fee_bps=float(raw["execution"]["taker_fee_bps"]),
            slippage_bps=float(raw["execution"]["slippage_bps"]),
            fallback_half_spread_bps=float(raw["execution"]["fallback_half_spread_bps"]),
            impact_bps_per_unit_participation=float(
                raw["execution"].get("impact_bps_per_unit_participation", 0.0)
            ),
            latency_bars=int(raw["execution"]["latency_bars"]),
            market_participation_cap=float(raw["execution"].get("market_participation_cap", 1.0)),
            limit_participation_cap=float(raw["execution"].get("limit_participation_cap", 0.1)),
            limit_offset_bps=float(raw["execution"].get("limit_offset_bps", 0.0)),
            max_order_lifetime_bars=int(raw["execution"].get("max_order_lifetime_bars", 1)),
            allow_limit_orders=bool(raw["execution"]["allow_limit_orders"]),
        ),
        portfolio=PortfolioConfig(
            initial_cash=float(raw["portfolio"]["initial_cash"]),
            allow_short=bool(raw["portfolio"]["allow_short"]),
        ),
        risk=RiskConfig(
            max_holding_bars=int(raw.get("risk", {}).get("max_holding_bars", 0)),
            stop_loss_bps=float(raw.get("risk", {}).get("stop_loss_bps", 0.0)),
            take_profit_bps=float(raw.get("risk", {}).get("take_profit_bps", 0.0)),
            trailing_stop_bps=float(raw.get("risk", {}).get("trailing_stop_bps", 0.0)),
            cooldown_bars_after_stop=int(raw.get("risk", {}).get("cooldown_bars_after_stop", 0)),
            risk_per_trade_fraction=float(
                raw.get("risk", {}).get("risk_per_trade_fraction", 0.0)
            ),
            daily_loss_limit_r=float(raw.get("risk", {}).get("daily_loss_limit_r", 0.0)),
            max_open_positions=int(raw.get("risk", {}).get("max_open_positions", 1)),
        ),
        validation=ValidationConfig(
            train_bars=int(raw["validation"]["train_bars"]),
            test_bars=int(raw["validation"]["test_bars"]),
            step_bars=int(raw["validation"]["step_bars"]),
            embargo_bars=int(raw["validation"]["embargo_bars"]),
            bootstrap_resamples=int(raw["validation"].get("bootstrap_resamples", 1_000)),
            bootstrap_block_bars=int(raw["validation"].get("bootstrap_block_bars", 144)),
            grid_search_max_points=int(raw["validation"].get("grid_search_max_points", 0)),
        ),
        reporting=ReportingConfig(run_name=raw["reporting"]["run_name"]),
    )
