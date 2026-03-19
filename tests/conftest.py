from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from quantflow.core.config import (
    AppConfig,
    BinanceConfig,
    DatasetConfig,
    ExecutionConfig,
    FeaturesConfig,
    PathsConfig,
    PortfolioConfig,
    ReportingConfig,
    RiskConfig,
    StrategyConfig,
    ValidationConfig,
)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            repo_root=tmp_path,
            raw_dir=tmp_path / "raw",
            bronze_dir=tmp_path / "bronze",
            silver_dir=tmp_path / "silver",
            gold_dir=tmp_path / "gold",
            metadata_dir=tmp_path / "metadata",
            reports_dir=tmp_path / "reports",
        ),
        binance=BinanceConfig(
            market="spot",
            frequency="daily",
            verify_checksum=False,
            timeout_seconds=10,
        ),
        dataset=DatasetConfig(
            dataset_type="trades",
            symbols=["BTCUSDT"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 1),
        ),
        features=FeaturesConfig(
            bar_size="1s",
            feature_profile="order_flow_v1",
            rolling_window_bars=2,
            volatility_window_bars=2,
            range_window_bars=2,
            normalization_window_bars=2,
        ),
        strategy=StrategyConfig(
            name="cumulative_delta_breakout",
            position_size=1.0,
            params={},
        ),
        execution=ExecutionConfig(
            default_order_type="market",
            maker_fee_bps=1.0,
            taker_fee_bps=5.0,
            slippage_bps=2.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=10.0,
            latency_bars=1,
            market_participation_cap=0.25,
            limit_participation_cap=0.10,
            limit_offset_bps=1.0,
            max_order_lifetime_bars=3,
            allow_limit_orders=True,
        ),
        portfolio=PortfolioConfig(
            initial_cash=10_000.0,
            allow_short=True,
        ),
        risk=RiskConfig(
            max_holding_bars=0,
            stop_loss_bps=0.0,
            take_profit_bps=0.0,
            trailing_stop_bps=0.0,
            cooldown_bars_after_stop=0,
            risk_per_trade_fraction=0.0025,
            daily_loss_limit_r=2.0,
            max_open_positions=1,
        ),
        validation=ValidationConfig(
            train_bars=3,
            test_bars=2,
            step_bars=1,
            embargo_bars=1,
            bootstrap_resamples=100,
            bootstrap_block_bars=2,
            grid_search_max_points=0,
        ),
        reporting=ReportingConfig(run_name="test_run"),
    )
