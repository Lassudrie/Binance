from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from quantflow.cli.main import _gold_feature_path, _load_gold_features, _silver_partition_path
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


def _build_config(tmp_path: Path) -> AppConfig:
    paths = PathsConfig(
        repo_root=tmp_path,
        raw_dir=tmp_path / "raw",
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        gold_dir=tmp_path / "gold",
        metadata_dir=tmp_path / "metadata",
        reports_dir=tmp_path / "reports",
    )
    return AppConfig(
        paths=paths,
        binance=BinanceConfig(
            market="spot",
            frequency="daily",
            verify_checksum=True,
            timeout_seconds=30,
        ),
        dataset=DatasetConfig(
            dataset_type="trades",
            symbols=["BTCUSDT"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 2),
        ),
        features=FeaturesConfig(
            bar_size="5s",
            feature_profile="order_flow_suite_5s_v1",
            rolling_window_bars=24,
            volatility_window_bars=48,
            range_window_bars=24,
            normalization_window_bars=96,
        ),
        strategy=StrategyConfig(name="cumdelta_reversion_v1", position_size=0.01, params={}),
        execution=ExecutionConfig(
            default_order_type="market",
            maker_fee_bps=1.0,
            taker_fee_bps=5.0,
            slippage_bps=2.0,
            fallback_half_spread_bps=1.0,
            impact_bps_per_unit_participation=15.0,
            latency_bars=1,
            market_participation_cap=0.25,
            limit_participation_cap=0.10,
            limit_offset_bps=1.0,
            max_order_lifetime_bars=3,
            allow_limit_orders=True,
        ),
        portfolio=PortfolioConfig(initial_cash=100000.0, allow_short=False),
        risk=RiskConfig(
            max_holding_bars=0,
            stop_loss_bps=0.0,
            take_profit_bps=0.0,
            trailing_stop_bps=0.0,
            cooldown_bars_after_stop=0,
            risk_per_trade_fraction=0.0,
            daily_loss_limit_r=0.0,
            max_open_positions=1,
        ),
        validation=ValidationConfig(
            train_bars=10,
            test_bars=5,
            step_bars=5,
            embargo_bars=1,
            bootstrap_resamples=10,
            bootstrap_block_bars=2,
            grid_search_max_points=0,
        ),
        reporting=ReportingConfig(run_name="test"),
    )


def test_load_gold_features_rebuilds_when_silver_is_newer(monkeypatch, tmp_path: Path) -> None:
    config = _build_config(tmp_path)
    symbol = "BTCUSDT"
    dataset_type = "trades"

    for source_date in (date(2025, 1, 1), date(2025, 1, 2)):
        silver_path = _silver_partition_path(config, dataset_type, symbol, source_date)
        silver_path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "bar_end": [source_date.isoformat()],
                "symbol": [symbol],
                "value": [1.0],
            }
        ).write_parquet(silver_path)

    gold_path = _gold_feature_path(config, symbol, dataset_type)
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"bar_end": ["2025-01-01T00:00:00"], "symbol": [symbol], "value": [0.0]}).write_parquet(
        gold_path
    )

    old_timestamp = 1_700_000_000
    new_timestamp = 1_700_000_100
    gold_path.touch()
    gold_path.chmod(0o644)
    import os

    os.utime(gold_path, (old_timestamp, old_timestamp))
    latest_silver = _silver_partition_path(config, dataset_type, symbol, date(2025, 1, 2))
    os.utime(latest_silver, (new_timestamp, new_timestamp))

    rebuild_calls: list[tuple[str, str]] = []

    def _fake_build_feature_dataset(cfg, requested_symbol: str, *, dataset_type: str):
        rebuild_calls.append((requested_symbol, dataset_type))
        rebuilt_path = _gold_feature_path(cfg, requested_symbol, dataset_type)
        pl.DataFrame(
            {
                "bar_end": ["2025-01-02T00:00:00"],
                "symbol": [requested_symbol],
                "value": [2.0],
            }
        ).write_parquet(rebuilt_path)
        return rebuilt_path

    monkeypatch.setattr("quantflow.cli.main.build_feature_dataset", _fake_build_feature_dataset)
    monkeypatch.setattr(
        "quantflow.cli.main.build_dataset_range",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected dataset rebuild")),
    )

    frame = _load_gold_features(config, [symbol], dataset_type=dataset_type)

    assert rebuild_calls == [(symbol, dataset_type)]
    assert frame.select("value").to_series().to_list() == [2.0]
