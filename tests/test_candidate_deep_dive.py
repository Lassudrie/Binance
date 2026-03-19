from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import polars as pl
from quantflow.validation.candidate_deep_dive import run_candidate_deep_dive


def _candidate_frame(symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT")) -> pl.DataFrame:
    start = datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC)
    sessions = ["asia", "europe", "us"] * 6
    cumulative = [
        -1.8,
        -0.05,
        -1.6,
        -0.05,
        -1.4,
        -0.05,
        -1.7,
        -0.05,
        -1.5,
        -0.05,
        -1.3,
        -0.05,
        -1.6,
        -0.05,
        -1.4,
        -0.05,
        -1.2,
        -0.05,
    ]
    vol_regimes = [0.7, 0.9, 1.1, 1.3, 0.8, 1.2] * 3

    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        symbol_offset = symbol_index * 0.2
        for index, cumulative_delta in enumerate(cumulative):
            close = 100.0 + symbol_offset + (index * 0.03)
            rows.append(
                {
                    "symbol": symbol,
                    "bar_start": start + timedelta(seconds=15 * index),
                    "bar_end": start + timedelta(seconds=15 * (index + 1)),
                    "open": close - 0.02,
                    "high": close + 0.15,
                    "low": close - 0.35,
                    "close": close,
                    "volume": 100.0,
                    "trade_count": 20,
                    "session": sessions[index],
                    "hour_of_day": 0,
                    "vol_regime": vol_regimes[index],
                    "cumulative_delta_rz": cumulative_delta,
                    "rolling_delta_rz": cumulative_delta,
                }
            )
    return pl.DataFrame(rows).sort(["bar_end", "symbol"])


def test_candidate_deep_dive_runs_expected_phases(app_config) -> None:
    config = replace(
        app_config,
        dataset=replace(
            app_config.dataset,
            symbols=["BTCUSDT", "ETHUSDT"],
        ),
        features=replace(
            app_config.features,
            bar_size="15s",
            feature_profile="cumdelta_reversion_study_15s_v1",
        ),
        strategy=replace(
            app_config.strategy,
            name="cumdelta_reversion_v1",
            params={
                "entry_z": 1.0,
                "exit_z": 0.1,
                "vol_regime_min": 0.0,
                "allowed_sessions": [],
            },
        ),
        portfolio=replace(app_config.portfolio, allow_short=False),
        validation=replace(
            app_config.validation,
            train_bars=5,
            test_bars=3,
            step_bars=3,
            embargo_bars=1,
            bootstrap_resamples=20,
            bootstrap_block_bars=2,
        ),
    )

    result = run_candidate_deep_dive(features=_candidate_frame(), config=config)

    assert len(result.phase1_scenarios) == 12
    assert len(result.phase2_results) == 27
    assert result.phase1_scenarios[0]["rank"] == 1
    assert result.phase2_results[0]["rank"] == 1
    assert result.passive_sensitivity.aggregate["scenario_count"] == 54
    assert result.market_sanity_check["params"]["entry_z"] in {1.0, 1.25, 1.5}
    assert result.aggregate["execution_verdict"] in {
        "execution_fragile",
        "passive_survives",
        "passive_not_positive",
    }
    assert result.aggregate["final_verdict"] in {"validated_edge", "research_candidate"}
