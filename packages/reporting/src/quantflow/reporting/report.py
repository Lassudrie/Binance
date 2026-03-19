from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from quantflow.backtest.engine import BacktestResult
from quantflow.backtest.portfolio import TradeRecord
from quantflow.utils.io import dump_json, ensure_directory, load_json
from quantflow.validation.alpha import AlphaScanResult
from quantflow.validation.bootstrap import BootstrapResult
from quantflow.validation.candidate_deep_dive import CandidateDeepDiveResult
from quantflow.validation.optimization import GridSearchResult
from quantflow.validation.order_flow_suite import OrderFlowSuiteResult
from quantflow.validation.sensitivity import SensitivityResult
from quantflow.validation.walkforward import WalkForwardResult


def _report_dir(base_dir: Path, run_name: str) -> Path:
    return ensure_directory(base_dir / run_name)


def _trades_frame(trades: list[TradeRecord]) -> pl.DataFrame:
    if not trades:
        return pl.DataFrame(
            {
                "symbol": [],
                "side": [],
                "quantity": [],
                "entry_time": [],
                "exit_time": [],
                "entry_price": [],
                "exit_price": [],
                "gross_pnl": [],
                "net_pnl": [],
                "fees": [],
                "holding_seconds": [],
                "signal_reason": [],
                "entry_reason": [],
                "exit_reason": [],
                "hour_of_day": [],
                "session": [],
                "vol_regime": [],
                "atr_entry": [],
                "stop_price": [],
                "target_price": [],
                "risk_amount": [],
                "r_multiple": [],
                "mae_bps": [],
                "mfe_bps": [],
            }
        )
    return pl.DataFrame([trade.to_dict() for trade in trades])


def _metrics_by_hour(trades: list[TradeRecord]) -> pl.DataFrame:
    frame = _trades_frame(trades)
    if frame.is_empty():
        return frame
    return (
        frame.group_by("hour_of_day")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.col("net_pnl").sum().alias("net_pnl"),
                pl.col("net_pnl").mean().alias("expectancy"),
            ]
        )
        .sort("hour_of_day")
    )


def _metrics_by_session(trades: list[TradeRecord]) -> pl.DataFrame:
    frame = _trades_frame(trades)
    if frame.is_empty():
        return frame
    return (
        frame.group_by("session")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.col("net_pnl").sum().alias("net_pnl"),
                pl.col("net_pnl").mean().alias("expectancy"),
            ]
        )
        .sort("session")
    )


def _metrics_by_reason(trades: list[TradeRecord]) -> pl.DataFrame:
    frame = _trades_frame(trades)
    if frame.is_empty():
        return frame
    return (
        frame.group_by("signal_reason")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.col("net_pnl").sum().alias("net_pnl"),
                pl.col("net_pnl").mean().alias("expectancy"),
            ]
        )
        .sort("trade_count", descending=True)
    )


def _symbol_summary(trades: list[TradeRecord]) -> pl.DataFrame:
    frame = _trades_frame(trades)
    if frame.is_empty():
        return pl.DataFrame(
            {
                "symbol": [],
                "trade_count": [],
                "net_pnl": [],
                "expectancy": [],
                "hit_ratio": [],
                "profit_factor": [],
            }
        )
    enriched = frame.with_columns(
        [
            pl.when(pl.col("net_pnl") > 0.0)
            .then(pl.col("net_pnl"))
            .otherwise(pl.lit(0.0))
            .alias("positive_net_pnl"),
            pl.when(pl.col("net_pnl") < 0.0)
            .then(pl.col("net_pnl").abs())
            .otherwise(pl.lit(0.0))
            .alias("negative_net_pnl_abs"),
        ]
    )
    return (
        enriched.group_by("symbol")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.col("net_pnl").sum().alias("net_pnl"),
                pl.col("net_pnl").mean().alias("expectancy"),
                (pl.col("net_pnl") > 0.0).mean().alias("hit_ratio"),
                pl.when(pl.col("negative_net_pnl_abs").sum() > 0.0)
                .then(pl.col("positive_net_pnl").sum() / pl.col("negative_net_pnl_abs").sum())
                .otherwise(None)
                .alias("profit_factor"),
            ]
        )
        .sort("symbol")
    )


def _markdown_report(
    result: BacktestResult,
    *,
    metrics_by_hour: pl.DataFrame,
    metrics_by_session: pl.DataFrame,
    metrics_by_reason: pl.DataFrame,
) -> str:
    lines = [
        f"# Backtest Report: {result.strategy_name}",
        "",
        "## Summary",
    ]
    for key, value in result.metrics.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(
        [
            "",
            "## Methodology Warnings",
            "- Signals are generated after the feature bar closes.",
            (
                "- Orders are executed no earlier than the next bar, "
                "with latency, slippage and fees."
            ),
            (
                "- Market execution uses a spread proxy because full order book "
                "history is not wired into the MVP."
            ),
            "",
            "## Trades By Hour",
        ]
    )
    if metrics_by_hour.is_empty():
        lines.append("- No closed trades.")
    else:
        for row in metrics_by_hour.iter_rows(named=True):
            lines.append(
                f"- hour={row['hour_of_day']}: trades={row['trade_count']}, "
                f"net_pnl={row['net_pnl']}, expectancy={row['expectancy']}"
            )

    lines.extend(["", "## Trades By Session"])
    if metrics_by_session.is_empty():
        lines.append("- No closed trades.")
    else:
        for row in metrics_by_session.iter_rows(named=True):
            lines.append(
                f"- session={row['session']}: trades={row['trade_count']}, "
                f"net_pnl={row['net_pnl']}, expectancy={row['expectancy']}"
            )

    lines.extend(["", "## Trades By Reason"])
    if metrics_by_reason.is_empty():
        lines.append("- No closed trades.")
    else:
        for row in metrics_by_reason.iter_rows(named=True):
            lines.append(
                f"- reason={row['signal_reason']}: trades={row['trade_count']}, "
                f"net_pnl={row['net_pnl']}, expectancy={row['expectancy']}"
            )

    return "\n".join(lines) + "\n"


def write_backtest_report(result: BacktestResult, reports_dir: Path, run_name: str) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "summary.json"
    equity_path = target_dir / "equity_curve.csv"
    trades_path = target_dir / "trades.csv"
    detailed_trades_path = target_dir / "trade_log_detailed.csv"
    hour_path = target_dir / "metrics_by_hour.csv"
    session_path = target_dir / "metrics_by_session.csv"
    reason_path = target_dir / "metrics_by_reason.csv"
    symbol_summary_path = target_dir / "symbol_summary.csv"
    markdown_path = target_dir / "report.md"

    dump_json(summary_path, result.metrics)
    pl.DataFrame(result.equity_curve).write_csv(equity_path)
    trades_frame = _trades_frame(result.trades)
    trades_frame.write_csv(trades_path)
    trades_frame.write_csv(detailed_trades_path)
    metrics_by_hour = _metrics_by_hour(result.trades)
    metrics_by_session = _metrics_by_session(result.trades)
    metrics_by_reason = _metrics_by_reason(result.trades)
    symbol_summary = _symbol_summary(result.trades)
    metrics_by_hour.write_csv(hour_path)
    metrics_by_session.write_csv(session_path)
    metrics_by_reason.write_csv(reason_path)
    symbol_summary.write_csv(symbol_summary_path)
    markdown_path.write_text(
        _markdown_report(
            result,
            metrics_by_hour=metrics_by_hour,
            metrics_by_session=metrics_by_session,
            metrics_by_reason=metrics_by_reason,
        ),
        encoding="utf-8",
    )
    return markdown_path


def write_walkforward_report(result: WalkForwardResult, reports_dir: Path, run_name: str) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "walkforward_summary.json"
    windows_path = target_dir / "walkforward_windows.csv"

    dump_json(summary_path, {"aggregate": result.aggregate, "windows": result.windows})
    pl.DataFrame(result.windows).write_csv(windows_path)
    return summary_path


def write_bootstrap_report(result: BootstrapResult, reports_dir: Path, run_name: str) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "bootstrap_summary.json"
    windows_path = target_dir / "bootstrap_windows.csv"
    samples_path = target_dir / "bootstrap_samples.csv"
    markdown_path = target_dir / "bootstrap_report.md"

    dump_json(
        summary_path,
        {
            "aggregate": result.aggregate,
            "windows": result.windows,
        },
    )
    pl.DataFrame(result.windows).write_csv(windows_path)
    pl.DataFrame(result.samples).write_csv(samples_path)

    lines = ["# Bootstrap Validation Report", "", "## Summary"]
    for key, value in result.aggregate.items():
        lines.append(f"- `{key}`: {value}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def write_order_flow_suite_report(
    result: OrderFlowSuiteResult,
    reports_dir: Path,
    run_name: str,
) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "suite_summary.json"
    cells_path = target_dir / "suite_summary.csv"
    symbols_path = target_dir / "suite_symbol_summary.csv"
    candidates_path = target_dir / "suite_ranked_candidates.csv"
    markdown_path = target_dir / "suite_report.md"

    dump_json(
        summary_path,
        {
            "aggregate": result.aggregate,
            "cells": result.cells,
        },
    )
    cell_rows = [
        {
            **row,
            "default_params": json.dumps(row["default_params"], sort_keys=True),
            "best_grid_params": json.dumps(row["best_grid_params"], sort_keys=True),
        }
        for row in result.cells
    ]
    pl.DataFrame(cell_rows).write_csv(cells_path)
    pl.DataFrame(result.symbol_rows).write_csv(symbols_path)
    candidate_rows = [
        {
            **row,
            "params": json.dumps(row["params"], sort_keys=True),
        }
        for row in result.ranked_candidates
    ]
    pl.DataFrame(candidate_rows).write_csv(candidates_path)

    lines = ["# Order Flow Suite Report", "", "## Summary"]
    for key, value in result.aggregate.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Candidates"])
    for row in candidate_rows[:10]:
        lines.append(
            f"- rank={row['rank']}, strategy={row['strategy_name']}, "
            f"bar_size={row['bar_size']}, execution={row['execution_mode']}, "
            f"grid_point={row['grid_point']}, "
            f"oos_mean={row['walkforward_mean_test_net_pnl']}, "
            f"positive_ratio={row['positive_test_window_ratio']}, "
            f"fragile={row['execution_fragile']}, params={row['params']}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def write_candidate_deep_dive_report(
    result: CandidateDeepDiveResult,
    reports_dir: Path,
    run_name: str,
) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "study_summary.json"
    scenario_matrix_path = target_dir / "scenario_matrix.csv"
    phase2_path = target_dir / "phase2_grid_results.csv"
    symbol_breakdown_path = target_dir / "symbol_breakdown.csv"
    trades_path = target_dir / "best_candidate_trades.csv"
    walkforward_path = target_dir / "best_candidate_walkforward.json"
    bootstrap_path = target_dir / "best_candidate_bootstrap.json"
    passive_sensitivity_path = target_dir / "best_candidate_passive_sensitivity.csv"
    market_sanity_path = target_dir / "market_sanity_check.json"
    markdown_path = target_dir / "report.md"

    dump_json(
        summary_path,
        {
            "aggregate": result.aggregate,
            "best_scenario": result.best_scenario,
            "phase1_scenarios": result.phase1_scenarios,
            "phase2_results": result.phase2_results,
            "best_symbol_rows": result.best_symbol_rows,
            "passive_sensitivity_aggregate": result.passive_sensitivity.aggregate,
            "market_sanity_check": result.market_sanity_check,
        },
    )
    pl.DataFrame(result.phase1_scenarios).write_csv(scenario_matrix_path)
    pl.DataFrame(result.phase2_results).write_csv(phase2_path)
    pl.DataFrame(result.best_symbol_rows).write_csv(symbol_breakdown_path)
    _trades_frame(result.best_backtest.trades).write_csv(trades_path)
    dump_json(
        walkforward_path,
        {
            "aggregate": result.best_walkforward.aggregate,
            "windows": result.best_walkforward.windows,
        },
    )
    dump_json(
        bootstrap_path,
        {
            "aggregate": result.best_bootstrap.aggregate,
            "windows": result.best_bootstrap.windows,
            "samples": result.best_bootstrap.samples,
        },
    )
    pl.DataFrame(result.passive_sensitivity.scenarios).write_csv(passive_sensitivity_path)
    dump_json(market_sanity_path, result.market_sanity_check)

    family_best: dict[str, dict[str, object]] = {}
    for row in result.phase1_scenarios:
        family = str(row["filter_family"])
        family_best.setdefault(family, row)

    lines = ["# Candidate Deep Dive Report", "", "## Summary"]
    for key, value in result.aggregate.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Best Scenario"])
    for key, value in result.best_scenario.items():
        lines.append(f"- `{key}`: {value}")

    lines.extend(["", "## Filter Ablation"])
    for family in ["baseline", "vol_only", "session_only", "combined"]:
        row = family_best.get(family)
        if row is None:
            continue
        lines.append(
            f"- family={family}: scenario={row['scenario_name']}, "
            f"oos_mean={row['walkforward_mean_test_net_pnl']}, "
            f"positive_ratio={row['positive_test_window_ratio']}, "
            f"turnover={row['full_sample_turnover_multiple']}"
        )

    lines.extend(["", "## Symbol Breakdown"])
    for row in result.best_symbol_rows:
        lines.append(
            f"- symbol={row['symbol']}: trades={row['trade_count']}, "
            f"net_pnl={row['net_pnl']}, expectancy={row['expectancy']}, "
            f"hit_ratio={row['hit_ratio']}"
        )

    lines.extend(["", "## Execution Verdict"])
    lines.append(f"- passive verdict: {result.aggregate['execution_verdict']}")
    lines.append(
        "- market sanity oos mean: "
        f"{result.market_sanity_check['walkforward_aggregate']['mean_test_net_pnl']}"
    )
    lines.append(f"- final verdict: {result.aggregate['final_verdict']}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def write_sensitivity_report(result: SensitivityResult, reports_dir: Path, run_name: str) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "sensitivity_summary.json"
    scenarios_path = target_dir / "sensitivity_scenarios.csv"
    markdown_path = target_dir / "sensitivity_report.md"

    dump_json(summary_path, {"aggregate": result.aggregate, "scenarios": result.scenarios})
    scenarios = pl.DataFrame(result.scenarios).sort(
        ["net_pnl", "latency_bars", "slippage_bps", "taker_fee_bps"]
    )
    scenarios.write_csv(scenarios_path)

    lines = ["# Execution Sensitivity Report", "", "## Summary"]
    for key, value in result.aggregate.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Worst Scenarios"])
    for row in scenarios.head(10).iter_rows(named=True):
        lines.append(
            f"- {row['scenario_name']}: net_pnl={row['net_pnl']}, fees={row['total_fees']}, "
            f"drawdown={row['max_drawdown']}, trades={row['trade_count']}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def write_grid_search_report(result: GridSearchResult, reports_dir: Path, run_name: str) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "grid_search_summary.json"
    windows_path = target_dir / "grid_search_windows.csv"
    markdown_path = target_dir / "grid_search_report.md"

    dump_json(
        summary_path,
        {
            "aggregate": result.aggregate,
            "parameter_grid": result.parameter_grid,
            "windows": result.windows,
        },
    )
    windows_records = [
        {
            **window,
            "selected_params": json.dumps(window["selected_params"], sort_keys=True),
        }
        for window in result.windows
    ]
    pl.DataFrame(windows_records).write_csv(windows_path)

    lines = ["# Grid Search Report", "", "## Summary"]
    for key, value in result.aggregate.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Most Selected Params"])
    lines.append(f"- `{json.dumps(result.aggregate['most_selected_params'], sort_keys=True)}`")
    lines.extend(["", "## Window Selection"])
    for row in windows_records[:10]:
        lines.append(
            f"- window={row['window_index']}: train={row['train_objective']}, "
            f"test={row['test_objective']}, params={row['selected_params']}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def write_alpha_scan_report(result: AlphaScanResult, reports_dir: Path, run_name: str) -> Path:
    target_dir = _report_dir(reports_dir, run_name)
    summary_path = target_dir / "alpha_scan_summary.json"
    candidates_path = target_dir / "alpha_scan_candidates.csv"
    windows_path = target_dir / "alpha_scan_windows.csv"
    markdown_path = target_dir / "alpha_scan_report.md"

    dump_json(
        summary_path,
        {
            "aggregate": result.aggregate,
            "features": result.features,
            "horizons": result.horizons,
            "candidates": result.candidates,
        },
    )
    pl.DataFrame(result.candidates).write_csv(candidates_path)
    pl.DataFrame(result.windows).write_csv(windows_path)

    lines = ["# Alpha Scan Report", "", "## Summary"]
    for key, value in result.aggregate.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Top Candidates"])
    for row in result.candidates[:10]:
        lines.append(
            f"- feature={row['feature_name']}, horizon={row['horizon_bars']}, "
            f"side={row['inferred_side']}, "
            f"directional_test_spread={row['mean_directional_test_spread']}, "
            f"test_ic={row['mean_test_ic']}, stability={row['sign_stability_ratio']}"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path


def rebuild_report(report_dir: Path) -> Path:
    summary = load_json(report_dir / "summary.json", default={})
    markdown_path = report_dir / "report.md"
    lines = ["# Backtest Report", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
