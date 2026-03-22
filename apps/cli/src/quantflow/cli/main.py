from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl
from quantflow.backtest.engine import run_backtest
from quantflow.core import configure_logging, load_config
from quantflow.data_ingestion import build_dataset_range, download_dataset_range
from quantflow.data_ingestion.pipeline import load_silver_dataset
from quantflow.feature_engineering import build_feature_dataset
from quantflow.reporting import (
    rebuild_report,
    write_alpha_scan_report,
    write_backtest_report,
    write_bootstrap_report,
    write_candidate_deep_dive_report,
    write_grid_search_report,
    write_order_flow_suite_report,
    write_sensitivity_report,
    write_walkforward_report,
)
from quantflow.strategy import build_strategy
from quantflow.utils.dates import iter_dates
from quantflow.validation import (
    FEATURE_PROFILES,
    expand_param_grid,
    parse_param_grid_specs,
    run_alpha_scan,
    run_bootstrap_validation,
    run_candidate_deep_dive,
    run_execution_sensitivity,
    run_order_flow_suite,
    run_walk_forward,
    run_walk_forward_grid_search,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="qflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--config", required=True)
        command_parser.add_argument("--symbol")
        command_parser.add_argument("--start-date")
        command_parser.add_argument("--end-date")
        command_parser.add_argument("--dataset-type", choices=["trades", "aggTrades", "klines"])

    download_parser = subparsers.add_parser("download-data")
    add_common_arguments(download_parser)

    build_parser = subparsers.add_parser("build-dataset")
    add_common_arguments(build_parser)

    feature_parser = subparsers.add_parser("build-features")
    feature_parser.add_argument("--config", required=True)
    feature_parser.add_argument("--symbol")
    feature_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    backtest_parser = subparsers.add_parser("run-backtest")
    backtest_parser.add_argument("--config", required=True)
    backtest_parser.add_argument("--symbol")
    backtest_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    walkforward_parser = subparsers.add_parser("run-walkforward")
    walkforward_parser.add_argument("--config", required=True)
    walkforward_parser.add_argument("--symbol")
    walkforward_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    bootstrap_parser = subparsers.add_parser("run-bootstrap")
    bootstrap_parser.add_argument("--config", required=True)
    bootstrap_parser.add_argument("--symbol")
    bootstrap_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    grid_parser = subparsers.add_parser("run-grid-search")
    grid_parser.add_argument("--config", required=True)
    grid_parser.add_argument("--symbol")
    grid_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])
    grid_parser.add_argument(
        "--param-grid",
        action="append",
        default=[],
        help="Format: key=v1,v2,v3",
    )
    grid_parser.add_argument("--objective", default="net_pnl")

    sensitivity_parser = subparsers.add_parser("run-sensitivity")
    sensitivity_parser.add_argument("--config", required=True)
    sensitivity_parser.add_argument("--symbol")
    sensitivity_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    alpha_parser = subparsers.add_parser("run-alpha-scan")
    alpha_parser.add_argument("--config", required=True)
    alpha_parser.add_argument("--symbol")
    alpha_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])
    alpha_parser.add_argument("--feature", action="append", default=[])
    alpha_parser.add_argument("--horizon-bars", action="append", type=int, default=[])

    suite_parser = subparsers.add_parser("run-order-flow-suite")
    suite_parser.add_argument("--config", required=True)
    suite_parser.add_argument("--symbol")
    suite_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    deep_dive_parser = subparsers.add_parser("run-candidate-deep-dive")
    deep_dive_parser.add_argument("--config", required=True)
    deep_dive_parser.add_argument("--symbol")
    deep_dive_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])

    inspect_parser = subparsers.add_parser("inspect-dataset")
    inspect_parser.add_argument("--config", required=True)
    inspect_parser.add_argument("--symbol")
    inspect_parser.add_argument("--dataset-type", choices=["trades", "aggTrades", "klines"])
    inspect_parser.add_argument(
        "--layer", choices=["raw", "bronze", "silver", "gold"], default="silver"
    )

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--config", required=True)
    report_parser.add_argument("--dataset-type", choices=["trades", "aggTrades"])
    report_parser.add_argument("--run-dir")

    return parser.parse_args()


def _resolve_symbol(config: Any, symbol: str | None) -> str:
    return symbol or config.primary_symbol


def _resolve_symbols(config: Any, symbol: str | None) -> list[str]:
    if symbol is not None:
        return [symbol]
    return list(config.dataset.symbols)


def _resolve_date(value: str | None, fallback: date) -> date:
    return fallback if value is None else date.fromisoformat(value)


def _resolve_dataset_type(config: Any, dataset_type: str | None) -> str:
    return dataset_type or config.dataset.dataset_type


def _resolve_run_name(config: Any, dataset_type: str | None) -> str:
    resolved_dataset_type = _resolve_dataset_type(config, dataset_type)
    return f"{config.reporting.run_name}_{resolved_dataset_type}"


def _build_strategy_from_params(config: Any, params: dict[str, Any]) -> Any:
    merged_params = {**config.strategy.params, **params}
    strategy_config = replace(config.strategy, params=merged_params)
    return build_strategy(strategy_config)


def _gold_feature_path(config: Any, symbol: str, dataset_type: str) -> Path:
    return (
        config.paths.gold_dir
        / "features"
        / f"dataset_type={dataset_type}"
        / f"bar_size={config.features.bar_size}"
        / f"feature_profile={config.features.feature_profile}"
        / f"symbol={symbol}"
        / "part-0.parquet"
    )


def _feature_variant_config(config: Any, *, bar_size: str, feature_profile: str) -> Any:
    return replace(
        config,
        features=replace(
            config.features,
            bar_size=bar_size,
            feature_profile=feature_profile,
        ),
    )


def _silver_partition_path(config: Any, dataset_type: str, symbol: str, source_date: date) -> Path:
    return (
        config.paths.silver_dir
        / dataset_type
        / f"symbol={symbol}"
        / f"date={source_date.isoformat()}"
        / "part-0.parquet"
    )


def _ensure_silver_dataset_range(config: Any, symbol: str, dataset_type: str) -> None:
    expected_partitions = [
        _silver_partition_path(config, dataset_type, symbol, source_date)
        for source_date in iter_dates(config.dataset.start_date, config.dataset.end_date)
    ]
    if all(path.exists() for path in expected_partitions):
        return
    build_dataset_range(
        config,
        symbol=symbol,
        start_date=config.dataset.start_date,
        end_date=config.dataset.end_date,
        dataset_type=dataset_type,
    )


def _gold_features_need_rebuild(
    config: Any,
    *,
    dataset_type: str,
    symbol: str,
    path: Path,
) -> bool:
    if not path.exists():
        return True

    gold_mtime = path.stat().st_mtime
    for source_date in iter_dates(config.dataset.start_date, config.dataset.end_date):
        silver_path = _silver_partition_path(config, dataset_type, symbol, source_date)
        if not silver_path.exists():
            return True
        if silver_path.stat().st_mtime > gold_mtime:
            return True
    return False


def _load_gold_features(
    config: Any,
    symbols: list[str],
    dataset_type: str | None = None,
    *,
    force_rebuild: bool = False,
) -> pl.DataFrame:
    resolved_dataset_type = _resolve_dataset_type(config, dataset_type)
    frames: list[pl.DataFrame] = []
    for symbol in symbols:
        _ensure_silver_dataset_range(config, symbol, resolved_dataset_type)
        path = _gold_feature_path(config, symbol, resolved_dataset_type)
        if force_rebuild or _gold_features_need_rebuild(
            config,
            dataset_type=resolved_dataset_type,
            symbol=symbol,
            path=path,
        ):
            path = build_feature_dataset(config, symbol, dataset_type=resolved_dataset_type)
        frames.append(pl.read_parquet(path))
    return pl.concat(frames, how="vertical").sort(["bar_end", "symbol"])


def _load_order_flow_suite_features(
    config: Any,
    symbols: list[str],
    dataset_type: str,
) -> dict[str, pl.DataFrame]:
    frames: dict[str, pl.DataFrame] = {}
    for bar_size, feature_profile in FEATURE_PROFILES.items():
        variant_config = _feature_variant_config(
            config,
            bar_size=bar_size,
            feature_profile=feature_profile,
        )
        frames[bar_size] = _load_gold_features(
            variant_config,
            symbols,
            dataset_type=dataset_type,
        )
    return frames


def _load_candidate_deep_dive_features(
    config: Any,
    symbols: list[str],
    dataset_type: str,
) -> pl.DataFrame:
    return _load_gold_features(
        config,
        symbols,
        dataset_type=dataset_type,
        force_rebuild=True,
    )


def _list_layer_files(config: Any, layer: str, dataset_type: str, symbol: str) -> list[Path]:
    if layer == "raw":
        base = (
            config.paths.raw_dir
            / config.binance.market
            / config.binance.frequency
            / dataset_type
            / symbol
        )
    elif layer in {"bronze", "silver"}:
        base = getattr(config.paths, f"{layer}_dir") / dataset_type / f"symbol={symbol}"
    elif layer == "gold":
        base = _gold_feature_path(config, symbol, dataset_type).parent
    else:
        raise ValueError(f"Unsupported layer: {layer}")
    if not base.exists():
        return []
    return sorted(base.rglob("*"))


def main() -> None:
    args = _parse_args()
    configure_logging()
    config = load_config(args.config)

    if args.command == "download-data":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        start_date = _resolve_date(args.start_date, config.dataset.start_date)
        end_date = _resolve_date(args.end_date, config.dataset.end_date)
        for symbol in symbols:
            artifacts = download_dataset_range(
                config,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                dataset_type=dataset_type,
            )
            for artifact in artifacts:
                print(artifact.path)
        return

    if args.command == "build-dataset":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        start_date = _resolve_date(args.start_date, config.dataset.start_date)
        end_date = _resolve_date(args.end_date, config.dataset.end_date)
        for symbol in symbols:
            records = build_dataset_range(
                config,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                dataset_type=dataset_type,
            )
            for record in records:
                print(record.silver_path)
        return

    if args.command == "build-features":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        for symbol in symbols:
            output_path = build_feature_dataset(config, symbol, dataset_type=dataset_type)
            print(output_path)
        return

    if args.command == "run-backtest":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_gold_features(config, symbols, dataset_type=dataset_type)
        result = run_backtest(features, build_strategy(config.strategy), config)
        run_name = _resolve_run_name(config, dataset_type)
        report_path = write_backtest_report(
            result, config.paths.reports_dir, run_name
        )
        print(report_path)
        return

    if args.command == "run-walkforward":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_gold_features(config, symbols, dataset_type=dataset_type)
        result = run_walk_forward(
            features, config, strategy_factory=lambda: build_strategy(config.strategy)
        )
        run_name = _resolve_run_name(config, dataset_type)
        report_path = write_walkforward_report(
            result, config.paths.reports_dir, run_name
        )
        print(report_path)
        return

    if args.command == "run-bootstrap":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_gold_features(config, symbols, dataset_type=dataset_type)
        result = run_bootstrap_validation(
            features,
            config,
            strategy_factory=lambda: build_strategy(config.strategy),
        )
        run_name = f"{_resolve_run_name(config, dataset_type)}_bootstrap"
        report_path = write_bootstrap_report(result, config.paths.reports_dir, run_name)
        print(report_path)
        return

    if args.command == "run-grid-search":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_gold_features(config, symbols, dataset_type=dataset_type)
        parameter_grid = expand_param_grid(parse_param_grid_specs(list(args.param_grid)))
        result = run_walk_forward_grid_search(
            features,
            config,
            parameter_grid=parameter_grid,
            strategy_builder=lambda params: _build_strategy_from_params(config, params),
            objective=args.objective,
        )
        run_name = f"{_resolve_run_name(config, dataset_type)}_grid_search"
        report_path = write_grid_search_report(
            result,
            config.paths.reports_dir,
            run_name,
        )
        print(report_path)
        return

    if args.command == "run-sensitivity":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_gold_features(config, symbols, dataset_type=dataset_type)
        result = run_execution_sensitivity(
            features,
            config,
            strategy_factory=lambda: build_strategy(config.strategy),
        )
        run_name = _resolve_run_name(config, dataset_type)
        report_path = write_sensitivity_report(
            result, config.paths.reports_dir, run_name
        )
        print(report_path)
        return

    if args.command == "run-alpha-scan":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_gold_features(config, symbols, dataset_type=dataset_type)
        result = run_alpha_scan(
            features,
            config,
            feature_names=list(args.feature) or None,
            horizons=list(args.horizon_bars) or None,
        )
        run_name = f"{_resolve_run_name(config, dataset_type)}_alpha_scan"
        report_path = write_alpha_scan_report(result, config.paths.reports_dir, run_name)
        print(report_path)
        return

    if args.command == "run-order-flow-suite":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        feature_frames = _load_order_flow_suite_features(
            config,
            symbols,
            dataset_type,
        )
        result = run_order_flow_suite(feature_frames=feature_frames, config=config)
        run_name = f"{_resolve_run_name(config, dataset_type)}_order_flow_suite"
        report_path = write_order_flow_suite_report(
            result,
            config.paths.reports_dir,
            run_name,
        )
        print(report_path)
        return

    if args.command == "run-candidate-deep-dive":
        symbols = _resolve_symbols(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        features = _load_candidate_deep_dive_features(
            config,
            symbols,
            dataset_type,
        )
        result = run_candidate_deep_dive(features=features, config=config)
        run_name = f"{_resolve_run_name(config, dataset_type)}_candidate_deep_dive"
        report_path = write_candidate_deep_dive_report(
            result,
            config.paths.reports_dir,
            run_name,
        )
        print(report_path)
        return

    if args.command == "inspect-dataset":
        symbol = _resolve_symbol(config, args.symbol)
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        if args.layer == "silver":
            frame = load_silver_dataset(config, symbol, dataset_type=dataset_type)
        elif args.layer == "gold":
            frame = _load_gold_features(config, [symbol], dataset_type=dataset_type)
        else:
            files = _list_layer_files(config, args.layer, dataset_type, symbol)
            for file_path in files[:50]:
                print(file_path)
            return
        print(frame.schema)
        print(frame.head(5))
        print({"rows": frame.height, "columns": frame.width})
        return

    if args.command == "report":
        dataset_type = _resolve_dataset_type(config, args.dataset_type)
        run_dir = (
            Path(args.run_dir)
            if args.run_dir is not None
            else config.paths.reports_dir / _resolve_run_name(config, dataset_type)
        )
        markdown_path = rebuild_report(run_dir)
        print(markdown_path)
        return

    raise ValueError(f"Unknown command: {args.command}")
