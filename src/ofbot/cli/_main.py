from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from ofbot.config import load_config
from ofbot.engine import PaperEngine
from ofbot.gateway.binance_public_rest import BinancePublicRestClient
from ofbot.gateway.binance_public_ws import BinancePublicWebSocket
from ofbot.logging import configure_logging
from ofbot.maintenance import cleanup_runtime_artifacts, write_live_learning_status
from ofbot.memory import learning as learning_mod
from ofbot.memory.reports import build_daily_report, build_run_report
from ofbot.memory.store import MemoryStore
from ofbot.replay.eval import run_replay
from ofbot.utils.manual_orders import write_manual_order_request

DEFAULT_LOCAL_CONFIG = "config/local.paper.yaml"
DEFAULT_REPORT_DATE = "today"

console = Console()


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofbot")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subs = parser.add_subparsers(dest="command", required=True)

    live = subs.add_parser("live", help="run live paper trading loop")
    live.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    live.add_argument("--symbol", action="append", help="override configured symbols")
    live.add_argument("--max-events", type=int, default=0, help="stop after N raw events (0=run forever)")
    live.add_argument("--max-seconds", type=int, default=0, help="stop after N seconds (0=run forever)")

    replay = subs.add_parser("replay", help="run replay from recorded parquet")
    replay.add_argument("--input", required=True, help="path to raw replay parquet")
    replay.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)

    report = subs.add_parser("report", help="generate report from learning memory")
    report.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    report.add_argument(
        "--date",
        default=DEFAULT_REPORT_DATE,
        help="YYYY-MM-DD or 'today' (default: today)",
    )

    learn = subs.add_parser("learn", help="advance the learning loop for a completed run")
    learn_subs = learn.add_subparsers(dest="learn_command", required=True)
    learn_advance = learn_subs.add_parser("advance", help="build lessons, validations and candidate overlays")
    learn_advance.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    learn_advance.add_argument("--run-id", required=True)
    learn_advance.add_argument("--input", required=True, help="path to raw replay parquet for OOS validation")

    journal = subs.add_parser("journal", help="attach manual notes to the learning journal")
    journal_subs = journal.add_subparsers(dest="journal_command", required=True)

    note_run = journal_subs.add_parser("note-run", help="append a manual run-level note")
    note_run.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    note_run.add_argument("--run-id", required=True)
    note_run.add_argument("--tag", required=True)
    note_run.add_argument("--note", required=True)

    note_trade = journal_subs.add_parser("note-trade", help="append a manual trade-level note")
    note_trade.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    note_trade.add_argument("--run-id", required=True)
    note_trade.add_argument("--trade-id", type=int, required=True)
    note_trade.add_argument("--tag", required=True)
    note_trade.add_argument("--note", required=True)

    ops = subs.add_parser("ops", help="runtime cleanup and status helpers")
    ops_subs = ops.add_subparsers(dest="ops_command", required=True)

    cleanup = ops_subs.add_parser("cleanup", help="prune runtime artifacts to protect disk space")
    cleanup.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    cleanup.add_argument("--dry-run", action="store_true")
    cleanup.add_argument(
        "--aggressive-historical",
        action="store_true",
        help="also delete historical bronze/silver/raw spot caches",
    )

    status_md = ops_subs.add_parser("status-md", help="write a synthetic live-learning markdown status")
    status_md.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    status_md.add_argument("--output", help="output markdown path")

    paper_order = subs.add_parser("paper-order", help="enqueue a manual paper-only order for a live session")
    paper_order.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    paper_order.add_argument("--symbol", required=True)
    paper_order.add_argument("--side", choices=["buy", "sell", "flat"], required=True)
    paper_order.add_argument("--qty", type=float, default=0.0, help="delta quantity to trade")
    paper_order.add_argument("--notional-usd", type=float, default=0.0, help="delta notional in USD")
    paper_order.add_argument("--order-type", choices=["market", "limit"], default="market")
    paper_order.add_argument("--reason", default="manual_paper_order")
    paper_order.add_argument("--force", action="store_true", help="bypass pre-entry risk for paper-only testing")
    return parser


async def _run_live(config_path: str, max_events: int, max_seconds: int, symbols: list[str] | None) -> None:
    config = load_config(config_path)
    if symbols:
        config.symbols = symbols
        config.paths = config.paths
    if config.mode == "paper_testnet" and config.enable_testnet_signed_endpoints:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")
        if not api_key or not api_secret:
            config.enable_testnet_signed_endpoints = False
    depth_snapshot_client = BinancePublicRestClient(
        base_url=config.gateway.public_rest_base,
        timeout_seconds=config.gateway.request_timeout_s,
        depth_snapshot_limit=config.gateway.depth_snapshot_limit,
    )
    client = PaperEngine(config, depth_snapshot_client=depth_snapshot_client)
    ws = BinancePublicWebSocket(symbols=config.symbol_set, config=config.gateway)

    processed = 0
    console.rule("ofbot live run")
    console.print(f"run_id={client.run_id} symbols={','.join(config.symbol_set)}")

    async def _consume_events() -> None:
        nonlocal processed
        async for event in ws.events():
            client.process_event(event.event, received_at=event.received_at)
            processed += 1
            if max_events > 0 and processed >= max_events:
                break

    try:
        if max_seconds > 0:
            async with asyncio.timeout(max_seconds):
                await _consume_events()
        else:
            await _consume_events()
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
    except TimeoutError:
        console.print(f"[yellow]max_seconds reached ({max_seconds}s)[/yellow]")
    finally:
        report_path = client.finalize()
        if report_path is not None:
            console.print(f"report: {report_path}")
        client.close()


def _run_replay(config_path: str, input_path: str) -> None:
    config = load_config(config_path)
    report_path = run_replay(config, Path(input_path))
    if report_path is not None:
        console.rule("ofbot replay report")
        console.print(f"report: {report_path}")


def _run_report(config_path: str, date_value: str) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        report_path = build_daily_report(store=store, date_value=date_value, output_root=config.paths.reports_dir)
    finally:
        store.close()
    console.rule("ofbot daily report")
    console.print(f"report: {report_path}")


def _run_learn_advance(config_path: str, *, run_id: str, input_path: str) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        report_path = build_run_report(store, run_id, config.paths.reports_dir / run_id)
        overview_path = learning_mod.advance_learning_cycle(
            store=store,
            config=config,
            run_id=run_id,
            raw_input_path=Path(input_path),
            report_dir=report_path.parent,
        )
        status_path = write_live_learning_status(config=config, store=store)
    finally:
        store.close()
    console.rule("ofbot learning advance")
    console.print(f"overview: {overview_path}")
    console.print(f"status: {status_path}")


def _run_note_run(config_path: str, *, run_id: str, tag: str, note: str) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        store.insert_manual_note(
            run_id=run_id,
            note_scope="run",
            tag=tag,
            note_text=note,
        )
        overview_path = _refresh_learning_after_note(store=store, config=config, run_id=run_id)
        status_path = write_live_learning_status(config=config, store=store)
    finally:
        store.close()
    console.rule("ofbot run note")
    console.print(f"overview: {overview_path}")
    console.print(f"status: {status_path}")


def _run_note_trade(config_path: str, *, run_id: str, trade_id: int, tag: str, note: str) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        trade = store.get_trade_context(trade_id)
        if trade is None:
            raise SystemExit(f"trade_context id={trade_id} not found")
        if str(trade[1]) != run_id:
            raise SystemExit(f"trade_context id={trade_id} belongs to run_id={trade[1]}")
        store.insert_manual_note(
            run_id=run_id,
            note_scope="trade",
            trade_context_id=trade_id,
            tag=tag,
            note_text=note,
        )
        overview_path = _refresh_learning_after_note(store=store, config=config, run_id=run_id)
        status_path = write_live_learning_status(config=config, store=store)
    finally:
        store.close()
    console.rule("ofbot trade note")
    console.print(f"overview: {overview_path}")
    console.print(f"status: {status_path}")


def _run_ops_cleanup(
    config_path: str,
    *,
    dry_run: bool,
    aggressive_historical: bool,
) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        summary = cleanup_runtime_artifacts(
            config=config,
            store=store,
            dry_run=dry_run,
            aggressive_historical=aggressive_historical,
        )
        status_path = write_live_learning_status(config=config, store=store)
    finally:
        store.close()
    console.rule("ofbot cleanup")
    console.print(f"freed: {summary['freed_human']}")
    console.print(f"raw_removed: {len(summary['removed_raw_files'])}")
    console.print(f"report_csv_removed: {len(summary['removed_report_csvs'])}")
    console.print(f"status: {status_path}")


def _run_ops_status_md(config_path: str, *, output: str | None) -> None:
    config = load_config(config_path)
    store = MemoryStore(config.learning.duckdb_path)
    try:
        output_path = Path(output) if output else None
        status_path = write_live_learning_status(
            config=config,
            store=store,
            output_path=output_path,
        )
    finally:
        store.close()
    console.rule("ofbot status")
    console.print(f"status: {status_path}")


def _refresh_learning_after_note(*, store: MemoryStore, config, run_id: str) -> Path:
    review = learning_mod.get_run_review(store, run_id)
    if review is None:
        report_path = build_run_report(store, run_id, config.paths.reports_dir / run_id)
        return learning_mod.advance_learning_cycle(
            store=store,
            config=config,
            run_id=run_id,
            raw_input_path=None,
            report_dir=report_path.parent,
            validate_pending=False,
            generate_candidates=False,
        )
    return learning_mod.refresh_learning_artifacts(
        store=store,
        config=config,
        run_id=run_id,
    )


def _run_paper_order(
    config_path: str,
    *,
    symbol: str,
    side: str,
    qty: float,
    notional_usd: float,
    order_type: str,
    reason: str,
    force: bool,
) -> None:
    config = load_config(config_path)
    if config.mode != "paper_local":
        raise SystemExit("manual paper orders are only supported in paper_local mode")

    qty_value = float(qty) if qty > 0 else None
    notional_value = float(notional_usd) if notional_usd > 0 else None
    if side != "flat" and qty_value is None and notional_value is None:
        raise SystemExit("provide --qty or --notional-usd for buy/sell manual orders")

    side_value = {"buy": 1, "sell": -1, "flat": 0}[side]
    request_path = write_manual_order_request(
        config.paths.memory_dir,
        symbol=symbol,
        side=side_value,
        order_type=order_type,
        qty=qty_value,
        notional_usd=notional_value,
        reason=reason,
        force=force,
    )
    console.rule("ofbot paper order queued")
    console.print(f"request: {request_path}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)
    if args.command == "live":
        asyncio.run(_run_live(args.config, args.max_events, args.max_seconds, args.symbol))
    elif args.command == "replay":
        _run_replay(args.config, args.input)
    elif args.command == "report":
        date_value = args.date
        if date_value == "today":
            date_value = datetime.now(UTC).date().isoformat()
        _run_report(args.config, date_value)
    elif args.command == "learn":
        if args.learn_command == "advance":
            _run_learn_advance(args.config, run_id=args.run_id, input_path=args.input)
    elif args.command == "journal":
        if args.journal_command == "note-run":
            _run_note_run(args.config, run_id=args.run_id, tag=args.tag, note=args.note)
        elif args.journal_command == "note-trade":
            _run_note_trade(args.config, run_id=args.run_id, trade_id=args.trade_id, tag=args.tag, note=args.note)
    elif args.command == "ops":
        if args.ops_command == "cleanup":
            _run_ops_cleanup(
                args.config,
                dry_run=args.dry_run,
                aggressive_historical=args.aggressive_historical,
            )
        elif args.ops_command == "status-md":
            _run_ops_status_md(args.config, output=args.output)
    elif args.command == "paper-order":
        _run_paper_order(
            args.config,
            symbol=args.symbol,
            side=args.side,
            qty=args.qty,
            notional_usd=args.notional_usd,
            order_type=args.order_type,
            reason=args.reason,
            force=args.force,
        )


if __name__ == "__main__":
    main()
