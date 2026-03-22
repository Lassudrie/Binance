from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import signal
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

from rich.console import Console

from ofbot import alpha_loop as alpha_loop_mod
from ofbot import research as research_mod
from ofbot.config import load_config
from ofbot.engine import PaperEngine
from ofbot.gateway.binance_public_rest import BinancePublicRestClient
from ofbot.gateway.binance_public_ws import BinancePublicWebSocket
from ofbot.logging import configure_logging
from ofbot.maintenance import cleanup_runtime_artifacts, write_live_learning_status
from ofbot.memory import learning as learning_mod
from ofbot.memory.reports import build_daily_report, build_run_report
from ofbot.research.registry import ResearchRegistry
from ofbot.memory.store import MemoryStore
from ofbot.replay.eval import run_replay
from ofbot.utils.manual_orders import write_manual_order_request

DEFAULT_LOCAL_CONFIG = "config/local.paper.yaml"
DEFAULT_REPORT_DATE = "today"

console = Console()


def _make_async_signal_handler(
    loop: asyncio.AbstractEventLoop,
    callback: Callable[[str], None],
) -> Callable[[int, FrameType | None], None]:
    def _handler(signum: int, _frame: FrameType | None) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        loop.call_soon_threadsafe(callback, signal_name)

    return _handler


def _install_live_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    callback: Callable[[str], None],
) -> list[tuple[signal.Signals, Any]]:
    handler = _make_async_signal_handler(loop, callback)
    installed: list[tuple[signal.Signals, Any]] = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
            signal.signal(sig, handler)
        except (ValueError, OSError, RuntimeError):
            continue
        installed.append((sig, previous))
    return installed


def _restore_live_signal_handlers(handlers: list[tuple[signal.Signals, Any]]) -> None:
    for sig, previous in handlers:
        try:
            signal.signal(sig, previous)
        except (ValueError, OSError, RuntimeError):
            continue


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ofbot")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    subs = parser.add_subparsers(dest="command", required=True)

    live = subs.add_parser("live", help="run live paper trading loop")
    live.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    live.add_argument("--symbol", action="append", help="override configured symbols")
    live.add_argument("--max-events", type=int, default=0, help="stop after N raw events (0=run forever)")
    live.add_argument("--max-seconds", type=int, default=0, help="stop after N seconds (0=run forever)")

    alpha_loop = subs.add_parser("alpha-loop", help="run the continuous offline+paper alpha hunt loop")
    alpha_loop_subs = alpha_loop.add_subparsers(dest="alpha_loop_command", required=True)
    alpha_loop_run = alpha_loop_subs.add_parser("run", help="run one or more alpha-hunt cycles")
    alpha_loop_run.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    alpha_loop_run.add_argument("--suite-config", default="configs/order_flow_suite_spot_hunt.toml")
    alpha_loop_run.add_argument("--deep-dive-config", default="configs/candidate_order_flow_hunt.toml")
    alpha_loop_run.add_argument("--live-run-seconds", type=int, default=120)
    alpha_loop_run.add_argument("--offline-refresh-hours", type=float, default=6.0)
    alpha_loop_run.add_argument("--max-cycles", type=int, default=0)
    alpha_loop_run.add_argument("--once", action="store_true")

    research = subs.add_parser("research", help="controlled replay research loop")
    research_subs = research.add_subparsers(dest="research_command", required=True)

    baseline_bt = research_subs.add_parser("backtest-baseline", help="run baseline replay backtest over train/validation/test")
    baseline_bt.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    baseline_bt.add_argument("--research-config", required=True)
    baseline_bt.add_argument("--input", help="override raw replay parquet path")

    research_run = research_subs.add_parser("run", help="run bounded variant search with train/validation/test")
    research_run.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    research_run.add_argument("--research-config", required=True)
    research_run.add_argument("--input", help="override raw replay parquet path")

    validate = research_subs.add_parser("validate", help="re-validate one candidate from the local registry")
    validate.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    validate.add_argument("--research-config", required=True)
    validate.add_argument("--candidate-id", required=True)
    validate.add_argument("--input", help="override raw replay parquet path")

    promote = research_subs.add_parser("promote-paper", help="promote one validated candidate to paper-only config")
    promote.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    promote.add_argument("--research-config", required=True)
    promote.add_argument("--candidate-id", required=True)

    campaign = research_subs.add_parser("campaign", help="run sequential paper sessions with bounded between-session research")
    campaign.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    campaign.add_argument("--research-config", required=True)
    campaign.add_argument("--sessions", type=int, default=10)
    campaign.add_argument("--session-seconds", type=int, default=600)
    campaign.add_argument("--symbol", action="append", help="override configured symbols")
    campaign.add_argument(
        "--keep-active",
        action="store_true",
        help="reuse the current active paper config instead of resetting to the baseline before session 1",
    )

    rollback = research_subs.add_parser("rollback-paper", help="restore the baseline paper config and journal the rollback")
    rollback.add_argument("--config", default=DEFAULT_LOCAL_CONFIG)
    rollback.add_argument("--research-config", required=True)
    rollback.add_argument("--candidate-id")
    rollback.add_argument("--reason", default="paper_underperformance")

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


async def _run_live(
    config_path: str,
    max_events: int,
    max_seconds: int,
    symbols: list[str] | None,
) -> Path | None:
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
    stop_reason: str | None = None
    consume_task: asyncio.Task[None] | None = None
    loop = asyncio.get_running_loop()
    console.rule("ofbot live run")
    console.print(f"run_id={client.run_id} symbols={','.join(config.symbol_set)}")

    async def _consume_events() -> None:
        nonlocal processed
        async for event in ws.events():
            client.process_event(event.event, received_at=event.received_at)
            processed += 1
            if max_events > 0 and processed >= max_events:
                break

    def _request_shutdown(reason: str) -> None:
        nonlocal stop_reason, consume_task
        if stop_reason is None:
            stop_reason = reason
        if consume_task is not None and not consume_task.done():
            consume_task.cancel()

    installed_handlers = _install_live_signal_handlers(loop, _request_shutdown)

    try:
        consume_task = asyncio.create_task(_consume_events())
        if stop_reason is not None and not consume_task.done():
            consume_task.cancel()
        if max_seconds > 0:
            async with asyncio.timeout(max_seconds):
                await consume_task
        else:
            await consume_task
    except KeyboardInterrupt:
        console.print("[yellow]interrupted[/yellow]")
    except asyncio.CancelledError:
        if stop_reason is None:
            raise
        console.print(f"[yellow]termination requested ({stop_reason})[/yellow]")
    except TimeoutError:
        console.print(f"[yellow]max_seconds reached ({max_seconds}s)[/yellow]")
    finally:
        _restore_live_signal_handlers(installed_handlers)
        if consume_task is not None and not consume_task.done():
            consume_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await consume_task
        report_path = client.finalize()
        if report_path is not None:
            console.print(f"report: {report_path}")
        client.close()
    return report_path


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


def _run_alpha_loop(
    config_path: str,
    *,
    suite_config_path: str,
    deep_dive_config_path: str,
    live_run_seconds: int,
    offline_refresh_hours: float,
    max_cycles: int,
    once: bool,
) -> None:
    base_config_path = Path(config_path)
    base_config = load_config(base_config_path)
    next_offline_at = datetime.now(UTC)
    cycle_index = 0

    while True:
        store = MemoryStore(base_config.learning.duckdb_path)
        active_candidate_id: str | None = None
        active_runtime_config = alpha_loop_mod.current_runtime_config_path(
            base_config_path=base_config_path
        )
        try:
            now = datetime.now(UTC)
            if now >= next_offline_at:
                offline_result = alpha_loop_mod.run_offline_research_cycle(
                    store=store,
                    suite_config_path=Path(suite_config_path),
                    deep_dive_config_path=Path(deep_dive_config_path),
                )
                next_offline_at = alpha_loop_mod.next_offline_refresh(
                    now=now,
                    hours=offline_refresh_hours,
                )
                console.rule("ofbot alpha offline")
                console.print(f"status: {offline_result['status']}")
                if offline_result.get("selected_candidate") is not None:
                    console.print(
                        "selected: "
                        f"{offline_result['selected_candidate']['strategy_name']} "
                        f"{offline_result['selected_candidate']['bar_size']} "
                        f"{offline_result['selected_candidate']['execution_mode']}"
                    )
            active_candidate = alpha_loop_mod.ensure_live_canary_candidate(
                store=store,
                base_config_path=base_config_path,
            )
            if active_candidate is not None:
                active_candidate_id = str(active_candidate["candidate_id"])
                active_runtime_config = Path(
                    str(active_candidate.get("active_config_path") or active_runtime_config)
                )
            status_path = alpha_loop_mod.write_alpha_hunt_status(
                store=store,
                base_config_path=base_config_path,
            )
            console.rule("ofbot alpha status")
            console.print(f"status: {status_path}")
            deployed = alpha_loop_mod.current_active_runtime_candidate(store)
            if deployed is not None and deployed["status"] == "deployed_local":
                break
        finally:
            store.close()

        report_path = asyncio.run(
            _run_live(
                str(active_runtime_config),
                max_events=0,
                max_seconds=max(1, live_run_seconds),
                symbols=None,
            )
        )
        cycle_index += 1

        store = MemoryStore(base_config.learning.duckdb_path)
        try:
            if active_candidate_id is not None and report_path is not None:
                run_id = _run_id_from_report_path(report_path)
                rollout_status = alpha_loop_mod.record_live_canary_result(
                    store=store,
                    base_config_path=base_config_path,
                    candidate_id=active_candidate_id,
                    run_id=run_id,
                    active_config_path=active_runtime_config,
                )
                console.rule("ofbot alpha canary")
                console.print(f"candidate: {active_candidate_id}")
                console.print(f"status: {rollout_status}")
            status_path = alpha_loop_mod.write_alpha_hunt_status(
                store=store,
                base_config_path=base_config_path,
            )
            deployed = alpha_loop_mod.current_active_runtime_candidate(store)
            if deployed is not None and deployed["status"] == "deployed_local":
                console.print(f"deployed: {deployed['candidate_id']}")
                console.print(f"status: {status_path}")
                break
        finally:
            store.close()

        if once:
            break
        if max_cycles > 0 and cycle_index >= max_cycles:
            break


def _run_research_backtest_baseline(
    *,
    config_path: str,
    research_config_path: str,
    input_path: str | None,
) -> None:
    report_path = research_mod.baseline_backtest_report(
        base_config_path=Path(config_path),
        research_config_path=Path(research_config_path),
        raw_input_path=Path(input_path) if input_path else None,
    )
    console.rule("ofbot research baseline")
    console.print(f"report: {report_path}")


def _run_research(
    *,
    config_path: str,
    research_config_path: str,
    input_path: str | None,
) -> None:
    result = research_mod.run_research_cycle(
        base_config_path=Path(config_path),
        research_config_path=Path(research_config_path),
        raw_input_path=Path(input_path) if input_path else None,
    )
    console.rule("ofbot research run")
    console.print(f"status: {result['summary']['status']}")
    console.print(f"report: {result['report_path']}")
    candidate = result["summary"].get("selected_candidate")
    if candidate is not None:
        console.print(f"candidate_id: {candidate['candidate_id']}")


def _run_research_validate(
    *,
    config_path: str,
    research_config_path: str,
    candidate_id: str,
    input_path: str | None,
) -> None:
    result = research_mod.validate_candidate(
        base_config_path=Path(config_path),
        research_config_path=Path(research_config_path),
        candidate_id=candidate_id,
        raw_input_path=Path(input_path) if input_path else None,
    )
    console.rule("ofbot research validate")
    console.print(f"status: {result['summary']['status']}")
    console.print(f"report: {result['report_path']}")


def _run_research_promote_paper(
    *,
    config_path: str,
    research_config_path: str,
    candidate_id: str,
) -> None:
    research_config = research_mod.load_research_config(research_config_path)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    target_path = research_mod.promote_candidate_to_paper(
        base_config_path=Path(config_path),
        registry=registry,
        deployment_dir=research_config.paths.deployment_dir,
        active_config_path=research_config.paths.active_paper_config_path,
        candidate_id=candidate_id,
        paper_window_minutes=research_config.selection.paper_window_minutes,
    )
    console.rule("ofbot research promote")
    console.print(f"paper_config: {target_path}")
    console.print(f"active_config: {research_config.paths.active_paper_config_path}")


def _run_research_campaign(
    *,
    config_path: str,
    research_config_path: str,
    sessions: int,
    session_seconds: int,
    symbols: list[str] | None,
    keep_active: bool,
) -> None:
    result = asyncio.run(
        research_mod.run_paper_research_campaign(
            base_config_path=Path(config_path),
            research_config_path=Path(research_config_path),
            sessions=sessions,
            session_seconds=session_seconds,
            live_runner=_run_live,
            symbols=symbols,
            reset_active_paper=not keep_active,
        )
    )
    console.rule("ofbot research campaign")
    console.print(f"campaign_id: {result['campaign_id']}")
    console.print(f"status: {result['status']}")
    console.print(f"campaign_dir: {result['campaign_dir']}")
    console.print(f"sessions_completed: {result['sessions_completed']}/{result['sessions_requested']}")
    if result.get("active_candidate_id") is not None:
        console.print(f"active_candidate_id: {result['active_candidate_id']}")


def _run_research_rollback_paper(
    *,
    config_path: str,
    research_config_path: str,
    candidate_id: str | None,
    reason: str,
) -> None:
    research_config = research_mod.load_research_config(research_config_path)
    registry = ResearchRegistry(research_config.paths.registry_dir)
    active_path = research_mod.rollback_active_paper_candidate(
        base_config_path=Path(config_path),
        registry=registry,
        active_config_path=research_config.paths.active_paper_config_path,
        candidate_id=candidate_id,
        reason=reason,
    )
    console.rule("ofbot research rollback")
    console.print(f"active_config: {active_path}")


def _run_id_from_report_path(report_path: Path) -> str:
    if report_path.parent.parent.name.startswith("run_"):
        return report_path.parent.parent.name
    if report_path.parent.name.startswith("report_run_"):
        return report_path.parent.name.removeprefix("report_")
    raise SystemExit(f"Unable to infer run_id from report path: {report_path}")


def main() -> None:
    args = _build_arg_parser().parse_args()
    configure_logging(verbose=args.verbose)
    if args.command == "live":
        asyncio.run(_run_live(args.config, args.max_events, args.max_seconds, args.symbol))
    elif args.command == "alpha-loop":
        if args.alpha_loop_command == "run":
            _run_alpha_loop(
                args.config,
                suite_config_path=args.suite_config,
                deep_dive_config_path=args.deep_dive_config,
                live_run_seconds=args.live_run_seconds,
                offline_refresh_hours=args.offline_refresh_hours,
                max_cycles=args.max_cycles,
                once=args.once,
            )
    elif args.command == "research":
        if args.research_command == "backtest-baseline":
            _run_research_backtest_baseline(
                config_path=args.config,
                research_config_path=args.research_config,
                input_path=args.input,
            )
        elif args.research_command == "run":
            _run_research(
                config_path=args.config,
                research_config_path=args.research_config,
                input_path=args.input,
            )
        elif args.research_command == "validate":
            _run_research_validate(
                config_path=args.config,
                research_config_path=args.research_config,
                candidate_id=args.candidate_id,
                input_path=args.input,
            )
        elif args.research_command == "promote-paper":
            _run_research_promote_paper(
                config_path=args.config,
                research_config_path=args.research_config,
                candidate_id=args.candidate_id,
            )
        elif args.research_command == "campaign":
            _run_research_campaign(
                config_path=args.config,
                research_config_path=args.research_config,
                sessions=args.sessions,
                session_seconds=args.session_seconds,
                symbols=args.symbol,
                keep_active=args.keep_active,
            )
        elif args.research_command == "rollback-paper":
            _run_research_rollback_paper(
                config_path=args.config,
                research_config_path=args.research_config,
                candidate_id=args.candidate_id,
                reason=args.reason,
            )
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
