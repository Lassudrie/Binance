from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from collections import defaultdict

import pandas as pd

from ofbot.memory.analytics import write_daily_artifacts, write_run_artifacts


def build_daily_report(store, date_value: str, output_root: Path) -> Path:
    if date_value == "today":
        date_value = date.today().isoformat()
    output_dir = output_root / f"report_{date_value}"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = output_dir / "report.md"
    trades = store._conn.execute(
        "SELECT * FROM trade_contexts WHERE CAST(entry_time AS DATE) = CAST(? AS DATE)",
        [date_value],
    ).fetch_df()
    events = store._conn.execute(
        "SELECT * FROM event_journal WHERE CAST(event_time AS DATE) = CAST(? AS DATE)",
        [date_value],
    ).fetch_df()
    summary = _summarize_frames(trades, events)
    artifacts = write_daily_artifacts(store=store, date_value=date_value, output_dir=output_dir)
    markdown.write_text(
        "\n".join(
            [
                "# ofbot Daily Report",
                "",
                f"Date: `{date_value}`",
                "",
                "## Artifacts",
                f"- [Trades CSV]({Path(artifacts['trades_csv']).name})",
                f"- [Events CSV]({Path(artifacts['events_csv']).name})",
                "",
                "## Summary",
                f"- Closed trades: `{summary['trades']}`",
                f"- Net PnL: `{summary['net_pnl']:.6f}`",
                f"- Gross PnL: `{summary['gross_pnl']:.6f}`",
                f"- Gross fees: `{summary['fees']:.6f}`",
                f"- Entry fees: `{summary['entry_fees']:.6f}`",
                f"- Exit fees: `{summary['exit_fees']:.6f}`",
                f"- Mean holding time (s): `{summary['mean_holding']:.2f}` (min `{summary['min_holding']:.2f}`, max `{summary['max_holding']:.2f}`)",
                f"- Mean trade notional (USD): `{summary['mean_entry_notional']:.2f}`",
                f"- Mean entry fills / exit fills: `{summary['mean_entry_fill_count']:.2f}` / `{summary['mean_exit_fill_count']:.2f}`",
                f"- Mean estimated entry cost (bps): `{summary['mean_estimated_entry_cost_bps']:.3f}`",
                f"- Mean expected net edge (bps): `{summary['mean_expected_net_edge_bps']:.3f}`",
                f"- Expected-edge blocked entries: `{summary['expected_edge_blocks']}`",
                "",
                "## By Strategy",
                *(f"- {k}: {v}" for k, v in summary["by_strategy"].items()),
                "",
                "## Exit Reasons",
                *(f"- {k}: {v}" for k, v in summary["exit_reasons"].items()),
            ]
        )
    )
    return markdown


def build_run_report(store, run_id: str, output_root: Path) -> Path:
    output_dir = output_root / f"report_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown = output_dir / "report.md"
    trades = store._conn.execute(
        "SELECT * FROM trade_contexts WHERE run_id=?",
        [run_id],
    ).fetch_df()
    events = store._conn.execute(
        "SELECT * FROM event_journal WHERE run_id=?",
        [run_id],
    ).fetch_df()
    summary = _summarize_frames(trades, events)
    artifacts = write_run_artifacts(store=store, run_id=run_id, output_dir=output_dir)
    markdown.write_text(
        "\n".join(
            [
                "# ofbot Run Report",
                "",
                f"Run ID: `{run_id}`",
                "",
                "## Artifacts",
                f"- [Trades CSV]({Path(artifacts['trades_csv']).name})",
                f"- [Events CSV]({Path(artifacts['events_csv']).name})",
                "",
                "## Summary",
                f"- Closed trades: `{summary['trades']}`",
                f"- Net PnL: `{summary['net_pnl']:.6f}`",
                f"- Gross PnL: `{summary['gross_pnl']:.6f}`",
                f"- Gross fees: `{summary['fees']:.6f}`",
                f"- Entry fees: `{summary['entry_fees']:.6f}`",
                f"- Exit fees: `{summary['exit_fees']:.6f}`",
                f"- Mean holding time (s): `{summary['mean_holding']:.2f}` (min `{summary['min_holding']:.2f}`, max `{summary['max_holding']:.2f}`)",
                f"- Mean trade notional (USD): `{summary['mean_entry_notional']:.2f}`",
                f"- Mean entry fills / exit fills: `{summary['mean_entry_fill_count']:.2f}` / `{summary['mean_exit_fill_count']:.2f}`",
                f"- Mean estimated entry cost (bps): `{summary['mean_estimated_entry_cost_bps']:.3f}`",
                f"- Mean expected net edge (bps): `{summary['mean_expected_net_edge_bps']:.3f}`",
                f"- Expected-edge blocked entries: `{summary['expected_edge_blocks']}`",
                "",
                "## By Strategy",
                *(f"- {k}: {v}" for k, v in summary["by_strategy"].items()),
                "",
                "## Exit Reasons",
                *(f"- {k}: {v}" for k, v in summary["exit_reasons"].items()),
            ]
        )
    )
    return markdown


def _summarize_frames(trades: pd.DataFrame, events: pd.DataFrame) -> dict[str, object]:
    event_metrics = _event_metrics(events)
    if trades.empty:
        return {
            "trades": 0,
            "net_pnl": 0.0,
            "gross_pnl": 0.0,
            "fees": 0.0,
            "entry_fees": 0.0,
            "exit_fees": 0.0,
            "mean_holding": 0.0,
            "min_holding": 0.0,
            "max_holding": 0.0,
            "mean_entry_fill_count": 0.0,
            "mean_exit_fill_count": 0.0,
            "mean_entry_notional": 0.0,
            "mean_estimated_entry_cost_bps": event_metrics["mean_estimated_entry_cost_bps"],
            "mean_expected_net_edge_bps": event_metrics["mean_expected_net_edge_bps"],
            "expected_edge_blocks": event_metrics["expected_edge_blocks"],
            "by_strategy": {},
            "exit_reasons": event_metrics["exit_reasons"],
        }

    df = trades.copy()
    by_strategy = defaultdict(float)
    for row in df.itertuples(index=False):
        by_strategy[str(row.strategy)] += float(row.realized_pnl)
    entry_fees = df["entry_fees"].fillna(0.0) if "entry_fees" in df.columns else df["fees"].fillna(0.0)
    exit_fees = df["exit_fees"].fillna(0.0) if "exit_fees" in df.columns else 0.0
    fee_series = entry_fees + exit_fees
    return {
        "trades": int(len(df)),
        "net_pnl": float(df["realized_pnl"].sum()),
        "gross_pnl": float(df["gross_pnl"].sum()),
        "fees": float(fee_series.sum()),
        "entry_fees": float(entry_fees.sum()),
        "exit_fees": float(exit_fees.sum()),
        "mean_holding": float(df["holding_seconds"].mean()),
        "min_holding": float(df["holding_seconds"].min()),
        "max_holding": float(df["holding_seconds"].max()),
        "mean_entry_fill_count": float(df["entry_fill_count"].fillna(0.0).mean()) if "entry_fill_count" in df.columns else 0.0,
        "mean_exit_fill_count": float(df["exit_fill_count"].fillna(0.0).mean()) if "exit_fill_count" in df.columns else 0.0,
        "mean_entry_notional": float(df["entry_notional"].fillna(0.0).mean()) if "entry_notional" in df.columns else 0.0,
        "mean_estimated_entry_cost_bps": event_metrics["mean_estimated_entry_cost_bps"],
        "mean_expected_net_edge_bps": event_metrics["mean_expected_net_edge_bps"],
        "expected_edge_blocks": event_metrics["expected_edge_blocks"],
        "by_strategy": {key: f"{value:.6f}" for key, value in sorted(by_strategy.items(), key=lambda item: item[0])},
        "exit_reasons": event_metrics["exit_reasons"],
    }


def _event_metrics(events: pd.DataFrame) -> dict[str, object]:
    if events.empty:
        return {
            "mean_estimated_entry_cost_bps": 0.0,
            "mean_expected_net_edge_bps": 0.0,
            "expected_edge_blocks": 0,
            "exit_reasons": {},
        }

    params = events["params"].apply(_load_json) if "params" in events.columns else pd.Series([{}] * len(events))
    estimated_entry_cost = pd.to_numeric(params.apply(lambda item: item.get("estimated_entry_cost_bps")), errors="coerce")
    expected_net_edge = pd.to_numeric(params.apply(lambda item: item.get("expected_net_edge_bps")), errors="coerce")

    action = events["action"].fillna("") if "action" in events.columns else ""
    order_intent = events["order_intent"].fillna("") if "order_intent" in events.columns else ""
    reason = events["reason"].fillna("") if "reason" in events.columns else ""
    entry_mask = (
        action.eq("signal")
        & order_intent.eq("submit")
        & expected_net_edge.notna()
    )
    exit_mask = action.eq("signal") & order_intent.eq("submit") & ~entry_mask
    exit_reasons = reason[exit_mask].value_counts().to_dict()
    return {
        "mean_estimated_entry_cost_bps": float(estimated_entry_cost[entry_mask].mean()) if entry_mask.any() else 0.0,
        "mean_expected_net_edge_bps": float(expected_net_edge[entry_mask].mean()) if entry_mask.any() else 0.0,
        "expected_edge_blocks": int((action.eq("signal") & order_intent.eq("risk_expected_net_edge")).sum()),
        "exit_reasons": {str(key): int(value) for key, value in exit_reasons.items()},
    }


def _load_json(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value:
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return payload
    return {}
