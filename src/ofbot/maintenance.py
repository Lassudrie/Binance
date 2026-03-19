from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ofbot.config import AppConfig
from ofbot.memory.store import MemoryStore


def cleanup_runtime_artifacts(
    *,
    config: AppConfig,
    store: MemoryStore,
    dry_run: bool = False,
    aggressive_historical: bool = False,
) -> dict[str, Any]:
    recent_raw_run_ids = _recent_run_ids(store, config.housekeeping.keep_recent_run_raw)
    recent_report_run_ids = _recent_run_ids(store, config.housekeeping.keep_recent_report_csv)
    candidate_source_run_ids = (
        _pending_candidate_source_run_ids(store)
        if config.housekeeping.preserve_pending_candidate_sources
        else set()
    )

    keep_raw_run_ids = recent_raw_run_ids | candidate_source_run_ids
    keep_report_run_ids = recent_report_run_ids | candidate_source_run_ids

    removed_raw_files: list[str] = []
    removed_report_csvs: list[str] = []
    removed_tmp_paths: list[str] = []
    removed_historical_paths: list[str] = []
    freed_bytes = 0

    raw_dir = config.paths.raw_dir
    for raw_path in sorted(raw_dir.glob("raw_run_*.parquet")):
        run_id = raw_path.stem.removeprefix("raw_")
        if run_id in keep_raw_run_ids:
            continue
        freed_bytes += _delete_path(raw_path, dry_run=dry_run)
        removed_raw_files.append(str(raw_path))

    reports_root = config.paths.reports_dir
    for run_root in sorted(reports_root.glob("run_*")):
        if not run_root.is_dir():
            continue
        run_id = run_root.name
        if run_id in keep_report_run_ids:
            continue
        report_dir = run_root / f"report_{run_id}"
        for artifact_name in ("events.csv", "trades.csv"):
            artifact_path = report_dir / artifact_name
            if not artifact_path.exists():
                continue
            freed_bytes += _delete_path(artifact_path, dry_run=dry_run)
            removed_report_csvs.append(str(artifact_path))

    tmp_pytest_dir = config.paths.root / ".tmp_pytest"
    if config.housekeeping.remove_tmp_pytest and tmp_pytest_dir.exists():
        freed_bytes += _delete_path(tmp_pytest_dir, dry_run=dry_run)
        removed_tmp_paths.append(str(tmp_pytest_dir))

    if aggressive_historical:
        for path in _historical_cleanup_paths(config):
            if not path.exists():
                continue
            freed_bytes += _delete_path(path, dry_run=dry_run)
            removed_historical_paths.append(str(path))

    return {
        "dry_run": dry_run,
        "freed_bytes": freed_bytes,
        "freed_human": _human_bytes(freed_bytes),
        "kept_raw_run_count": len(keep_raw_run_ids),
        "kept_report_run_count": len(keep_report_run_ids),
        "removed_raw_files": removed_raw_files,
        "removed_report_csvs": removed_report_csvs,
        "removed_tmp_paths": removed_tmp_paths,
        "removed_historical_paths": removed_historical_paths,
    }


def write_live_learning_status(
    *,
    config: AppConfig,
    store: MemoryStore,
    output_path: Path | None = None,
) -> Path:
    destination = output_path or config.housekeeping.status_markdown_path
    destination.parent.mkdir(parents=True, exist_ok=True)

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    disk = _disk_snapshot(config)
    recent_runs = _recent_runs(store, limit=8)
    candidates = _candidate_rollup(store, limit=8)
    lesson_counts = _recent_lesson_counts(store, limit=5)
    candidate_counts = _candidate_status_counts(store)
    total_reviews = int(
        store._conn.execute("SELECT COUNT(*) FROM run_reviews").fetchone()[0]
    )

    promoted_count = candidate_counts.get("promotable", 0)
    pending_count = candidate_counts.get("pending", 0)
    alpha_status = "not_confirmed"
    if promoted_count > 0:
        alpha_status = "promotable_candidate_available"
    elif pending_count > 0:
        alpha_status = "candidate_pending_more_oos"

    lines = [
        "# Live Learning Status",
        "",
        f"Generated: `{generated_at}`",
        "",
        "## Snapshot",
        f"- Alpha status: `{alpha_status}`",
        f"- Reviewed runs: `{total_reviews}`",
        f"- Candidates: `{sum(candidate_counts.values())}` ({_format_status_counts(candidate_counts)})",
        f"- Best active candidate delta vs baseline: `{_best_candidate_delta(candidates):+.6f}`",
        "",
        "## Disk",
        f"- Free space: `{disk['disk_free_human']}` / `{disk['disk_total_human']}`",
        f"- Runtime raw runs: `{disk['runtime_raw_runs_human']}`",
        f"- Runtime reports: `{disk['runtime_reports_human']}`",
        f"- Memory: `{disk['memory_human']}`",
        f"- Temp pytest: `{disk['tmp_pytest_human']}`",
        f"- Historical raw cache: `{disk['historical_raw_spot_human']}`",
        f"- Historical bronze: `{disk['historical_bronze_human']}`",
        f"- Historical silver: `{disk['historical_silver_human']}`",
        "",
        "## Retention",
        f"- Keep recent raw runs: `{config.housekeeping.keep_recent_run_raw}`",
        f"- Keep recent report CSV sets: `{config.housekeeping.keep_recent_report_csv}`",
        f"- Preserve pending candidate source runs: `{config.housekeeping.preserve_pending_candidate_sources}`",
        "",
        "## Recent Runs",
    ]
    if recent_runs:
        for row in recent_runs:
            lines.append(
                "- "
                f"{row['run_id']} verdict=`{row['verdict']}` trades=`{row['trade_count']}` "
                f"net=`{row['net_pnl']:+.6f}` dominant=`{row['dominant_strategy']}` "
                f"edge_blocks=`{row['expected_edge_blocks']}`"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Candidates",
        ]
    )
    if candidates:
        for candidate in candidates:
            lines.append(
                "- "
                f"{candidate['candidate_id']} status=`{candidate['status']}` "
                f"delta=`{candidate['delta_net_pnl']:+.6f}` "
                f"baseline=`{candidate['baseline_net_pnl']:+.6f}` "
                f"candidate=`{candidate['candidate_net_pnl']:+.6f}` "
                f"validations=`{candidate['validation_count']}` "
                f"trades=`{candidate['candidate_trade_count']}`"
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Recent Blockers",
        ]
    )
    if lesson_counts:
        for lesson_type, count in lesson_counts:
            lines.append(f"- `{lesson_type}`: `{count}`")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Next Actions",
        ]
    )
    lines.extend(_build_next_actions(candidates=candidates, disk=disk))

    destination.write_text("\n".join(lines) + "\n")
    return destination


def _recent_run_ids(store: MemoryStore, keep_count: int) -> set[str]:
    if keep_count <= 0:
        return set()
    rows = store._conn.execute(
        """
        SELECT run_id
        FROM run_reviews
        ORDER BY updated_at DESC, run_id DESC
        LIMIT ?
        """,
        [keep_count],
    ).fetchall()
    return {str(row[0]) for row in rows}


def _pending_candidate_source_run_ids(store: MemoryStore) -> set[str]:
    rows = store._conn.execute(
        """
        SELECT DISTINCT source_run_id
        FROM learning_candidates
        WHERE status IN ('pending', 'promotable')
        """
    ).fetchall()
    return {str(row[0]) for row in rows if row[0]}


def _historical_cleanup_paths(config: AppConfig) -> list[Path]:
    data_root = config.paths.root / "data"
    return [
        data_root / "bronze",
        data_root / "silver",
        data_root / "gold",
        data_root / "reports",
        data_root / "raw" / "spot",
    ]


def _delete_path(path: Path, *, dry_run: bool) -> int:
    size_bytes = _path_size(path)
    if dry_run:
        return size_bytes
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=False)
    elif path.exists():
        path.unlink()
    return size_bytes


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    return sum(
        int(child.stat().st_size)
        for child in path.rglob("*")
        if child.is_file()
    )


def _disk_snapshot(config: AppConfig) -> dict[str, Any]:
    usage = shutil.disk_usage(config.paths.root)
    data_root = config.paths.root / "data"
    runtime_raw_runs_size = sum(
        _path_size(path) for path in config.paths.raw_dir.glob("raw_run_*.parquet")
    )
    return {
        "disk_free_bytes": usage.free,
        "disk_total_bytes": usage.total,
        "disk_free_human": _human_bytes(usage.free),
        "disk_total_human": _human_bytes(usage.total),
        "runtime_raw_runs_human": _human_bytes(runtime_raw_runs_size),
        "runtime_reports_human": _human_bytes(_path_size(config.paths.reports_dir)),
        "memory_human": _human_bytes(_path_size(config.paths.memory_dir)),
        "tmp_pytest_human": _human_bytes(_path_size(config.paths.root / ".tmp_pytest")),
        "historical_raw_spot_human": _human_bytes(_path_size(data_root / "raw" / "spot")),
        "historical_bronze_human": _human_bytes(_path_size(data_root / "bronze")),
        "historical_silver_human": _human_bytes(_path_size(data_root / "silver")),
    }


def _recent_runs(store: MemoryStore, limit: int) -> list[dict[str, Any]]:
    rows = store._conn.execute(
        """
        SELECT run_id, verdict, summary
        FROM run_reviews
        ORDER BY updated_at DESC, run_id DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    recent: list[dict[str, Any]] = []
    for run_id, verdict, summary_raw in rows:
        summary = _load_json_dict(summary_raw)
        recent.append(
            {
                "run_id": str(run_id),
                "verdict": str(verdict),
                "trade_count": int(summary.get("trade_count", 0) or 0),
                "net_pnl": float(summary.get("net_pnl", 0.0) or 0.0),
                "dominant_strategy": str(summary.get("dominant_strategy") or "-"),
                "expected_edge_blocks": int(summary.get("expected_edge_blocks", 0) or 0),
            }
        )
    return recent


def _candidate_rollup(store: MemoryStore, limit: int) -> list[dict[str, Any]]:
    rows = store._conn.execute(
        """
        SELECT
            lc.candidate_id,
            lc.source_run_id,
            lc.status,
            lc.strategy,
            lc.archetype,
            COUNT(cv.validation_run_id) AS validation_count,
            COALESCE(SUM(cv.baseline_net_pnl), 0.0) AS baseline_net_pnl,
            COALESCE(SUM(cv.candidate_net_pnl), 0.0) AS candidate_net_pnl,
            COALESCE(SUM(cv.baseline_trade_count), 0) AS baseline_trade_count,
            COALESCE(SUM(cv.candidate_trade_count), 0) AS candidate_trade_count
        FROM learning_candidates lc
        LEFT JOIN candidate_validations cv
            ON cv.candidate_id = lc.candidate_id
        GROUP BY 1,2,3,4,5
        ORDER BY
            (COALESCE(SUM(cv.candidate_net_pnl), 0.0) - COALESCE(SUM(cv.baseline_net_pnl), 0.0)) DESC,
            COUNT(cv.validation_run_id) DESC,
            lc.candidate_id ASC
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    candidates: list[dict[str, Any]] = []
    for row in rows:
        baseline_net = float(row[6] or 0.0)
        candidate_net = float(row[7] or 0.0)
        candidates.append(
            {
                "candidate_id": str(row[0]),
                "source_run_id": str(row[1]),
                "status": str(row[2]),
                "strategy": str(row[3]),
                "archetype": str(row[4]),
                "validation_count": int(row[5] or 0),
                "baseline_net_pnl": baseline_net,
                "candidate_net_pnl": candidate_net,
                "baseline_trade_count": int(row[8] or 0),
                "candidate_trade_count": int(row[9] or 0),
                "delta_net_pnl": candidate_net - baseline_net,
            }
        )
    return candidates


def _candidate_status_counts(store: MemoryStore) -> dict[str, int]:
    rows = store._conn.execute(
        """
        SELECT status, COUNT(*)
        FROM learning_candidates
        GROUP BY status
        """
    ).fetchall()
    return {str(status): int(count) for status, count in rows}


def _recent_lesson_counts(store: MemoryStore, limit: int) -> list[tuple[str, int]]:
    rows = store._conn.execute(
        """
        WITH recent_runs AS (
            SELECT run_id
            FROM run_reviews
            ORDER BY updated_at DESC, run_id DESC
            LIMIT 10
        )
        SELECT lesson_type, COUNT(*)
        FROM lessons_learned
        WHERE run_id IN (SELECT run_id FROM recent_runs)
        GROUP BY lesson_type
        ORDER BY COUNT(*) DESC, lesson_type ASC
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    return [(str(lesson_type), int(count)) for lesson_type, count in rows]


def _build_next_actions(*, candidates: list[dict[str, Any]], disk: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if candidates:
        best = candidates[0]
        if best["status"] == "pending" and best["candidate_trade_count"] < 20:
            actions.append(
                f"- Collect more OOS runs for `{best['candidate_id']}` until candidate trades reach `20`."
            )
        if best["delta_net_pnl"] <= 0.0:
            actions.append("- No candidate is outperforming baseline right now; tune entry filters before adding more variants.")
    else:
        actions.append("- No candidate is active yet; keep collecting live runs with closed trades.")

    if disk["historical_raw_spot_human"] != "0B" or disk["historical_silver_human"] != "0B":
        actions.append("- Historical market caches still dominate disk usage; use aggressive cleanup only if you no longer need offline datasets.")
    return actions or ["- Keep the current loop running and refresh this status file after new validations."]


def _best_candidate_delta(candidates: list[dict[str, Any]]) -> float:
    if not candidates:
        return 0.0
    return float(candidates[0]["delta_net_pnl"])


def _format_status_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))


def _load_json_dict(value: object) -> dict[str, Any]:
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


def _human_bytes(value: int) -> str:
    if value <= 0:
        return "0B"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)}{unit}"
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"
