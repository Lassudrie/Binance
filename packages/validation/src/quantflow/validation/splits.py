from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True, slots=True)
class TimeWalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_start_time: datetime
    train_end_time: datetime
    test_start_time: datetime
    test_end_time: datetime


def walk_forward_splits(
    total_rows: int,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int,
) -> list[WalkForwardWindow]:
    windows: list[WalkForwardWindow] = []
    train_start = 0
    while True:
        train_end = train_start + train_bars
        test_start = train_end + embargo_bars
        test_end = test_start + test_bars
        if test_end > total_rows:
            break
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        train_start += step_bars
    return windows


def purged_kfold_splits(
    total_rows: int, n_splits: int, embargo_bars: int
) -> list[tuple[range, range]]:
    fold_size = total_rows // n_splits
    splits: list[tuple[range, range]] = []
    for fold in range(n_splits):
        test_start = fold * fold_size
        test_end = total_rows if fold == n_splits - 1 else test_start + fold_size
        train_left = range(0, max(0, test_start - embargo_bars))
        train_right = range(min(total_rows, test_end + embargo_bars), total_rows)
        test_range = range(test_start, test_end)
        splits.append((range(train_left.start, train_left.stop), test_range))
        if train_right.start < train_right.stop:
            splits.append((range(train_right.start, train_right.stop), test_range))
    return splits


def unique_timestamps(features: pl.DataFrame, timestamp_column: str = "bar_end") -> list[datetime]:
    if timestamp_column not in features.columns:
        raise ValueError(f"Timestamp column not found: {timestamp_column}")
    return (
        features.select(timestamp_column)
        .unique()
        .sort(timestamp_column)
        .get_column(timestamp_column)
        .to_list()
    )


def walk_forward_time_splits(
    features: pl.DataFrame,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int,
    embargo_bars: int,
    timestamp_column: str = "bar_end",
) -> list[TimeWalkForwardWindow]:
    timestamps = unique_timestamps(features, timestamp_column=timestamp_column)
    inferred_step = (
        timestamps[1] - timestamps[0] if len(timestamps) >= 2 else timedelta(seconds=1)
    )
    row_windows = walk_forward_splits(
        len(timestamps),
        train_bars=train_bars,
        test_bars=test_bars,
        step_bars=step_bars,
        embargo_bars=embargo_bars,
    )
    windows: list[TimeWalkForwardWindow] = []
    for window in row_windows:
        windows.append(
            TimeWalkForwardWindow(
                train_start=window.train_start,
                train_end=window.train_end,
                test_start=window.test_start,
                test_end=window.test_end,
                train_start_time=timestamps[window.train_start],
                train_end_time=_boundary_time(timestamps, window.train_end, inferred_step),
                test_start_time=timestamps[window.test_start],
                test_end_time=_boundary_time(timestamps, window.test_end, inferred_step),
            )
        )
    return windows


def slice_frame_by_time_window(
    features: pl.DataFrame,
    *,
    start_time: datetime,
    end_time: datetime,
    timestamp_column: str = "bar_end",
) -> pl.DataFrame:
    return (
        features.filter(
            (pl.col(timestamp_column) >= pl.lit(start_time))
            & (pl.col(timestamp_column) < pl.lit(end_time))
        )
        .sort([timestamp_column, "symbol"])
    )


def _boundary_time(
    timestamps: list[datetime],
    index: int,
    step: timedelta,
) -> datetime:
    if index < len(timestamps):
        return timestamps[index]
    return timestamps[-1] + step
