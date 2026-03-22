from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from ofbot.research.config import DatasetConfig


@dataclass(frozen=True, slots=True)
class ReplayFrameSlice:
    name: str
    frame: pl.DataFrame
    start_time: datetime
    end_time: datetime


@dataclass(frozen=True, slots=True)
class ReplayDatasetSplit:
    train: ReplayFrameSlice
    validation: ReplayFrameSlice
    test: ReplayFrameSlice

    @property
    def train_validation_frame(self) -> pl.DataFrame:
        return pl.concat([self.train.frame, self.validation.frame], how="vertical")


@dataclass(frozen=True, slots=True)
class ReplayWalkForwardWindow:
    index: int
    train: ReplayFrameSlice
    test: ReplayFrameSlice


def load_replay_frame(path: str | pl.DataFrame | object) -> pl.DataFrame:
    if isinstance(path, pl.DataFrame):
        return path.sort(["event_time", "sequence"])
    return pl.read_parquet(path).sort(["event_time", "sequence"])


def split_replay_dataset(frame: pl.DataFrame, config: DatasetConfig) -> ReplayDatasetSplit:
    timestamps = _unique_timestamps(frame, column=config.timestamp_column)
    if len(timestamps) < config.min_unique_timestamps:
        raise ValueError(
            f"dataset too short for research split: {len(timestamps)} < {config.min_unique_timestamps}"
        )

    train_end = max(1, int(len(timestamps) * config.train_ratio))
    validation_end = train_end + max(1, int(len(timestamps) * config.validation_ratio))
    validation_end = min(validation_end, len(timestamps) - 1)
    train_end = min(train_end, validation_end - 1)

    train = _slice_by_indices(frame, timestamps, 0, train_end, config.timestamp_column, "train")
    validation = _slice_by_indices(
        frame,
        timestamps,
        train_end,
        validation_end,
        config.timestamp_column,
        "validation",
    )
    test = _slice_by_indices(
        frame,
        timestamps,
        validation_end,
        len(timestamps),
        config.timestamp_column,
        "test",
    )
    return ReplayDatasetSplit(train=train, validation=validation, test=test)


def split_subperiods(
    frame: pl.DataFrame,
    *,
    count: int,
    timestamp_column: str,
) -> list[ReplayFrameSlice]:
    timestamps = _unique_timestamps(frame, column=timestamp_column)
    if not timestamps:
        return []
    bucket_count = max(1, min(count, len(timestamps)))
    step = max(1, len(timestamps) // bucket_count)
    slices: list[ReplayFrameSlice] = []
    start = 0
    for index in range(bucket_count):
        end = len(timestamps) if index == bucket_count - 1 else min(len(timestamps), start + step)
        if end <= start:
            break
        slices.append(_slice_by_indices(frame, timestamps, start, end, timestamp_column, f"subperiod_{index + 1}"))
        start = end
        if start >= len(timestamps):
            break
    return [item for item in slices if item.frame.height > 0]


def build_walk_forward_windows(
    frame: pl.DataFrame,
    *,
    timestamp_column: str,
    train_ratio: float,
    window_count: int,
) -> list[ReplayWalkForwardWindow]:
    timestamps = _unique_timestamps(frame, column=timestamp_column)
    if len(timestamps) < 3 or window_count <= 0:
        return []

    train_size = max(1, int(len(timestamps) * train_ratio))
    remaining = len(timestamps) - train_size
    if remaining <= 0:
        return []
    test_size = max(1, remaining // max(1, window_count))

    windows: list[ReplayWalkForwardWindow] = []
    start = 0
    for index in range(window_count):
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > len(timestamps):
            break
        train_slice = _slice_by_indices(
            frame,
            timestamps,
            start,
            train_end,
            timestamp_column,
            f"wf_train_{index + 1}",
        )
        test_slice = _slice_by_indices(
            frame,
            timestamps,
            train_end,
            test_end,
            timestamp_column,
            f"wf_test_{index + 1}",
        )
        windows.append(ReplayWalkForwardWindow(index=index, train=train_slice, test=test_slice))
        start += test_size
    return windows


def _unique_timestamps(frame: pl.DataFrame, *, column: str) -> list[datetime]:
    if column not in frame.columns:
        raise ValueError(f"timestamp column not found: {column}")
    return frame.select(column).unique().sort(column).get_column(column).to_list()


def _slice_by_indices(
    frame: pl.DataFrame,
    timestamps: list[datetime],
    start_index: int,
    end_index: int,
    timestamp_column: str,
    name: str,
) -> ReplayFrameSlice:
    if end_index <= start_index:
        raise ValueError(f"empty slice requested for {name}")
    start_time = timestamps[start_index]
    end_time = _boundary_time(timestamps, end_index)
    sliced = (
        frame.filter(
            (pl.col(timestamp_column) >= pl.lit(start_time))
            & (pl.col(timestamp_column) < pl.lit(end_time))
        )
        .sort([timestamp_column, "sequence"])
    )
    return ReplayFrameSlice(name=name, frame=sliced, start_time=start_time, end_time=end_time)


def _boundary_time(timestamps: list[datetime], end_index: int) -> datetime:
    if end_index < len(timestamps):
        return timestamps[end_index]
    if len(timestamps) == 1:
        return timestamps[0] + timedelta(microseconds=1)
    step = max(timestamps[-1] - timestamps[-2], timedelta(microseconds=1))
    return timestamps[-1] + step
