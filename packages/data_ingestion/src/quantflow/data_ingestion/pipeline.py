from __future__ import annotations

import io
import logging
import zipfile
from datetime import date
from pathlib import Path

import polars as pl
from quantflow.core.config import AppConfig
from quantflow.data_ingestion.binance import DownloadedArtifact, download_dataset_file
from quantflow.data_ingestion.manifest import IngestionRecord, ManifestRepository
from quantflow.data_ingestion.schemas import BinanceDatasetSchema, get_spot_schema
from quantflow.utils.dates import iter_dates, utc_now
from quantflow.utils.io import ensure_directory

LOGGER = logging.getLogger(__name__)


def detect_timestamp_unit(values: pl.Series) -> str:
    max_value = values.max()
    if max_value is None:
        raise ValueError("Cannot detect timestamp unit from empty series")
    return "us" if int(max_value) >= 10**15 else "ms"


def parse_zip_csv(zip_path: Path, schema: BinanceDatasetSchema) -> pl.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as archive:
        member_names = archive.namelist()
        if not member_names:
            raise ValueError(f"Empty archive: {zip_path}")
        with archive.open(member_names[0], "r") as file_obj:
            payload = file_obj.read()

    frame = pl.read_csv(
        io.BytesIO(payload),
        has_header=False,
        new_columns=list(schema.columns),
        schema_overrides=schema.dtypes,
        infer_schema_length=100,
    )
    if frame.is_empty():
        raise ValueError(f"No rows parsed from {zip_path}")
    return frame


def _normalize_time_expr(column: str, time_unit: str) -> pl.Expr:
    if time_unit == "us":
        return pl.from_epoch(pl.col(column), time_unit="us").dt.replace_time_zone("UTC")
    return pl.from_epoch(pl.col(column), time_unit="ms").dt.replace_time_zone("UTC")


def build_bronze_frame(
    raw_frame: pl.DataFrame,
    *,
    symbol: str,
    market: str,
    dataset_type: str,
    source_date: date,
    timestamp_unit: str,
    schema: BinanceDatasetSchema,
) -> pl.DataFrame:
    return (
        raw_frame.with_columns(
            [
                pl.lit(symbol).alias("symbol"),
                pl.lit(market).alias("market"),
                pl.lit(dataset_type).alias("dataset_type"),
                pl.lit(source_date).alias("source_date"),
                pl.lit(timestamp_unit).alias("timestamp_unit"),
                _normalize_time_expr(schema.timestamp_column, timestamp_unit).alias("event_time"),
            ]
        )
        .with_columns(pl.col("event_time").dt.date().alias("event_date"))
        .sort(["event_time"])
    )


def build_silver_frame(bronze_frame: pl.DataFrame, dataset_type: str) -> pl.DataFrame:
    if dataset_type == "trades":
        silver = (
            bronze_frame.select(
                [
                    "symbol",
                    "market",
                    "dataset_type",
                    "source_date",
                    "event_date",
                    "event_time",
                    "trade_id",
                    "price",
                    "qty",
                    "quote_qty",
                    "is_buyer_maker",
                    "is_best_match",
                ]
            )
            .with_columns(
                pl.when(pl.col("is_buyer_maker"))
                .then(pl.lit(-1))
                .otherwise(pl.lit(1))
                .cast(pl.Int8)
                .alias("side_sign")
            )
            .with_columns(
                [
                    (pl.col("qty") * pl.col("side_sign")).alias("signed_qty"),
                    (pl.col("quote_qty") * pl.col("side_sign")).alias("signed_quote_qty"),
                ]
            )
            .sort(["symbol", "event_time", "trade_id"])
            .unique(subset=["symbol", "trade_id"], maintain_order=True)
        )
    elif dataset_type == "aggTrades":
        silver = (
            bronze_frame.select(
                [
                    "symbol",
                    "market",
                    "dataset_type",
                    "source_date",
                    "event_date",
                    "event_time",
                    pl.col("aggregate_trade_id").alias("trade_id"),
                    "price",
                    "qty",
                    (pl.col("price") * pl.col("qty")).alias("quote_qty"),
                    "is_buyer_maker",
                    "is_best_match",
                    "first_trade_id",
                    "last_trade_id",
                ]
            )
            .with_columns(
                pl.when(pl.col("is_buyer_maker"))
                .then(pl.lit(-1))
                .otherwise(pl.lit(1))
                .cast(pl.Int8)
                .alias("side_sign")
            )
            .with_columns(
                [
                    (pl.col("qty") * pl.col("side_sign")).alias("signed_qty"),
                    (pl.col("quote_qty") * pl.col("side_sign")).alias("signed_quote_qty"),
                ]
            )
            .sort(["symbol", "event_time", "trade_id"])
            .unique(subset=["symbol", "trade_id"], maintain_order=True)
        )
    else:
        raise NotImplementedError(
            f"Silver normalization not implemented for dataset type: {dataset_type}"
        )

    null_counts = silver.null_count().row(0)
    if any(count > 0 for count in null_counts):
        raise ValueError(f"Silver {dataset_type} frame contains nulls in mandatory columns")
    return silver


def partition_directory(base_dir: Path, dataset_type: str, symbol: str, source_date: date) -> Path:
    return base_dir / dataset_type / f"symbol={symbol}" / f"date={source_date.isoformat()}"


def write_partitioned_parquet(
    frame: pl.DataFrame,
    *,
    base_dir: Path,
    dataset_type: str,
    symbol: str,
    source_date: date,
) -> Path:
    target_dir = partition_directory(base_dir, dataset_type, symbol, source_date)
    ensure_directory(target_dir)
    target_path = target_dir / "part-0.parquet"
    frame.write_parquet(target_path)
    return target_path


def download_dataset_range(
    config: AppConfig,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    dataset_type: str | None = None,
) -> list[DownloadedArtifact]:
    resolved_dataset_type = dataset_type or config.dataset.dataset_type
    artifacts: list[DownloadedArtifact] = []
    for source_date in iter_dates(start_date, end_date):
        artifact = download_dataset_file(
            raw_dir=config.paths.raw_dir,
            market=config.binance.market,
            frequency=config.binance.frequency,
            dataset_type=resolved_dataset_type,
            symbol=symbol,
            source_date=source_date,
            timeout_seconds=config.binance.timeout_seconds,
            verify_checksum=config.binance.verify_checksum,
        )
        artifacts.append(artifact)
        LOGGER.info("Downloaded %s", artifact.path)
    return artifacts


def _build_single_dataset(
    config: AppConfig,
    artifact: DownloadedArtifact,
    manifest: ManifestRepository,
) -> IngestionRecord:
    schema = get_spot_schema(artifact.dataset_type)
    raw_frame = parse_zip_csv(artifact.path, schema)
    timestamp_unit = detect_timestamp_unit(raw_frame.get_column(schema.timestamp_column))
    bronze = build_bronze_frame(
        raw_frame,
        symbol=artifact.symbol,
        market=artifact.market,
        dataset_type=artifact.dataset_type,
        source_date=artifact.source_date,
        timestamp_unit=timestamp_unit,
        schema=schema,
    )
    silver = build_silver_frame(bronze, artifact.dataset_type)
    bronze_path = write_partitioned_parquet(
        bronze,
        base_dir=config.paths.bronze_dir,
        dataset_type=artifact.dataset_type,
        symbol=artifact.symbol,
        source_date=artifact.source_date,
    )
    silver_path = write_partitioned_parquet(
        silver,
        base_dir=config.paths.silver_dir,
        dataset_type=artifact.dataset_type,
        symbol=artifact.symbol,
        source_date=artifact.source_date,
    )
    record = IngestionRecord(
        dataset_type=artifact.dataset_type,
        symbol=artifact.symbol,
        market=artifact.market,
        source_date=artifact.source_date,
        raw_path=artifact.path,
        bronze_path=bronze_path,
        silver_path=silver_path,
        row_count=silver.height,
        timestamp_unit=timestamp_unit,
        sha256=artifact.sha256,
        checksum_verified=artifact.checksum_verified,
        checksum_value=artifact.checksum_value,
        created_at=utc_now(),
    )
    manifest.upsert(record)
    LOGGER.info("Built bronze/silver for %s %s", artifact.symbol, artifact.source_date.isoformat())
    return record


def build_dataset_range(
    config: AppConfig,
    *,
    symbol: str,
    start_date: date,
    end_date: date,
    dataset_type: str | None = None,
) -> list[IngestionRecord]:
    manifest = ManifestRepository(config.paths.metadata_dir)
    artifacts = download_dataset_range(
        config,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        dataset_type=dataset_type,
    )
    records = [_build_single_dataset(config, artifact, manifest) for artifact in artifacts]
    return records


def load_silver_dataset(
    config: AppConfig,
    symbol: str,
    dataset_type: str | None = None,
) -> pl.DataFrame:
    resolved_dataset_type = dataset_type or config.dataset.dataset_type
    root = config.paths.silver_dir / resolved_dataset_type / f"symbol={symbol}"
    if not root.exists():
        raise FileNotFoundError(
            f"No silver dataset found for {symbol} dataset={resolved_dataset_type}: {root}"
        )
    parquet_files = sorted(root.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root}")
    frame = pl.concat([pl.read_parquet(path) for path in parquet_files], how="vertical")
    return frame.sort(["event_time", "trade_id"])


def load_silver_trades(config: AppConfig, symbol: str) -> pl.DataFrame:
    return load_silver_dataset(config, symbol=symbol, dataset_type="trades")
