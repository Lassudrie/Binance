from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True, slots=True)
class BinanceDatasetSchema:
    dataset_type: str
    columns: tuple[str, ...]
    dtypes: dict[str, pl.DataType]
    timestamp_column: str


SPOT_SCHEMAS: dict[str, BinanceDatasetSchema] = {
    "trades": BinanceDatasetSchema(
        dataset_type="trades",
        columns=(
            "trade_id",
            "price",
            "qty",
            "quote_qty",
            "raw_time",
            "is_buyer_maker",
            "is_best_match",
        ),
        dtypes={
            "trade_id": pl.Int64,
            "price": pl.Float64,
            "qty": pl.Float64,
            "quote_qty": pl.Float64,
            "raw_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "is_best_match": pl.Boolean,
        },
        timestamp_column="raw_time",
    ),
    "aggTrades": BinanceDatasetSchema(
        dataset_type="aggTrades",
        columns=(
            "aggregate_trade_id",
            "price",
            "qty",
            "first_trade_id",
            "last_trade_id",
            "raw_time",
            "is_buyer_maker",
            "is_best_match",
        ),
        dtypes={
            "aggregate_trade_id": pl.Int64,
            "price": pl.Float64,
            "qty": pl.Float64,
            "first_trade_id": pl.Int64,
            "last_trade_id": pl.Int64,
            "raw_time": pl.Int64,
            "is_buyer_maker": pl.Boolean,
            "is_best_match": pl.Boolean,
        },
        timestamp_column="raw_time",
    ),
    "klines": BinanceDatasetSchema(
        dataset_type="klines",
        columns=(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ),
        dtypes={
            "open_time": pl.Int64,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "close_time": pl.Int64,
            "quote_asset_volume": pl.Float64,
            "number_of_trades": pl.Int64,
            "taker_buy_base_asset_volume": pl.Float64,
            "taker_buy_quote_asset_volume": pl.Float64,
            "ignore": pl.Float64,
        },
        timestamp_column="open_time",
    ),
}


def get_spot_schema(dataset_type: str) -> BinanceDatasetSchema:
    try:
        return SPOT_SCHEMAS[dataset_type]
    except KeyError as error:
        raise ValueError(f"Unsupported spot dataset type: {dataset_type}") from error
