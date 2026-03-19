from __future__ import annotations

import zipfile
from datetime import date

import polars as pl
import pytest
from quantflow.data_ingestion.pipeline import (
    build_bronze_frame,
    build_silver_frame,
    detect_timestamp_unit,
    parse_zip_csv,
)
from quantflow.data_ingestion.schemas import get_spot_schema


def test_parse_binance_trades_zip_and_normalize(app_config, tmp_path) -> None:
    zip_path = tmp_path / "BTCUSDT-trades-2025-01-01.zip"
    csv_rows = "\n".join(
        [
            "1,100.0,0.5,50.0,1735689600010866,True,True",
            "2,101.0,0.2,20.2,1735689601010866,False,True",
        ]
    )
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("BTCUSDT-trades-2025-01-01.csv", csv_rows)

    schema = get_spot_schema("trades")
    raw_frame = parse_zip_csv(zip_path, schema)
    assert raw_frame.height == 2
    assert detect_timestamp_unit(raw_frame.get_column("raw_time")) == "us"

    bronze = build_bronze_frame(
        raw_frame,
        symbol="BTCUSDT",
        market="spot",
        dataset_type="trades",
        source_date=date(2025, 1, 1),
        timestamp_unit="us",
        schema=schema,
    )
    silver = build_silver_frame(bronze, "trades")

    assert silver.height == 2
    assert silver.schema["event_time"] == pl.Datetime(time_unit="us", time_zone="UTC")
    assert silver["side_sign"].to_list() == [-1, 1]
    assert silver["signed_qty"].to_list() == [-0.5, 0.2]


def test_parse_binance_aggtrades_zip_and_normalize(app_config, tmp_path) -> None:
    zip_path = tmp_path / "BTCUSDT-aggTrades-2025-01-01.zip"
    csv_rows = "\n".join(
        [
            "10,100.0,0.5,1,2,1735689600010866,True,True",
            "11,101.0,0.2,3,3,1735689601010866,False,True",
        ]
    )
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("BTCUSDT-aggTrades-2025-01-01.csv", csv_rows)

    schema = get_spot_schema("aggTrades")
    raw_frame = parse_zip_csv(zip_path, schema)
    bronze = build_bronze_frame(
        raw_frame,
        symbol="BTCUSDT",
        market="spot",
        dataset_type="aggTrades",
        source_date=date(2025, 1, 1),
        timestamp_unit="us",
        schema=schema,
    )
    silver = build_silver_frame(bronze, "aggTrades")

    assert silver["trade_id"].to_list() == [10, 11]
    assert silver["quote_qty"].to_list() == pytest.approx([50.0, 20.2])
    assert silver["signed_quote_qty"].to_list() == pytest.approx([-50.0, 20.2])
    assert silver["first_trade_id"].to_list() == [1, 3]
