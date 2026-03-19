from __future__ import annotations

import asyncio
import json
import logging

import pytest

from ofbot.config import GatewayConfig
from ofbot.gateway.binance_public_ws import BinancePublicWebSocket
from ofbot.market.trades import DepthEvent


def _depth_payload(stream: str = "btcusdt@depth@100ms") -> str:
    return json.dumps(
        {
            "stream": stream,
            "data": {
                "s": "BTCUSDT",
                "E": 1_700_000_000_000,
                "U": 100,
                "u": 101,
                "b": [["100.0", "2.0"]],
                "a": [["100.1", "1.5"]],
            },
        }
    )


class _FakeWebSocket:
    def __init__(self, messages: list[str]) -> None:
        self._messages = iter(messages)

    async def recv(self) -> str:
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise asyncio.TimeoutError() from exc


class _FakeConnect:
    def __init__(self, messages: list[str]) -> None:
        self._messages = messages

    async def __aenter__(self) -> _FakeWebSocket:
        return _FakeWebSocket(list(self._messages))

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


def test_decode_depth_100ms_stream_into_depth_event() -> None:
    client = BinancePublicWebSocket(symbols=["BTCUSDT"], config=GatewayConfig())

    event = client._decode(_depth_payload("btcusdt@depth@100ms"))

    assert isinstance(event.event, DepthEvent)
    assert event.event.event_type == "depth"
    assert event.event.final_update_id == 101


def test_build_url_reuses_stream_suffix_when_base_already_contains_stream() -> None:
    client = BinancePublicWebSocket(
        symbols=["BTCUSDT"],
        config=GatewayConfig(public_ws_base="wss://stream.binance.com:9443/stream"),
    )

    url = client._build_url()

    assert url.startswith("wss://stream.binance.com:9443/stream?streams=")
    assert "/stream/stream?" not in url


def test_build_url_adds_stream_suffix_when_base_is_host_only() -> None:
    client = BinancePublicWebSocket(
        symbols=["BTCUSDT"],
        config=GatewayConfig(public_ws_base="wss://stream.binance.com:9443"),
    )

    url = client._build_url()

    assert url.startswith("wss://stream.binance.com:9443/stream?streams=")


def test_decode_depth_stream_without_interval_into_depth_event() -> None:
    client = BinancePublicWebSocket(symbols=["BTCUSDT"], config=GatewayConfig())

    event = client._decode(_depth_payload("btcusdt@depth"))

    assert isinstance(event.event, DepthEvent)
    assert event.event.event_type == "depth"


def test_decode_bookticker_without_exchange_event_time_uses_received_time() -> None:
    client = BinancePublicWebSocket(symbols=["BTCUSDT"], config=GatewayConfig())
    payload = json.dumps(
        {
            "stream": "btcusdt@bookTicker",
            "data": {
                "s": "BTCUSDT",
                "b": "100.0",
                "B": "2.0",
                "a": "100.1",
                "A": "1.5",
            },
        }
    )

    event = client._decode(payload)

    assert event.event.event_type == "bookTicker"
    assert event.event.event_time == event.received_at
    assert event.event.event_time.year >= 2026


def test_decode_unknown_stream_raises_value_error() -> None:
    client = BinancePublicWebSocket(symbols=["BTCUSDT"], config=GatewayConfig())

    payload = json.dumps(
        {
            "stream": "btcusdt@miniTicker",
            "data": {"s": "BTCUSDT", "E": 1_700_000_000_000},
        }
    )

    with pytest.raises(ValueError, match="Unsupported websocket stream"):
        client._decode(payload)


def test_events_logs_decode_errors_and_keeps_stream_alive(monkeypatch, caplog) -> None:
    client = BinancePublicWebSocket(symbols=["BTCUSDT"], config=GatewayConfig())
    bad_payload = json.dumps(
        {
            "stream": "btcusdt@miniTicker",
            "data": {"s": "BTCUSDT", "E": 1_700_000_000_000},
        }
    )
    monkeypatch.setattr(
        "ofbot.gateway.binance_public_ws.websockets.connect",
        lambda *args, **kwargs: _FakeConnect([bad_payload, _depth_payload()]),
    )

    async def _collect_one() -> DepthEvent:
        stream = client.events()
        item = await asyncio.wait_for(anext(stream), timeout=1.0)
        await stream.aclose()
        return item.event

    with caplog.at_level(logging.INFO):
        event = asyncio.run(_collect_one())

    assert isinstance(event, DepthEvent)
    assert "ws_decode_error stream=btcusdt@miniTicker symbol=BTCUSDT" in caplog.text
    assert "ws_first_event symbol=BTCUSDT type=depth" in caplog.text
