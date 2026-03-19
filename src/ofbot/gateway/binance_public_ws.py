from __future__ import annotations

import asyncio
import json
import random
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime

import websockets

from ofbot.config import GatewayConfig
from ofbot.logging import get_logger
from ofbot.market.trades import (
    BookTickerEvent,
    DepthEvent,
    KlineEvent,
    MarketEvent,
    AggTradeEvent,
    parse_agg_trade,
    parse_book_ticker,
    parse_depth,
    parse_kline,
)


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class StreamEvent:
    event: MarketEvent
    received_at: datetime


class BinancePublicWebSocket:
    """Small resilient websocket client for Binance public market streams."""

    def __init__(self, *, symbols: list[str], config: GatewayConfig) -> None:
        self.symbols = [symbol.lower() for symbol in symbols]
        self.config = config
        self._ws = None
        self._connected = False
        self._recent_agg_trade_ids: dict[str, deque[int]] = {symbol: deque(maxlen=512) for symbol in self.symbols}
        self._recent_depth_updates: dict[str, deque[int]] = {symbol: deque(maxlen=512) for symbol in self.symbols}
        self._decoded_events = 0
        self._duplicate_events = 0
        self._seen_event_types: set[tuple[str, str]] = set()

    def _build_url(self) -> str:
        streams = [f"{symbol.lower()}@aggTrade" for symbol in self.symbols]
        streams.extend(f"{symbol.lower()}@bookTicker" for symbol in self.symbols)
        streams.extend([f"{symbol.lower()}@depth@{self.config.depth_stream}" for symbol in self.symbols])
        if self.config.include_kline_1m:
            streams.extend([f"{symbol.lower()}@kline_1m" for symbol in self.symbols])
        stream_query = "/".join(streams)
        base = self.config.public_ws_base.rstrip("/")
        if base.endswith("/stream"):
            return f"{base}?streams={stream_query}"
        return f"{base}/stream?streams={stream_query}"

    async def events(self) -> AsyncIterator[StreamEvent]:
        retry_delay = self.config.public_ws_reconnect_initial_delay_s
        while True:
            reconnect_delay = min(retry_delay, self.config.public_ws_reconnect_max_delay_s)
            try:
                async with websockets.connect(
                    self._build_url(),
                    max_queue=self.config.max_queue_size,
                    ping_interval=self.config.heartbeat_timeout_s,
                    ping_timeout=self.config.heartbeat_timeout_s * 2,
                ) as ws:
                    self._connected = True
                    LOGGER.info("ws_connected symbols=%s", ",".join(self.symbols))
                    retry_delay = self.config.public_ws_reconnect_initial_delay_s
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=self.config.heartbeat_timeout_s * 2)
                        try:
                            event = self._decode(msg)
                        except Exception as exc:
                            self._log_decode_error(payload=msg, exc=exc)
                            continue
                        if self._is_duplicate(event.event):
                            self._duplicate_events += 1
                            continue
                        self._decoded_events += 1
                        self._log_first_event(event.event)
                        self._log_stats_if_due()
                        yield event
            except asyncio.CancelledError:
                self._connected = False
                raise
            except asyncio.TimeoutError as exc:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    self._connected = False
                    raise
                self._connected = False
                LOGGER.warning("ws_timeout reconnect_in=%.2fs error=%s", reconnect_delay, exc)
            except Exception as exc:
                self._connected = False
                LOGGER.warning("ws_connection_error reconnect_in=%.2fs error=%s", reconnect_delay, exc)
            await asyncio.sleep(reconnect_delay)
            retry_delay = min(
                self.config.public_ws_reconnect_max_delay_s,
                retry_delay * 1.9 + random.uniform(0.05, 0.25),
            )

    def _decode(self, payload: str) -> StreamEvent:
        payload_json = json.loads(payload)
        if "stream" not in payload_json or "data" not in payload_json:
            raise ValueError("Malformed websocket payload")
        data = payload_json["data"]
        stream = str(payload_json.get("stream", ""))
        received_at = datetime.now(UTC)
        event_type = self._stream_kind(stream)
        parsed: MarketEvent
        if event_type == "aggTrade":
            parsed = parse_agg_trade(data)
        elif event_type == "bookTicker":
            parsed = parse_book_ticker(data, fallback_event_time=received_at)
        elif event_type == "depth":
            parsed = parse_depth(data)
        elif event_type == "kline_1m":
            parsed = parse_kline(data)
        else:
            raise ValueError(f"Unsupported websocket stream '{stream}'")
        parsed.raw = data
        return StreamEvent(event=parsed, received_at=received_at)

    @staticmethod
    def _stream_kind(stream: str) -> str:
        parts = stream.split("@")
        if len(parts) < 2 or not parts[1]:
            raise ValueError(f"Malformed websocket stream '{stream}'")
        return parts[1]

    def _log_decode_error(self, *, payload: str, exc: Exception) -> None:
        stream = "unknown"
        symbol = "unknown"
        try:
            payload_json = json.loads(payload)
            stream = str(payload_json.get("stream", stream))
            data = payload_json.get("data", {})
            if isinstance(data, dict):
                symbol = str(data.get("s", symbol))
        except Exception:
            pass
        LOGGER.error(
            "ws_decode_error stream=%s symbol=%s error=%s payload=%s",
            stream,
            symbol,
            exc,
            payload[:240],
        )

    def _log_first_event(self, event: MarketEvent) -> None:
        key = (event.symbol, event.event_type)
        if key in self._seen_event_types:
            return
        self._seen_event_types.add(key)
        LOGGER.info("ws_first_event symbol=%s type=%s", event.symbol, event.event_type)

    def _log_stats_if_due(self) -> None:
        total = self._decoded_events + self._duplicate_events
        if total == 0 or total % 1_000 != 0:
            return
        LOGGER.info("ws_stats decoded=%d duplicates=%d", self._decoded_events, self._duplicate_events)

    def _is_duplicate(self, event: MarketEvent) -> bool:
        symbol = event.symbol.lower()
        if isinstance(event, AggTradeEvent):
            seen = self._recent_agg_trade_ids.setdefault(symbol, deque(maxlen=512))
            if event.agg_trade_id in seen:
                return True
            seen.append(event.agg_trade_id)
            return False
        if isinstance(event, DepthEvent):
            seen = self._recent_depth_updates.setdefault(symbol, deque(maxlen=512))
            if event.final_update_id in seen:
                return True
            seen.append(event.final_update_id)
            return False
        return False

    @property
    def is_healthy(self) -> bool:
        return self._connected
