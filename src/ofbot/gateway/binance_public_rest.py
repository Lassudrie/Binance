from __future__ import annotations

from datetime import UTC, datetime

import httpx

from ofbot.market.trades import DepthSnapshotEvent


class BinancePublicRestClient:
    """Synchronous public REST adapter used for local order book snapshots."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.binance.com/api",
        timeout_seconds: int = 20,
        depth_snapshot_limit: int = 1000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.depth_snapshot_limit = depth_snapshot_limit
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def get_depth_snapshot(
        self,
        symbol: str,
        *,
        event_time: datetime | None = None,
    ) -> DepthSnapshotEvent:
        response = self.client.get(
            "/v3/depth",
            params={"symbol": symbol.upper(), "limit": self.depth_snapshot_limit},
        )
        response.raise_for_status()
        payload = response.json()
        snapshot_time = event_time or datetime.now(UTC)
        payload["s"] = symbol.upper()
        return DepthSnapshotEvent(
            symbol=symbol.upper(),
            event_time=snapshot_time,
            event_type="depth_snapshot",
            raw=payload,
            last_update_id=int(payload["lastUpdateId"]),
            bids=[(float(price), float(qty)) for price, qty in payload.get("bids", [])],
            asks=[(float(price), float(qty)) for price, qty in payload.get("asks", [])],
        )
