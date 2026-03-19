from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode

import httpx


class BinanceSpotTestnetClient:
    """Optional low-risk REST adapter for signed-endpoint smoke checks."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_secret: str | None = None,
        base_url: str = "https://testnet.binance.vision/api",
        timeout_seconds: int = 20,
        enabled: bool = False,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.enabled = enabled and bool(api_key and api_secret)
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout_seconds, headers=self._headers())

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["X-MBX-APIKEY"] = self.api_key
        return headers

    async def close(self) -> None:
        await self.client.aclose()

    async def ping(self) -> dict[str, str | int]:
        if not self.enabled:
            return {"status": "disabled"}
        response = await self.client.get("/v3/ping")
        return {
            "status": "ok" if response.status_code == 200 else "error",
            "status_code": response.status_code,
        }

    async def get_account(self) -> dict[str, object]:
        if not self.enabled:
            return {"enabled": False}
        params = self._signed_params({})
        response = await self.client.get("/api/v3/account", params=params)
        return response.json()

    def _signed_params(self, params: dict[str, str | int | float]) -> dict[str, str]:
        if not self.enabled:
            return {}
        payload = {**params, "timestamp": int(time.time() * 1000)}
        query = urlencode(payload)
        secret = (self.api_secret or "").encode()
        signature = hmac.new(secret, query.encode(), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return {str(k): str(v) for k, v in payload.items()}
