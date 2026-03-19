from __future__ import annotations

import asyncio
from pathlib import Path

from ofbot.cli._main import _run_live, _run_paper_order
from ofbot.utils.manual_orders import iter_manual_order_requests
from tests.ofbot_helpers import build_test_config


class _IdleWebSocket:
    def __init__(self, *args, **kwargs) -> None:
        return

    async def events(self):
        while True:
            await asyncio.sleep(60)
            yield


class _FakeEngine:
    def __init__(self, config, *, run_id: str | None = None) -> None:
        self.config = config
        self.run_id = run_id or "test_run"
        self.processed = 0
        self.finalized = False
        self.closed = False
        self.received_at = None

    def process_event(self, event, *, received_at=None) -> None:
        self.processed += 1
        self.received_at = received_at

    def finalize(self) -> Path:
        self.finalized = True
        return self.config.paths.reports_dir / "report.md"

    def close(self) -> None:
        self.closed = True


def test_run_live_honors_max_seconds_without_events(monkeypatch, tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    fake_engine = _FakeEngine(config)

    monkeypatch.setattr("ofbot.cli._main.load_config", lambda path: config)
    monkeypatch.setattr("ofbot.cli._main.BinancePublicRestClient", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        "ofbot.cli._main.PaperEngine",
        lambda config, depth_snapshot_client=None: fake_engine,
    )
    monkeypatch.setattr("ofbot.cli._main.BinancePublicWebSocket", _IdleWebSocket)

    asyncio.run(_run_live("ignored.yaml", max_events=0, max_seconds=1, symbols=None))

    assert fake_engine.processed == 0
    assert fake_engine.finalized is True
    assert fake_engine.closed is True


def test_run_paper_order_writes_manual_request(monkeypatch, tmp_path) -> None:
    config = build_test_config(tmp_path, symbols=["BTCUSDT"], mode="paper_local")
    monkeypatch.setattr("ofbot.cli._main.load_config", lambda path: config)

    _run_paper_order(
        "ignored.yaml",
        symbol="BTCUSDT",
        side="buy",
        qty=0.25,
        notional_usd=0.0,
        order_type="market",
        reason="manual_test",
        force=False,
    )

    requests = iter_manual_order_requests(config.paths.memory_dir, symbol="BTCUSDT")
    assert len(requests) == 1
    path, request = requests[0]
    assert path.exists()
    assert request.symbol == "BTCUSDT"
    assert request.side == 1
    assert request.qty == 0.25
    assert request.reason == "manual_test"
