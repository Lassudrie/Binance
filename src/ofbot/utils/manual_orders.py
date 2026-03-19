from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4


@dataclass(slots=True)
class ManualOrderRequest:
    request_id: str
    created_at: datetime
    symbol: str
    side: int
    order_type: str
    qty: float | None = None
    notional_usd: float | None = None
    reason: str = "manual_paper_order"
    force: bool = False

    def to_payload(self) -> dict[str, str | float | bool | None]:
        return {
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "qty": self.qty,
            "notional_usd": self.notional_usd,
            "reason": self.reason,
            "force": self.force,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "ManualOrderRequest":
        created_raw = payload.get("created_at")
        if not isinstance(created_raw, str):
            raise ValueError("manual order missing created_at")
        created_at = datetime.fromisoformat(created_raw)
        symbol = payload.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            raise ValueError("manual order missing symbol")
        side = int(payload.get("side", 0))
        order_type = str(payload.get("order_type", "market")).lower()
        qty = payload.get("qty")
        notional_usd = payload.get("notional_usd")
        return cls(
            request_id=str(payload.get("request_id", uuid4().hex)),
            created_at=created_at,
            symbol=symbol.upper(),
            side=side,
            order_type=order_type if order_type in {"market", "limit"} else "market",
            qty=float(qty) if isinstance(qty, (int, float)) else None,
            notional_usd=float(notional_usd) if isinstance(notional_usd, (int, float)) else None,
            reason=str(payload.get("reason", "manual_paper_order")),
            force=bool(payload.get("force", False)),
        )


def control_dir(memory_dir: Path) -> Path:
    return memory_dir / "control" / "manual_orders"


def write_manual_order_request(
    memory_dir: Path,
    *,
    symbol: str,
    side: int,
    order_type: str = "market",
    qty: float | None = None,
    notional_usd: float | None = None,
    reason: str = "manual_paper_order",
    force: bool = False,
) -> Path:
    request = ManualOrderRequest(
        request_id=uuid4().hex,
        created_at=datetime.now(UTC),
        symbol=symbol.upper(),
        side=side,
        order_type=order_type.lower(),
        qty=qty,
        notional_usd=notional_usd,
        reason=reason,
        force=force,
    )
    inbox = control_dir(memory_dir)
    inbox.mkdir(parents=True, exist_ok=True)
    target = inbox / f"{request.created_at.strftime('%Y%m%dT%H%M%S%f')}_{request.request_id}.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(request.to_payload(), separators=(",", ":")))
    tmp.replace(target)
    return target


def iter_manual_order_requests(memory_dir: Path, *, symbol: str | None = None) -> list[tuple[Path, ManualOrderRequest]]:
    inbox = control_dir(memory_dir)
    if not inbox.exists():
        return []

    requests: list[tuple[Path, ManualOrderRequest]] = []
    for path in sorted(inbox.glob("*.json")):
        payload = json.loads(path.read_text())
        request = ManualOrderRequest.from_payload(payload)
        if symbol is not None and request.symbol != symbol.upper():
            continue
        requests.append((path, request))
    return requests
