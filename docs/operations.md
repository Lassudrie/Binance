# ofbot Operations

## Safe defaults

- Default mode is `paper_local`.
- No signed Binance order endpoints are called unless `mode=paper_testnet` and both keys are present.
- Risk defaults are conservative (small max notional, max holding, spread/vol gates, cooldowns).
- Fee and slippage are applied to all fills.

## Daily workflow

1. `uv sync`
2. `uv run python -m ofbot.cli live --config config/local.paper.yaml`
3. Interrupt manually or with `--max-events / --max-seconds`
4. Review:
   - raw parquet under `data/raw`
   - reports under `data/reports/<run_id>/`
   - memory DB under `data/memory/learning.duckdb`

## What is intentionally not automated

- No automatic config mutation.
- No auto-creation of new strategy families.
- No production key usage for order placement in paper mode.
- No cross-run memory edits.
