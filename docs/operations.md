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

## Automated alpha hunt

Run the combined offline hunt + paper loop:

```bash
uv run python -m ofbot.cli alpha-loop run \
  --config config/local.paper.yaml \
  --suite-config configs/order_flow_suite_spot_hunt.toml \
  --deep-dive-config configs/candidate_order_flow_hunt.toml \
  --live-run-seconds 120
```

Artifacts:

- runtime deployment configs under `data/memory/deployments/`
- runtime candidate memory in `data/memory/learning.duckdb`
- offline research candidate tracking in the same DuckDB
- synthetic hunt status in `docs/alpha_hunt_status.md`

Behavior:

- `qflow` and `ofbot` stay separate: offline validated edges are tracked, not auto-bridged into runtime
- runtime promotion mutates only generated configs under `data/memory/deployments/`
- the base file `config/local.paper.yaml` is never rewritten
- `active.paper.yaml` is the only config switched automatically for paper-local canaries

## What is intentionally not automated

- No auto-creation of new strategy families.
- No production key usage for order placement in paper mode.
- No cross-run memory edits.
- No automatic bridge from offline `qflow` candidates into runtime `ofbot` strategies.
